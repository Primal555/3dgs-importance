"""MesonGS sender and archive-only receiver for the pinned research baseline.

Encoding uses the unmodified upstream CUDA implementation.  The receiver uses
its RAHT tree builder and the same inverse transform on CPU, then exports the
decoded Euler rotations as the w,x,y,z quaternions required by standard 3DGS.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np


UPSTREAM_REVISION = "e491557a39e480105ae944eb2481d141808efbef"
ARCHIVE_FIELDS = {
    "oct.npz": {"points", "params"}, "ntk.npz": {"ntk"},
    "um.npz": {"umap"}, "orgb.npz": {"f", "i"},
    "ct.npz": {"i"}, "t.npz": {"t"},
}
MAX_ARCHIVE_BYTES = 8 * 1024 ** 3


def _verify_revision(repo: Path) -> str:
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"MesonGS revision must be {UPSTREAM_REVISION}, got {revision}")
    return revision


def load_config(repo: Path, config: str, scene: str) -> dict:
    if config not in ("c1", "c3"):
        raise ValueError("config must be c1 or c3")
    path = Path(repo) / "configs" / (config + ".json")
    values = json.loads(path.read_text(encoding="utf-8"))
    try:
        result = {"percent": float(values["prune"][scene]),
                  "codebook_size": int(values["cb"][scene]),
                  "depth": int(values["depth"][scene]),
                  "n_block": int(values["n_block"][scene])}
    except KeyError as error:
        raise ValueError(f"官方 {config}.json 没有场景 {scene!r} 的配置") from error
    if not 0 <= result["percent"] < 1 or not 1 <= result["depth"] <= 21:
        raise ValueError("invalid upstream prune/depth configuration")
    if result["codebook_size"] <= 0 or result["n_block"] <= 0:
        raise ValueError("invalid upstream codebook/block configuration")
    return result


def pruning_mask(percent: float, importance) -> np.ndarray:
    importance = np.asarray(importance).reshape(-1)
    if not 0 <= percent < 1 or not importance.size or not np.isfinite(importance).all():
        raise ValueError("invalid importance or pruning percentage")
    if percent == 0:
        return np.zeros(importance.size, dtype=bool)
    threshold = np.sort(importance)[int(percent * (importance.size - 1))]
    return importance <= threshold


def _safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    if len(path.parts) != 1 or name not in ARCHIVE_FIELDS:
        raise ValueError(f"unsupported MesonGS archive member: {name!r}")
    return name


def read_archive(archive: Path) -> dict:
    """Read the six-file upstream block-quantized format without extraction."""
    arrays = {}
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        names = [_safe_member(member.filename) for member in members]
        if len(names) != len(set(names)) or set(names) != set(ARCHIVE_FIELDS):
            raise ValueError("MesonGS archive must contain exactly six distinct NPZ members")
        if sum(member.file_size for member in members) > MAX_ARCHIVE_BYTES:
            raise ValueError("MesonGS archive exceeds unpacked size limit")
        for member in members:
            if member.flag_bits & 1 or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("encrypted or symbolic-link archive member")
            buffer = io.BytesIO(stream.read(member))
            # NPZ is itself a ZIP; validate its names and total expansion first.
            with zipfile.ZipFile(buffer) as inner:
                expected = {key + ".npy" for key in ARCHIVE_FIELDS[member.filename]}
                inner_members = inner.infolist()
                if len(inner_members) != len(expected) or set(inner.namelist()) != expected:
                    raise ValueError(f"unexpected NPZ fields in {member.filename}")
                if sum(entry.file_size for entry in inner_members) > MAX_ARCHIVE_BYTES:
                    raise ValueError("NPZ exceeds unpacked size limit")
            buffer.seek(0)
            with np.load(buffer, allow_pickle=False) as loaded:
                arrays[member.filename] = {key: loaded[key] for key in ARCHIVE_FIELDS[member.filename]}
    _validate_arrays(arrays)
    return arrays


def _validate_arrays(arrays: dict) -> None:
    for group in arrays.values():
        for array in group.values():
            if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
                raise ValueError("nonfinite or nonnumeric MesonGS data")
    trans = arrays["t.npz"]["t"]
    if trans.ndim != 1 or trans.size < 2:
        raise ValueError("invalid quantization header")
    depth, blocks = int(trans[0]), int(trans[1])
    if trans[0] != depth or trans[1] != blocks or not 1 <= depth <= 21 or blocks <= 0:
        raise ValueError("invalid depth/block count")
    if trans.size != 2 + 20 * blocks:
        raise ValueError("unsupported quantization layout; expected block-quantized save_npz")
    pairs = trans[2:].reshape(-1, 2)
    if (pairs[:, 0] <= 0).any() or (pairs[:, 1] < 0).any() or (pairs[:, 1] > 255).any():
        raise ValueError("invalid quantization scale or zero point")
    octree, bounds = arrays["oct.npz"]["points"], arrays["oct.npz"]["params"]
    if octree.ndim != 1 or octree.dtype.kind not in "iu" or octree.size < 2:
        raise ValueError("invalid octree")
    # searchsorted(..., side='right') gives the upper endpoint index 2**depth.
    # Its packed code can exceed 2**(3*depth); y/z endpoint carries are retained.
    grid = 2 ** depth
    largest = min(grid ** 3 + grid ** 2 + grid, np.iinfo(np.int64).max)
    if (octree < 0).any() or (octree > largest).any() or np.unique(octree).size != octree.size:
        raise ValueError("octree codes are duplicated or outside the declared depth")
    if bounds.shape != (6,) or (bounds[1::2] < bounds[::2]).any():
        raise ValueError("invalid octree bounds")
    _octree_voxels(octree, depth)
    count = octree.size
    if count - 1 < blocks:
        raise ValueError("point count must exceed quantization block count")
    indices, codebook = arrays["ntk.npz"]["ntk"], arrays["um.npz"]["umap"]
    if indices.shape != (count,) or indices.dtype.kind not in "iu":
        raise ValueError("invalid VQ index shape/type")
    if codebook.ndim != 2 or codebook.shape[0] == 0 or codebook.shape[1] != 45:
        raise ValueError("expected degree-3 SH codebook with 45 coefficients")
    if (indices < 0).any() or (indices >= codebook.shape[0]).any():
        raise ValueError("VQ index outside codebook")
    if arrays["orgb.npz"]["f"].shape != (7,):
        raise ValueError("invalid RAHT DC shape")
    for array, size in ((arrays["orgb.npz"]["i"], 7 * (count - 1)),
                        (arrays["ct.npz"]["i"], 3 * count)):
        if array.size != size or array.dtype != np.uint8 or array.ndim not in (1, 2):
            raise ValueError("invalid quantized attribute shape/type")


def _octree_voxels(octree, depth):
    # Match GaussianModel.decode_oct's division and truncation exactly. Bit
    # shifts differ near large-code float64 rounding boundaries at depth 20.
    codes = np.asarray(octree).astype(np.int64)
    grid = 2 ** depth
    x = (codes / (grid ** 2)).astype(np.int64)
    remainder = codes - x * (grid ** 2)
    y = (remainder / grid).astype(np.int64)
    z = (remainder - y * grid).astype(np.int64)
    voxel = np.stack([x, y, z], -1)
    # Upstream indexes linspace directly, including valid negative indices
    # caused by its float64 rounding. Check the actual indexing bounds.
    if (voxel < -(grid + 1)).any() or (voxel > grid).any():
        raise ValueError("octree coordinate lies outside upstream decode_oct's grid")
    return voxel


def _raht_module(repo: Path):
    path = Path(repo).resolve() / "raht_torch.py"
    spec = importlib.util.spec_from_file_location("_mesongs_receiver_raht", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _signed_raht_counts(module)
    return module


def _signed_raht_counts(module):
    # NumPy 2 subtracts unsigned counts using unsigned arithmetic. Upstream
    # inverse RAHT needs negative AC offsets; signed counts preserve its math.
    original = module.get_RAHT_tree

    def get_tree(*args, **kwargs):
        tree = original(*args, **kwargs)
        tree["iM"] = tree["iM"].astype(np.int64)
        return tree

    module.get_RAHT_tree = get_tree


def _dequantize(values, pairs, blocks):
    length = values.shape[0]
    sizes = [length // blocks + (i < length % blocks) for i in range(blocks)]
    result = np.empty_like(values, dtype=np.float32)
    for channel in range(values.shape[1]):
        offset = 0
        for block, size in enumerate(sizes):
            scale, zero = pairs[channel * blocks + block]
            result[offset:offset + size, channel] = (
                values[offset:offset + size, channel].astype(np.float32) - zero) * scale
            offset += size
    return result


def euler_to_quaternion(euler):
    # Upstream stores roll,pitch,yaw; its renderer uses Rz(yaw) Ry(pitch) Rx(roll).
    roll, pitch, yaw = (np.asarray(euler, dtype=np.float64) / 2).T
    cr, cp, cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr, sp, sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    result = np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                       cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], -1)
    return (result / np.linalg.norm(result, axis=1, keepdims=True)).astype(np.float32)


def decode(*, archive: Path, ply: Path, repo: Path, device: str = "cpu") -> dict:
    """Recover all Gaussian attributes using only the received archive."""
    arrays = read_archive(Path(archive))
    _verify_revision(Path(repo).resolve())
    trans = arrays["t.npz"]["t"]
    depth, blocks = int(trans[0]), int(trans[1])
    octree = arrays["oct.npz"]["points"]
    voxel = _octree_voxels(octree, depth)
    bounds = arrays["oct.npz"]["params"]
    xyz = np.stack([np.linspace(bounds[2 * axis], bounds[2 * axis + 1], 2 ** depth + 1)[voxel[:, axis]]
                    for axis in range(3)], -1)
    raht = _raht_module(Path(repo))
    _, _, reorder = raht.copyAsort(voxel)
    tree = raht.inv_haar3D_param(voxel, depth)
    pairs = trans[2:].reshape(-1, 2)
    ac = arrays["orgb.npz"]["i"].reshape(7, -1).T
    restored = _dequantize(ac, pairs[:7 * blocks], blocks)
    coefficients = np.concatenate([arrays["orgb.npz"]["f"].reshape(1, 7), restored], 0).astype(np.float32)
    temporary = coefficients[tree["pos"].astype(int)].copy()
    output = np.zeros_like(coefficients)
    # This is load_npz's inverse RAHT, with CPU arrays replacing CUDA tensors.
    for level in range(depth * 3):
        output[tree["iTrans_idx"][level]] = temporary[tree["iTrans_idx_CT"][level]]
        left = temporary[tree["iLeft_idx_CT"][level]]
        right = temporary[tree["iRight_idx_CT"][level]]
        w1 = np.asarray(tree["iW1"][level], np.float32)[:, None]
        w2 = np.asarray(tree["iW2"][level], np.float32)[:, None]
        output[tree["iLeft_idx"][level]] = w1 * left - w2 * right
        output[tree["iRight_idx"][level]] = w2 * left + w1 * right
        size = tree["iS"][level]
        temporary[:size] = output[:size]
    features = np.empty_like(output)
    features[reorder] = output
    scale = _dequantize(arrays["ct.npz"]["i"].reshape(3, -1).T, pairs[7 * blocks:], blocks)
    rest = arrays["um.npz"]["umap"][arrays["ntk.npz"]["ntk"]].reshape(-1, 15, 3)
    attributes = np.concatenate([xyz, np.zeros_like(xyz), features[:, 4:],
                                 rest.transpose(0, 2, 1).reshape(-1, 45), features[:, :1],
                                 scale, euler_to_quaternion(features[:, 1:4])], -1).astype(np.float32)
    if not np.isfinite(attributes).all():
        raise ValueError("decoded Gaussian attributes contain nonfinite values")
    from plyfile import PlyData, PlyElement
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(45)] + ["opacity"]
    names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    vertex = np.empty(octree.size, dtype=[(name, "<f4") for name in names])
    for index, name in enumerate(names):
        vertex[name] = attributes[:, index]
    destination = Path(ply).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Atomic installation ensures a failed decode never leaves a partial PLY.
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".ply", delete=False) as temporary_file:
        staged = Path(temporary_file.name)
    try:
        PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(staged))
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    return {"point_count": int(octree.size), "ply": str(destination), "decoder_device": "cpu"}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ArchiveOS:
    """Replace only upstream's shell ZIP packaging, preserving its codec."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def __getattr__(self, name):
        return getattr(os, name)

    def system(self, command):
        bins = (self.directory / "bins").resolve()
        if bins.parent != self.directory or set(p.name for p in bins.iterdir()) != set(ARCHIVE_FIELDS):
            raise ValueError("unexpected upstream codec output")
        with zipfile.ZipFile(self.directory / "bins.zip", "w", compression=zipfile.ZIP_DEFLATED) as stream:
            for name in sorted(ARCHIVE_FIELDS):
                stream.write(bins / name, arcname=name)
        return 0


def encode(*, source: Path, ply: Path, out: Path, repo: Path, config="c1",
           iterations=0, resolution=2, images="images", device="cuda",
           white_background=False) -> dict:
    source, ply, out, repo = (Path(path).resolve() for path in (source, ply, out, repo))
    if not source.is_dir() or not ply.is_file():
        raise ValueError("source directory and input PLY must exist")
    import torch
    selected_device = torch.device(device)
    if iterations < 0 or (resolution <= 0 and resolution != -1) or selected_device.type != "cuda":
        raise ValueError("MesonGS encoding requires CUDA, iterations>=0 and positive resolution or -1")
    scene_name = source.name
    selected = load_config(repo, config, scene_name)
    revision = _verify_revision(repo)
    if not torch.cuda.is_available():
        raise RuntimeError("MesonGS encoding requires the dedicated CUDA environment")
    if selected_device.index is not None:
        torch.cuda.set_device(selected_device)
    # The CLI runs in its own process. Reject module contamination explicitly.
    for name in ("scene", "utils", "arguments", "gaussian_renderer", "mesongs"):
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "__file__", None):
            if not Path(existing.__file__).resolve().is_relative_to(repo):
                raise RuntimeError(f"foreign module {name!r}; run this adapter in a fresh Python process")
    sys.path.insert(0, str(repo))
    import mesongs as upstream
    import scene.gaussian_model as gaussian_module
    import raht_torch
    _signed_raht_counts(raht_torch)
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import GaussianModel, Scene
    from gaussian_renderer import ft_render
    from utils.loss_utils import l1_loss, ssim
    parser = argparse.ArgumentParser()
    model_params, optimization, pipeline_params = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    arguments = parser.parse_args([])
    dataset, opt, pipe = model_params.extract(arguments), optimization.extract(arguments), pipeline_params.extract(arguments)
    for key, value in selected.items():
        setattr(dataset, key, value)
    dataset.source_path, dataset.images, dataset.resolution = str(source), images, resolution
    dataset.eval = True
    dataset.white_background = bool(white_background)
    pipe.scene_imp, pipe.n_block = scene_name, selected["n_block"]
    pipe.save_ft_type = ""
    opt.iterations = iterations
    opt.finetune_lr_scale = upstream.lr_scale_list[scene_name] * 4
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = {"input_sha256": _sha256(ply), "input_ply": str(ply), "source": str(source),
                "config": config, "config_values": selected, "upstream_revision": revision,
                "iterations": iterations, "resolution": resolution, "images": images,
                "device": device, "seed": 0, "white_background": dataset.white_background,
                "adaptations": ["official JSON mapping", "zero-percent prune preserves every point",
                                "Python ZIP packaging", "sender image evaluation omitted",
                                "final iteration archive", "single-view quantizer initialization"]}
    metadata["adaptations"].append("signed RAHT counts for NumPy 2 compatibility")
    metadata["adaptations"].append("receiver reuses official RAHT tree; identical inverse equations run on CPU")
    metadata["decoder_provenance"] = {
        "implementation": "official raht_torch.inv_haar3D_param tree with float32 NumPy inverse equations from GaussianModel.load_npz",
        "format": "save_npz-six-files", "device": "cpu",
        "rotation_export": "decoded roll,pitch,yaw converted to standard normalized w,x,y,z quaternion",
        "geometry_decode": "upstream decode_oct float64 division/truncation and linspace indexing, including packed endpoint carries",
        "equivalence_validation": "tests/test_comparison_mesongs_adapter.py::MesonGSAdapterTests::test_inverse_matches_official_forward_tree_for_many_leaves_and_blocks",
        "validation_method": "official forward RAHT tree on basis vectors; quantized coefficients recovered against transpose of the orthogonal forward matrix",
        "endpoint_validation": "official octreecodes/decode_oct AST execution at depths 4 and 20, including upper endpoints and float64 rounding boundaries",
        "serialization_validation": "official save_npz AST execution on CPU with device placement redirected; actual six-file NPZ fields, shapes and dtypes checked",
    }
    metadata["codec_parameters"] = {"sh_degree": dataset.sh_degree, "num_bits": dataset.num_bits,
                                    "vq_steps": dataset.steps, "vq_batch_size": dataset.batch_size,
                                    "oct_merge": dataset.oct_merge, "raht": dataset.raht,
                                    "per_block_quant": dataset.per_block_quant,
                                    "format": "save_npz-six-files", "finetune_lr_scale": opt.finetune_lr_scale,
                                    "white_background": dataset.white_background, "eval": dataset.eval}
    with open(out / "encode.log", "w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        with tempfile.TemporaryDirectory(prefix="mesongs-", dir=out) as scratch_name:
            scratch = Path(scratch_name).resolve()
            if scratch.parent != out or not scratch.is_relative_to(out):
                raise ValueError("scratch path must be strictly inside the requested output directory")
            dataset.model_path = str(scratch)
            gaussians = GaussianModel(dataset.sh_degree, depth=dataset.depth, num_bits=dataset.num_bits)
            scene = Scene(dataset, gaussians, given_ply_path=str(ply), shuffle=False)
            gaussians.training_setup(opt)
            background = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.], device="cuda")
            cameras = scene.getTrainCameras()
            if not cameras:
                raise ValueError("scene has no training cameras")
            metadata["input_point_count"] = int(gaussians.get_xyz.shape[0])
            with torch.no_grad():
                importance = upstream.cal_imp(gaussians, cameras, pipe, background)
                mask = torch.as_tensor(pruning_mask(dataset.percent, importance.cpu().numpy()), device=importance.device)
                importance = importance[~mask]
                gaussians.prune_points(mask)
                if gaussians.get_xyz.shape[0] <= dataset.n_block:
                    raise ValueError("too few points for the official quantization block count")
                gaussians.octree_coding(importance, dataset.oct_merge, raht=dataset.raht)
                if gaussians.get_xyz.shape[0] <= dataset.n_block:
                    raise ValueError("octree merging left too few points for block quantization")
                gaussians.init_qas(dataset.n_block)
                gaussians.vq_fe(importance, dataset.codebook_size, dataset.batch_size, dataset.steps)
                # ft_render initializes official quantizer buffers for all attributes.
                ft_render(cameras[0], gaussians, pipe, background, training=False, raht=True,
                          per_channel_quant=False, per_block_quant=True, clamp_color=True)
            if iterations:
                gaussians.finetuning_setup(opt)
                camera_stack = []
                for iteration in range(1, iterations + 1):
                    if not camera_stack:
                        camera_stack = cameras.copy()
                    camera = camera_stack.pop(random.randrange(len(camera_stack)))
                    result = ft_render(camera, gaussians, pipe, background, training=True, raht=True,
                                       per_channel_quant=False, per_block_quant=True, clamp_color=True)
                    target = camera.original_image.cuda()
                    loss = (1 - opt.lambda_dssim) * l1_loss(result["render"], target)
                    loss += opt.lambda_dssim * (1 - ssim(result["render"], target))
                    loss.backward()
                    with torch.no_grad():
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none=True)
                        gaussians.update_learning_rate(iteration + 30000)
            with torch.no_grad():
                # Refresh quantizer ranges after the last optimizer step.
                ft_render(cameras[0], gaussians, pipe, background, training=False, raht=True,
                          per_channel_quant=False, per_block_quant=True, clamp_color=True)
                codec_output = scratch / "codec"
                original_os = gaussian_module.os
                gaussian_module.os = _ArchiveOS(codec_output)
                try:
                    gaussians.save_npz(str(codec_output), pipe, per_channel_quant=False, per_block_quant=True)
                finally:
                    gaussian_module.os = original_os
                read_archive(codec_output / "bins.zip")
                os.replace(codec_output / "bins.zip", out / "bins.zip")
            metadata["point_count"] = metadata["actual_point_count"] = int(gaussians.get_xyz.shape[0])
    metadata["archive_bytes"] = (out / "bins.zip").stat().st_size
    metadata["elapsed_seconds"] = time.monotonic() - started
    (out / "encode.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sender = commands.add_parser("encode")
    for flag in ("source", "ply", "out", "repo"):
        sender.add_argument("--" + flag, type=Path, required=True)
    sender.add_argument("--config", choices=("c1", "c3"), default="c1")
    sender.add_argument("--iterations", type=int, default=0)
    sender.add_argument("--resolution", type=int, default=2)
    sender.add_argument("--images", default="images")
    sender.add_argument("--device", default="cuda")
    sender.add_argument("--white-background", action=argparse.BooleanOptionalAction, default=False)
    receiver = commands.add_parser("decode")
    for flag in ("archive", "ply", "repo"):
        receiver.add_argument("--" + flag, type=Path, required=True)
    receiver.add_argument("--device", default="cpu")
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    result = encode(**arguments) if command == "encode" else decode(**arguments)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
