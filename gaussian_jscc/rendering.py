"""Differentiable rendering adapter; imported only for CUDA image training/evaluation."""

from pathlib import Path
from types import SimpleNamespace
import math

import torch


class RenderReference:
    """Frozen source-PLY teacher, lazily rendered once per camera on the CPU cache.

    Never sent to the receiver. Retains no source render autograd graph and no
    permanently resident full-scene CUDA copy. Images mode is explicit opt-in.
    """

    def __init__(self, raw, degree, white_background=False, target="source"):
        if target not in ("source", "images"):
            raise ValueError("unknown rendering target")
        self.raw = raw.detach().cpu()
        self.degree, self.white_background, self.target = degree, white_background, target
        self.cache = {}

    @torch.no_grad()
    def get(self, camera, device):
        if self.target == "images":
            return camera.original_image[:3].detach().to(device)
        key = id(camera)
        if key not in self.cache:
            self.cache[key] = render(self.raw.to(device), camera, self.degree,
                                     self.white_background).detach().cpu()
        return self.cache[key].to(device)


HYBRID_VARIANT_DEFINITIONS = {
    "reference": "source xyz and source non-position attributes",
    "seed_position_error_only": "decoder position-seed xyz and source non-position attributes",
    "position_error_only": "decoded xyz and source non-position attributes",
    "attribute_error_only": "source xyz and decoded non-position attributes",
    "received": "decoded xyz and decoded non-position attributes",
}


def hybrid_parameter_scenes(received, reference, position_seed=None):
    """Build row-aligned scenes that isolate position and attribute errors."""
    if received.shape != reference.shape or received.ndim != 2 or received.shape[1] < 4:
        raise ValueError("received and reference must be matching [N,D] Gaussian tensors")
    if position_seed is not None and position_seed.shape != (len(reference), 3):
        raise ValueError("position_seed must be a matching [N,3] tensor")
    scenes = {"reference": reference}
    if position_seed is not None:
        scenes["seed_position_error_only"] = torch.cat((position_seed, reference[:, 3:]), -1)
    scenes.update({
        "position_error_only": torch.cat((received[:, :3], reference[:, 3:]), -1),
        "attribute_error_only": torch.cat((reference[:, :3], received[:, 3:]), -1),
        "received": received,
    })
    return scenes


def load_cameras(source, resolution=2, white_background=False, images="images", split="test"):
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from utils.camera_utils import cameraList_from_camInfos

    source = str(Path(source).resolve())
    if Path(source, "sparse").exists():
        info = sceneLoadTypeCallbacks["Colmap"](source, images, True)
    elif Path(source, "transforms_train.json").exists():
        info = sceneLoadTypeCallbacks["Blender"](source, white_background, True)
    else:
        raise ValueError("source must be a COLMAP or Blender scene")
    infos = info.train_cameras if split == "train" else info.test_cameras
    if not infos:
        raise ValueError(f"scene has no {split} cameras")
    args = SimpleNamespace(resolution=resolution, data_device="cpu")
    return cameraList_from_camInfos(infos, 1.0, args)


def render(raw, camera, degree, white_background=False, existence=None):
    if existence is None:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    else:
        # Unlike multiplying opacity before the vanilla rasterizer's cutoff,
        # MaskGaussian's kernel also computes mask gradients for inactive splats.
        from mask_diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        if existence.shape != (len(raw),):
            raise ValueError("existence mask must have shape [N]")

    bg = raw.new_full((3,), float(white_background))
    if not len(raw):
        return bg[:, None, None].expand(3, camera.image_height, camera.image_width)
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
        bg=bg, scale_modifier=1., viewmatrix=camera.world_view_transform.to(raw.device),
        projmatrix=camera.full_proj_transform.to(raw.device), sh_degree=degree,
        campos=camera.camera_center.to(raw.device), prefiltered=False, debug=False)
    dc = raw[:, 11:14].reshape(-1, 1, 3)
    rest = raw[:, 14:].reshape(len(raw), 3, (degree + 1) ** 2 - 1).transpose(1, 2)
    sh = torch.cat((dc, rest), 1).contiguous()
    image, _ = GaussianRasterizer(raster_settings=settings)(
        means3D=raw[:, :3].contiguous(), means2D=torch.zeros_like(raw[:, :3]),
        shs=sh, colors_precomp=None, opacities=raw[:, 3:4].sigmoid().contiguous(),
        scales=raw[:, 4:7].clamp(-20, 10).exp().contiguous(),
        rotations=torch.nn.functional.normalize(raw[:, 7:11], dim=-1).contiguous(),
        cov3D_precomp=None,
        **({} if existence is None else {"masks": existence[:, None].contiguous()}))
    return image


@torch.no_grad()
def evaluate_views(raw, reference, cameras, degree, white_background, directory=None, lpips=False,
                   hybrid_ablation=False, position_seed=None):
    """Evaluate decoded Gaussians and optionally isolate position/attribute errors.

    Hybrid construction assumes ``raw`` and ``reference`` contain the same rows in
    the same order. Rendering itself is permutation invariant, but mixing columns
    from unmatched rows is not.
    """
    from utils.loss_utils import ssim
    from PIL import Image
    import numpy as np

    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
    perceptual = None
    if lpips:
        from lpipsPyTorch.modules.lpips import LPIPS
        perceptual = LPIPS(net_type="vgg").to(raw.device).eval()
    if position_seed is not None:
        hybrid_ablation = True
    scenes = (hybrid_parameter_scenes(raw, reference, position_seed) if hybrid_ablation else
              {"received": raw, "reference": reference})
    rows = []
    for index, camera in enumerate(cameras):
        images = {name: render(scene, camera, degree, white_background).clamp(0, 1)
                  for name, scene in scenes.items()}
        baseline = images["reference"]
        gt = camera.original_image[:3].to(raw.device)
        row = {"view": camera.image_name}
        for name, image in images.items():
            mse = (image - gt).square().mean().clamp_min(1e-12)
            row[name + "_psnr"] = float(-10 * mse.log10())
            row[name + "_ssim"] = float(ssim(image, gt))
            if perceptual is not None:
                row[name + "_lpips"] = float(perceptual(image[None] * 2 - 1, gt[None] * 2 - 1))
        for name, image in images.items():
            if name == "reference":
                continue
            prefix = name + "_vs_reference"
            codec_mse = (image - baseline).square().mean().clamp_min(1e-12)
            row[prefix + "_psnr"] = float(-10 * codec_mse.log10())
            row[prefix + "_ssim"] = float(ssim(image, baseline))
            row[prefix + "_l1"] = float((image - baseline).abs().mean())
            row[f"psnr_delta_{name}_minus_reference"] = row[name + "_psnr"] - row["reference_psnr"]
            row[f"ssim_delta_{name}_minus_reference"] = row[name + "_ssim"] - row["reference_ssim"]
            if perceptual is not None:
                row[prefix + "_lpips"] = float(
                    perceptual(image[None] * 2 - 1, baseline[None] * 2 - 1))
                row[f"lpips_delta_{name}_minus_reference"] = (
                    row[name + "_lpips"] - row["reference_lpips"])
        rows.append(row)
        if directory:
            panel_names = ["ground_truth", *scenes]
            panel_images = {"ground_truth": gt, **images}
            comparison = torch.cat([panel_images[name] for name in panel_names], dim=2)
            pixels = (comparison.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(pixels).save(Path(directory) / f"{index:05d}.png")
    keys = [k for k in rows[0] if k != "view"]
    result = {"mean": {k: sum(r[k] for r in rows) / len(rows) for k in keys}, "views": rows,
              "lpips_input_range": "[-1,1]" if lpips else None}
    if hybrid_ablation:
        result.update(variant_definitions={name: HYBRID_VARIANT_DEFINITIONS[name]
                                           for name in scenes},
                      panel_order=["ground_truth", *scenes])
    return result
