"""Run the pinned upstream attribute codecs in their own Python environments."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


UPSTREAM = {
    'fcgs': ('https://github.com/YihangChen-ee/FCGS.git', '31e59a46f7e51505b46a7bbe6e53268ee3155fbf'),
    'c3dgs': ('https://github.com/KeKsBoTer/c3dgs.git', '2a234af55fbe8b90c8829c1436ce80088c4b622b'),
}
FCGS_RATES = (1e-4, 2e-4, 4e-4, 8e-4, 16e-4)


def fcgs_roundtrip(repo, raw, out, rate):
    from model import FCGS_model as upstream

    weights = repo / 'checkpoints' / 'checkpoint_{}.pkl'.format(rate)
    codec = upstream.FCGS(Q=1, resolutions_list=[300, 400, 500],
                         resolutions_list_3D=[70, 80, 90], norm_radius=3).cuda().eval()
    codec.load_state_dict(torch.load(str(weights), map_location='cpu', weights_only=True), strict=True)
    # Match the official chunking and its seeded within-chunk permutation.
    size = 1100000 if 1000000 < len(raw) < 1100000 else 1000000
    chunks = (2000000, 1000000, 1000000)
    rows = []
    for start in range(0, len(raw), size):
        packet = out / str(start // size)
        packet.mkdir()
        group = torch.from_numpy(raw[start:start + size]).cuda()
        xyz = group[:, :3].contiguous()
        sh = group[:, 14:].reshape(-1, 3, 15).transpose(1, 2).reshape(-1, 45)
        features = torch.cat((group[:, 11:14], sh, group[:, 3:11]), dim=1)
        # The common XYZ is already Morton-ordered, including stable row IDs
        # for coincident positions. Use it directly for both entropy contexts.
        with torch.no_grad(), patch.multiple(
                upstream,
                sorted_orig_voxels=lambda coords, attrs: (coords, attrs),
                compress_gaussian_params=lambda **kwargs: [{'file_size': {'total': 0}}],
                decompress_gaussian_params=lambda **kwargs: (xyz,)):
            encoded = codec.compress(xyz, features, root_path=str(packet),
                                     chunk_size_list=chunks, determ_codec=True)
            received_xyz, received = codec.decomprss(root_path=str(packet), chunk_size_list=chunks)
        order = torch.randperm(len(group), generator=torch.Generator().manual_seed(1)).to(group.device)
        if not torch.equal(received_xyz, xyz[order]):
            raise RuntimeError('FCGS decoder order differs from the pinned permutation')
        torch.testing.assert_close(received, encoded[1], atol=2e-5, rtol=1e-5)
        result = torch.empty_like(group)
        sh = received[:, 3:48].reshape(-1, 15, 3).transpose(1, 2).reshape(-1, 45)
        result[order] = torch.cat((received_xyz, received[:, 48:], received[:, :3], sh), dim=1)
        rows.append(result.cpu().numpy())
    return np.concatenate(rows), {
        'lambda': rate, 'weights': weights.name, 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'chunk_points': size,
    }


def c3dgs_roundtrip(raw, out, camera_file, seed):
    from arguments import CompressionParams, PipelineParams
    from compress import calc_importance
    from compression.vq import CompressionSettings, compress_gaussians
    from scene.gaussian_model import GaussianModel

    torch.manual_seed(seed)
    model = GaussianModel(3)
    model.active_sh_degree = 3
    group = torch.from_numpy(raw).cuda()
    xyz = group[:, :3].contiguous()
    sh = group[:, 14:].reshape(-1, 3, 15).transpose(1, 2).contiguous()
    scaling = group[:, 4:7].exp()
    scaling_norm = scaling.norm(2, dim=-1, keepdim=True)
    model._xyz = torch.nn.Parameter(xyz, requires_grad=False)
    model.xyz_qa = torch.nn.Identity()
    model._opacity = torch.nn.Parameter(group[:, 3:4].contiguous())
    model._scaling = torch.nn.Parameter(scaling / scaling_norm)
    model._scaling_factor = torch.nn.Parameter(scaling_norm.log())
    model._rotation = torch.nn.Parameter(group[:, 7:11].contiguous())
    model._features_dc = torch.nn.Parameter(group[:, None, 11:14].contiguous())
    model._features_rest = torch.nn.Parameter(sh)

    camera_rows = json.loads(camera_file.read_text(encoding='utf-8'))
    matrices = ('world_view_transform', 'full_proj_transform', 'camera_center')

    def training_cameras():
        for row in camera_rows:
            camera = dict(row)
            for key in matrices:
                camera[key] = torch.tensor(row[key], device='cuda', dtype=torch.float32)
            yield SimpleNamespace(**camera)

    parser = argparse.ArgumentParser()
    params, pipe = CompressionParams(parser), PipelineParams(parser)
    options = parser.parse_args([])
    config, pipeline = params.extract(options), pipe.extract(options)
    color, covariance = calc_importance(model, SimpleNamespace(getTrainCameras=training_cameras), pipeline)
    color_settings = CompressionSettings(config.color_codebook_size, config.color_importance_prune,
                                         config.color_importance_include, int(config.color_cluster_iterations),
                                         config.color_decay, config.color_batch_size)
    covariance_settings = CompressionSettings(config.gaussian_codebook_size, None,
                                              config.gaussian_importance_include,
                                              int(config.gaussian_cluster_iterations),
                                              config.gaussian_decay, config.gaussian_batch_size)
    compress_gaussians(model, color.amax(-1), covariance.amax(-1), color_settings,
                       covariance_settings, config.color_compress_non_dir, prune_threshold=-1)
    if not torch.equal(model._xyz, xyz):
        raise RuntimeError('C3DGS changed point identities during clustering')

    # Fit quantizer ranges on the final codebooks and dense attributes.
    # With zero fine-tuning, there is no later forward pass to update them.
    with torch.no_grad():
        for name, value in (
            ('features_dc', model._features_dc), ('features_rest', model._features_rest),
            ('opacity', model._opacity.sigmoid()), ('scaling', model.scaling_activation(model._scaling)),
            ('scaling_factor', model._scaling_factor), ('rotation', model._rotation),
        ):
            observer = getattr(model, name + '_qa')
            observer.activation_post_process.reset_min_max_vals()
            observer(value.detach())
            observer.disable_observer()

        archive = io.BytesIO()
        model.save_npz(archive, sort_morton=False)
        archive.seek(0)
        payload = out / 'attributes.npz'
        with np.load(archive, allow_pickle=False) as native:
            np.savez_compressed(payload, **{k: native[k] for k in native.files if k != 'xyz'})

        # Reconstruct the reader input from only the saved payload and shared XYZ.
        archive = io.BytesIO()
        with np.load(payload, allow_pickle=False) as attrs:
            np.savez(archive, xyz=xyz.cpu().numpy(), **{k: attrs[k] for k in attrs.files})
        archive.seek(0)
        del model
        decoded = GaussianModel(3)
        decoded.load_npz(archive)
        for name in ('features_dc', 'features_rest', 'opacity', 'scaling', 'scaling_factor', 'rotation'):
            obs = getattr(decoded, name + '_qa')
            obs.disable_observer()
            obs.disable_fake_quant()
        if not torch.equal(decoded._xyz, xyz):
            raise RuntimeError('C3DGS decoder modified the shared XYZ')
        # Encode exact opacity endpoints and degenerate scales in finite PLY units.
        opacity = decoded.get_opacity.clamp(1e-30, 1 - torch.finfo(group.dtype).eps).logit()
        scale = decoded.get_scaling.clamp_min(1e-30).log()
        color = decoded.get_features.transpose(1, 2).contiguous()
        result = torch.cat((xyz, opacity, scale, decoded.get_rotation, color[:, :, 0],
                            color[:, :, 1:].flatten(1)), dim=1)
    return result.cpu().numpy(), {
        'sensitivity_views': [row['image_name'] for row in camera_rows],
        'color_codebook_size': config.color_codebook_size,
        'covariance_codebook_size': config.gaussian_codebook_size,
        'point_pruning': False, 'finetune_steps': 0, 'seed': seed,
        'quantizer_calibration': 'post-clustering weights and codebooks',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=UPSTREAM)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cameras', type=Path)
    parser.add_argument('--fcgs-rate', type=float, choices=FCGS_RATES, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM[args.method][1]:
        raise ValueError('use the pinned upstream revision for ' + args.method)
    sys.path.insert(0, str(repo))
    raw = np.load(args.input, allow_pickle=False)
    streams = args.out / 'streams'
    streams.mkdir(parents=True)
    if args.method == 'fcgs':
        result, settings = fcgs_roundtrip(repo, raw, streams, args.fcgs_rate)
    else:
        result, settings = c3dgs_roundtrip(raw, streams, args.cameras, args.seed)
    if result.shape != raw.shape or not np.isfinite(result).all() or not np.array_equal(result[:, :3], raw[:, :3]):
        raise RuntimeError('attribute reconstruction must be finite and preserve every shared XYZ row')
    files = {str(p.relative_to(streams)): p.stat().st_size for p in sorted(streams.rglob('*')) if p.is_file()}
    total = sum(files.values())
    if not total:
        raise RuntimeError('missing encoded attribute payload')
    np.save(args.out / 'decoded.npy', result, allow_pickle=False)
    (args.out / 'payload.json').write_text(json.dumps({
        'upstream_commit': commit, 'settings': settings, 'attribute_bytes': total,
        'attribute_bits_per_gaussian': 8 * total / len(raw), 'files': files,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
