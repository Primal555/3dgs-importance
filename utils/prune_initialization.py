"""Read standard 3DGS PLYs without importing CUDA rasterizers.

PLY stores raw opacity logits, log scales and channel-major SH coefficients.
Do not activate/renormalize these values or pretend a PLY contains Adam state.
"""
import math
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


def ply_info(path, validate_values=False):
    vertex = PlyData.read(str(path))['vertex']
    names = set(vertex.data.dtype.names)
    required = {'x', 'y', 'z', 'opacity'}
    required.update(f'f_dc_{i}' for i in range(3))
    required.update(f'scale_{i}' for i in range(3))
    required.update(f'rot_{i}' for i in range(4))
    if not required <= names:
        raise ValueError(f'{path}: missing 3DGS attributes {sorted(required-names)}')
    rest = {n for n in names if n.startswith('f_rest_')}
    degree = math.isqrt(len(rest)//3+1)-1
    if len(rest) != 3*((degree+1)**2-1) or rest != {f'f_rest_{i}' for i in range(len(rest))}:
        raise ValueError(f'{path}: invalid SH coefficient count/order')
    if vertex.count == 0:
        raise ValueError(f'{path}: empty Gaussian cloud')
    if validate_values:
        for name in sorted(required | rest):
            if not np.isfinite(vertex.data[name]).all():
                raise ValueError(f'{path}: non-finite values in {name}')
        for start in range(0, vertex.count, 65536):
            rotations = np.stack([vertex.data[f'rot_{i}'][start:start+65536] for i in range(4)], 1)
            if (rotations == 0).all(1).any():
                raise ValueError(f'{path}: zero quaternion')
    return {'points':vertex.count, 'sh_degree':degree, 'bytes':Path(path).stat().st_size}


def read_prune_parameters(path, sh_degree, device='cpu'):
    info = ply_info(path, validate_values=True)
    if info['sh_degree'] != sh_degree:
        raise ValueError(f'{path}: SH degree {info["sh_degree"]}, requested {sh_degree}')
    vertex = PlyData.read(str(path))['vertex'].data
    count = len(vertex)

    def columns(names):
        array = np.stack([vertex[n] for n in names], axis=1).astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f'{path}: non-finite values in {names}')
        return torch.from_numpy(array).to(device)

    rest_count = (sh_degree+1)**2-1
    rest = (columns([f'f_rest_{i}' for i in range(3*rest_count)])
            .reshape(count, 3, rest_count).transpose(1, 2).contiguous()
            if rest_count else torch.empty((count, 0, 3), device=device))
    rotation = columns([f'rot_{i}' for i in range(4)])
    if (rotation.square().sum(1) == 0).any():
        raise ValueError(f'{path}: zero quaternion')
    mask = torch.tensor([10., 1.], device=device).repeat(count, 1)
    return {
        '_xyz':columns(['x','y','z']),
        '_features_dc':columns([f'f_dc_{i}' for i in range(3)]).unsqueeze(1),
        '_features_rest':rest,
        '_opacity':columns(['opacity']),
        '_scaling':columns([f'scale_{i}' for i in range(3)]),
        '_rotation':rotation,
        '_mask_score':mask,
    }


def initialize_from_ply(model, path, training_args, spatial_lr_scale, device='cuda'):
    if not math.isfinite(spatial_lr_scale) or spatial_lr_scale <= 0:
        raise ValueError('scene extent must be finite and positive')
    for key, value in read_prune_parameters(path, model.max_sh_degree, device).items():
        setattr(model, key, torch.nn.Parameter(value))
    model.active_sh_degree = model.max_sh_degree
    model.spatial_lr_scale = spatial_lr_scale
    model.max_radii2D = torch.zeros(len(model._xyz), device=device)
    model.training_setup(training_args)


def quantized_photo_metrics(image, gt):
    """Match original render.py PNG precision and metrics.py RGB reductions."""
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    image = (image.clamp(0, 1)*255+.5).to(torch.uint8).float()/255
    gt = (gt.clamp(0, 1)*255+.5).to(torch.uint8).float()/255
    image, gt = image.unsqueeze(0), gt.unsqueeze(0)
    return {'PSNR':psnr(image, gt).mean().item(), 'SSIM':ssim(image, gt).item()}
