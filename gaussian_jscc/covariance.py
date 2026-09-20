"""Experimental log-covariance representation and inference-only PLY conversion.

Features: XYZ(3), alpha logit(1), symmetric log covariance(6), DC(3), SH.
Render scenes have the SAME width, but contain covariance, NOT its logarithm.
Both use upper-triangle order xx,xy,xz,yy,yz,zz. Legacy PLY rows have one
additional column (scale3 + quaternion4). Widths are disjoint for SH 0..3.
No eigenvectors occur on the training path.
"""
import torch
from torch.nn import functional as F


def is_covariance_scene(scene):
    width = scene.shape[-1]
    if width in (13, 22, 37, 58):
        return True
    if width in (14, 23, 38, 59):
        return False
    raise ValueError(f'Invalid Gaussian scene width: {width}')


def pack_symmetric(matrix):
    return matrix[..., (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]


def unpack_symmetric(packed):
    xx, xy, xz, yy, yz, zz = packed.unbind(-1)
    return torch.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), -1).reshape(*packed.shape[:-1], 3, 3)


def quaternion_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def source_logcov(raw):
    rotation = quaternion_matrix(raw[..., 7:11])
    return (rotation * (2*raw[..., 4:7])[..., None, :]) @ rotation.transpose(-1, -2)


def feature_attributes(raw, cfg):
    if cfg.architecture != 'learned_split_logcov':
        return raw[..., 3:]
    return torch.cat((raw[..., 3:4], pack_symmetric(source_logcov(raw)), raw[..., 11:]), -1)


def covariance_exp(packed):
    # Float64 reduces roundoff for highly anisotropic source splats. Matrix
    # exponential has a well-defined derivative at repeated eigenvalues.
    result = torch.matrix_exp(unpack_symmetric(packed).double())
    result = ((result + result.transpose(-1, -2))*.5).to(packed.dtype)
    if not torch.isfinite(result).all():
        raise RuntimeError('Nonfinite log-covariance exponential; stop before optimizer update')
    return result


def scene_covariance(scene):
    if is_covariance_scene(scene):
        return unpack_symmetric(scene[..., 4:10])
    rotation = quaternion_matrix(scene[..., 7:11])
    factor = rotation * scene[..., 4:7].clamp(-20, 10).exp()[..., None, :]
    return factor @ factor.transpose(-1, -2)


def scene_sh_start(scene):
    return 10 if is_covariance_scene(scene) else 11


@torch.no_grad()
def max_log_radius(scene):
    if not is_covariance_scene(scene):
        return scene[..., 4:7].amax(-1)
    return .5*torch.linalg.eigvalsh(scene_covariance(scene).double())[..., -1].clamp_min(1e-30).log().to(scene)


@torch.no_grad()
def shape_spectrum(scene):
    if not is_covariance_scene(scene):
        return (2*scene[..., 4:7]).sort(-1).values
    return torch.linalg.eigvalsh(scene_covariance(scene).double()).clamp_min(1e-30).log().to(scene)


@torch.no_grad()
def matrix_quaternion(rotation):
    # Pick the largest quaternion component, avoiding trace singularities at
    # 180 degrees. This is export-only, including the discrete branch choice.
    r = rotation
    squared = torch.stack((1+r[...,0,0]+r[...,1,1]+r[...,2,2],
                           1+r[...,0,0]-r[...,1,1]-r[...,2,2],
                           1-r[...,0,0]+r[...,1,1]-r[...,2,2],
                           1-r[...,0,0]-r[...,1,1]+r[...,2,2]), -1).clamp_min(0)
    a = r[...,2,1]-r[...,1,2]
    b = r[...,0,2]-r[...,2,0]
    c = r[...,1,0]-r[...,0,1]
    d = r[...,1,0]+r[...,0,1]
    e = r[...,2,0]+r[...,0,2]
    f = r[...,2,1]+r[...,1,2]
    candidates = torch.stack((torch.stack((squared[...,0],a,b,c),-1),
                              torch.stack((a,squared[...,1],d,e),-1),
                              torch.stack((b,d,squared[...,2],f),-1),
                              torch.stack((c,e,f,squared[...,3]),-1)), -2)
    index = squared.argmax(-1)
    q = candidates.gather(-2, index[...,None,None].expand(*index.shape,1,4)).squeeze(-2)
    q = F.normalize(q, dim=-1)
    pivot = q.gather(-1, q.abs().argmax(-1, keepdim=True))
    return q*torch.where(pivot < 0, -1., 1.)


def export_raw(scene):
    """Covariance -> legacy PLY fields. Never allowed on an autograd path."""
    if not is_covariance_scene(scene):
        return scene
    if torch.is_grad_enabled() and scene.requires_grad:
        raise RuntimeError('Covariance PLY conversion is inference-only; use to_scene for training')
    with torch.no_grad():
        values, axes = torch.linalg.eigh(scene_covariance(scene).double())
        # Eigh may return an improper orthogonal frame; flip one axis without
        # changing covariance before converting to a proper rotation.
        axes[..., :, 0] *= torch.linalg.det(axes)[..., None]
        scales = .5*values.clamp_min(1e-30).log()
        q = matrix_quaternion(axes)
        return torch.cat((scene[..., :4], scales.to(scene), q.to(scene), scene[..., 10:]), -1)
