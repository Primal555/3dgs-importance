"""Training-only isolated Gaussian response, not a receiver or full renderer.

Orthographic footprints, shared centers, uniformly sampled viewing directions,
and equally weighted black/white backgrounds. Source and detached prediction
footprint probes prevent broad predictions from escaping the sampled domain.
No occlusion, perspective, or pixel low-pass filter: scene rendering remains
the final objective. Sampling and regularization are explicit design choices.
"""
import math
import torch
from torch.nn import functional as F
from .data import to_scene
from .covariance import is_covariance_scene, scene_covariance, scene_sh_start
from utils.sh_utils import eval_sh


def rotation_matrix(quaternion):
    w, x, y, z = F.normalize(quaternion, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1,3,3)


def view_frames(directions):
    d = F.normalize(directions, dim=-1)
    axis = F.one_hot(d.abs().argmin(-1), 3).to(d)
    u = F.normalize(torch.linalg.cross(d, axis), dim=-1)
    v = torch.linalg.cross(d, u)
    return d, torch.stack((u, v), -2)


def footprint(raw, frames):
    # Factor out the largest log scale before forming covariance. All matrix
    # solves then operate on bounded 2x2 matrices, not scene-sized coordinates.
    if is_covariance_scene(raw):
        cov3 = scene_covariance(raw).double()
        # Trace is a smooth positive scale even at repeated eigenvalues.
        # No predicted eigenvectors or eigenvalues in this autograd path.
        radius_squared = cov3.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-30)
        log_radius = .5*radius_squared.log()
        normalized = cov3 / radius_squared[:, None, None]
        frames = frames.double()
        covariance = frames[None] @ normalized[:, None] @ frames.transpose(-1,-2)[None]
    else:
        log_radius = raw[:,4:7].amax(-1)
        factor = rotation_matrix(raw[:,7:11]) * (raw[:,4:7]-log_radius[:,None]).exp()[:,None,:]
        projected = torch.einsum('vij,njk->nvik', frames, factor)
        covariance = projected @ projected.transpose(-1,-2)
    # Finite footprint floor relative to the largest 3D axis (not loss weight).
    covariance = covariance + 1e-5*torch.eye(2,device=raw.device,dtype=covariance.dtype)
    return torch.linalg.cholesky(covariance), log_radius


def stencil(like):
    angles = torch.arange(8,device=like.device,dtype=like.dtype)*(math.pi/4)
    circle = torch.stack((angles.cos(),angles.sin()),-1)
    return torch.cat((like.new_zeros(1,2), circle*.5, circle*1.5, circle*3.),0)


def response_terms(raw, directions, cholesky, log_radius, reference_radius, probes, degree):
    offsets = probes * (reference_radius-log_radius).exp()[:,None,None,None]
    whitened = torch.linalg.solve_triangular(cholesky,offsets.transpose(-1,-2),upper=False)
    squared_distance = whitened.square().sum(-2)
    alpha = raw[:,3].sigmoid()[:,None,None] * (-.5*squared_distance).exp()
    # PLY SH is channel-major; DC is stored separately from higher orders.
    start = scene_sh_start(raw)
    rest = raw[:,start+3:].reshape(len(raw),3,(degree+1)**2-1)
    sh = torch.cat((raw[:,start:start+3,None],rest),-1)
    color = (eval_sh(degree,sh[:,None],directions[None])+.5).clamp_min(0)
    black = alpha[...,None]*color[:,:,None,:]
    white = black + 1-alpha[...,None]
    return black, white


def local_response_loss(pred, target, geometry, model, views=4, directions=None):
    """Mean RGB response MSE per active Gaussian/view/probe/background.

    pred/target are [active_N,D] normalized features, with padding/q0 already
    excluded. XYZ is intentionally not supervised: only explicit XYZ delivery
    is supported. Both isolated splats are centered at the same origin. Teacher
    attributes and adaptive probes are detached and never enter the decoder.
    """
    if model.cfg.position_delivery == 'learned':
        raise ValueError('local response requires float32 or quantized XYZ delivery; it does not train XYZ')
    if views < 1 or pred.ndim != 2 or pred.shape != target.shape or not len(pred):
        raise ValueError('local response needs nonempty matching [N,D] inputs and positive views')
    target = target.detach()
    decoded = to_scene(pred,geometry,model)
    with torch.no_grad():
        source = to_scene(target,geometry,model)
        if directions is None:
            directions = torch.randn(views,3,device=pred.device,dtype=pred.dtype)
        if directions.ndim != 2 or directions.shape[-1] != 3 or not torch.isfinite(directions).all() or (directions.norm(dim=-1)<1e-8).any():
            raise ValueError('directions must be finite nonzero [views,3]')
        directions, frames = view_frames(directions.to(pred))
    return centered_response_loss(decoded, source, directions, frames, model.cfg.sh_degree)


def centered_response_loss(decoded, source, directions, frames, degree):
    """Native-scale appearance/footprint matching with both centers at zero.

    Training-only teacher. Neither XYZ tensor is used, so misplacement cannot
    improve this objective by broadening a splat. The explicit-position public
    loss above and the learned-XYZ bootstrap share precisely this computation.
    """
    source = source.detach()
    with torch.no_grad():
        source_chol, source_radius = footprint(source,frames)
    decoded_chol, decoded_radius = footprint(decoded,frames)
    with torch.no_grad():
        points = stencil(decoded).to(decoded_chol)
        source_probes = torch.einsum('nvij,pj->nvpi',source_chol,points)
        decoded_probes = torch.einsum('nvij,pj->nvpi',decoded_chol.detach(),points)
        decoded_probes *= (decoded_radius.detach()-source_radius).exp()[:,None,None,None]
        probes = torch.cat((source_probes,decoded_probes),-2)
        teacher = response_terms(source,directions,source_chol,source_radius,source_radius,probes,degree)
    received = response_terms(decoded,directions,decoded_chol,decoded_radius,source_radius,probes,degree)
    black = (received[0]-teacher[0]).square().mean()
    white = (received[1]-teacher[1]).square().mean()
    return (black+white)*.5, {'local_black_mse':float(black.detach()),
                             'local_white_mse':float(white.detach())}
