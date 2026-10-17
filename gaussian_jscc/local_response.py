"""Training-only isolated Gaussian response, not a receiver or full renderer.

Orthographic footprints, shared centers, random or supplied viewing directions,
and equally weighted black/white backgrounds. Source and detached prediction
footprint probes prevent broad predictions from escaping the sampled domain.
No occlusion, perspective, or pixel low-pass filter: scene rendering remains
the final objective. Sampling and regularization are explicit design choices.
"""
import math
import torch
from torch.nn import functional as F
from .data import to_raw
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
    log_radius = raw[:,4:7].amax(-1)
    factor = rotation_matrix(raw[:,7:11]) * (raw[:,4:7]-log_radius[:,None]).exp()[:,None,:]
    projected = (torch.einsum('vij,njk->nvik', frames, factor) if frames.ndim == 3 else
                 torch.einsum('nvij,njk->nvik', frames, factor))
    covariance = projected @ projected.transpose(-1,-2)
    # Finite footprint floor relative to the largest 3D axis (not loss weight).
    covariance = covariance + 1e-5*torch.eye(2,device=raw.device,dtype=raw.dtype)
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
    rest = raw[:,14:].reshape(len(raw),3,(degree+1)**2-1)
    sh = torch.cat((raw[:,11:14,None],rest),-1)
    dirs = directions[None] if directions.ndim == 2 else directions
    color = (eval_sh(degree,sh[:,None],dirs)+.5).clamp_min(0)
    black = alpha[...,None]*color[:,:,None,:]
    white = black + 1-alpha[...,None]
    return black, white


@torch.no_grad()
def camera_directions(points, camera_centers):
    """World-space camera -> point directions, matching the rasterizer SH sign.

    Training teacher inputs only, never passed to the receiver. The sampled
    centers must come from training cameras, not validation/test cameras.
    """
    centers = torch.as_tensor(camera_centers, device=points.device, dtype=points.dtype)
    if (points.ndim != 2 or points.shape[-1] != 3 or centers.ndim != 2 or
            centers.shape[-1] != 3 or not len(points) or not len(centers) or
            not torch.isfinite(points).all() or not torch.isfinite(centers).all()):
        raise ValueError('finite nonempty [points,3] and [cameras,3] required')
    delta = points.detach()[:, None] - centers.detach()[None]
    if (delta.norm(dim=-1) < 1e-8).any():
        raise ValueError('camera center coincides with a Gaussian center')
    return F.normalize(delta, dim=-1)


def directional_color_loss(pred, target, model, directions):
    """Unclipped SH RGB MSE, independent of alpha, scale, rotation and XYZ.

    Inputs are row-aligned normalized [N,D] features. Teacher attributes are
    detached. Directions are [V,3] or [N,V,3], in the renderer's camera->point
    convention. This supervision is training-only, never receiver side data.
    """
    if (pred.ndim != 2 or pred.shape != target.shape or not len(pred)
            or pred.shape[1] != 3+model.cfg.attr_dim):
        raise ValueError('color supervision requires nonempty matching feature rows')
    if (directions.ndim not in (2,3) or directions.shape[-1] != 3 or directions.shape[-2] < 1
            or (directions.ndim == 3 and directions.shape[0] != len(pred))
            or not torch.isfinite(directions).all() or (directions.norm(dim=-1)<1e-8).any()):
        raise ValueError('finite nonzero [V,3] or [N,V,3] color directions required')
    directions=F.normalize(directions.detach().to(pred),dim=-1)
    if directions.ndim == 2: directions=directions[None]
    degree=model.cfg.sh_degree
    def colors(features):
        coefficients=features[:,11:]*model.attr_std[8:]+model.attr_mean[8:]
        dc=coefficients[:,:3,None]
        rest=coefficients[:,3:].reshape(len(features),3,(degree+1)**2-1)
        sh=torch.cat((dc,rest),-1)
        return eval_sh(degree,sh[:,None],directions)+.5
    difference=colors(pred)-colors(target.detach()).detach()
    loss=difference.square().mean()
    if not torch.isfinite(loss): raise RuntimeError('nonfinite directional color loss')
    return loss, {'objective':'directional_color_mse_unclipped_v1',
                  'directional_color_mse':float(loss.detach()),'color_directions':directions.shape[-2],
                  **{f'directional_color_{c}_bias':float(difference[...,i].detach().mean())
                     for i,c in enumerate('rgb')}}


def local_response_loss(pred, target, geometry, model, views=4, directions=None):
    """Mean RGB response MSE per active Gaussian/view/probe/background.

    pred/target are [active_N,D] normalized features, with padding/q0 already
    excluded. XYZ is intentionally not supervised: only explicit XYZ delivery
    is supported. Both isolated splats are centered at the same origin. Teacher
    attributes and adaptive probes are detached and never enter the decoder.
    """
    if model.cfg.position_delivery == 'learned':
        raise ValueError('local response requires float32 or quantized XYZ delivery; it does not train XYZ')
    return isolated_attribute_response_loss(pred,target,geometry,model,views,directions)


def isolated_attribute_response_loss(pred, target, geometry, model, views=4, directions=None):
    """Attribute-only isolated response; explicitly ignores predicted centers.

    Unlike the legacy guarded entrypoint this is also valid for XYZ carried
    inside a payload, but MUST NOT be described as supervising its position.
    """
    if views < 1 or pred.ndim != 2 or pred.shape != target.shape or not len(pred):
        raise ValueError('local response needs nonempty matching [N,D] inputs and positive views')
    target = target.detach()
    decoded = to_raw(pred,geometry,model)
    with torch.no_grad():
        source = to_raw(target,geometry,model)
        if directions is None:
            directions = torch.randn(views,3,device=pred.device,dtype=pred.dtype)
        if (directions.ndim not in (2, 3) or directions.shape[-1] != 3 or
                directions.shape[-2] < 1 or (directions.ndim == 3 and directions.shape[0] != len(pred)) or
                not torch.isfinite(directions).all() or (directions.norm(dim=-1)<1e-8).any()):
            raise ValueError('directions must be finite nonzero [views,3] or [points,views,3]')
        directions, frames = view_frames(directions.to(pred))
        source_chol, source_radius = footprint(source,frames)
    decoded_chol, decoded_radius = footprint(decoded,frames)
    with torch.no_grad():
        points = stencil(pred)
        source_probes = torch.einsum('nvij,pj->nvpi',source_chol,points)
        decoded_probes = torch.einsum('nvij,pj->nvpi',decoded_chol.detach(),points)
        decoded_probes *= (decoded_radius.detach()-source_radius).exp()[:,None,None,None]
        probes = torch.cat((source_probes,decoded_probes),-2)
        teacher = response_terms(source,directions,source_chol,source_radius,source_radius,probes,model.cfg.sh_degree)
    received = response_terms(decoded,directions,decoded_chol,decoded_radius,source_radius,probes,model.cfg.sh_degree)
    black = (received[0]-teacher[0]).square().mean()
    white = (received[1]-teacher[1]).square().mean()
    return (black+white)*.5, {'local_black_mse':float(black.detach()),
                             'local_white_mse':float(white.detach())}
