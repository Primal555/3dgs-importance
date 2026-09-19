"""Experimental bootstrap loss: analytic, center-sensitive splat responses.

No renderer, anchor side stream, or target-conditioned decoder. Each view uses
L2-normalized 2D Gaussian footprints at several *fixed* bandwidths measured in
source bbox-diagonal units. Analytic integrals avoid missing displaced splats
with a small probe stencil. Broad scales provide cold-start attraction; narrow
scales provide precision. This surrogate is not scene-render distortion.
"""
import math
import torch
from .data import to_raw
from .local_response import footprint, view_frames
from utils.sh_utils import eval_sh


DEFAULT_BANDWIDTHS = (.5, .125, .03125, .0078125)


def position_metrics(pred, target, geometry):
    with torch.no_grad():
        delta = (pred[:, :3]-target[:, :3]) * geometry.span.to(pred)
        distance = delta.norm(dim=-1)
        diagonal = geometry.span.to(pred).norm()
        rmse = delta.square().mean().sqrt()
        return {'xyz_rmse_world': float(rmse),
                'xyz_nrmse_bbox': float(rmse/diagonal),
                'xyz_distance_p50_world': float(distance.median()),
                'xyz_distance_p95_world': float(torch.quantile(distance, .95))}


def spatial_response_loss(pred, target, geometry, model, views=4,
                          bandwidths=DEFAULT_BANDWIDTHS, directions=None):
    """Mean integrated squared response error over N, views, scales, channels.

    Seven channels: unit geometry response, RGB on black, RGB contrast to white.
    The unit channel prevents opacity/color suppression from hiding XYZ errors.
    All channels and scales have equal weight; their choice and bandwidths are
    experimental hyperparameters, not an established optimal physical loss.
    Unit-L2 footprints give self inner product 1; cross inner product K includes
    both center displacement and projected covariance mismatch.
    """
    if model.cfg.position_delivery != 'learned':
        raise ValueError('spatial-response requires learned XYZ, without a position side stream')
    if views < 1 or pred.ndim != 2 or pred.shape != target.shape or not len(pred):
        raise ValueError('spatial response needs matching nonempty [N,D] inputs and positive views')
    if not bandwidths or any(not math.isfinite(s) or s <= 0 for s in bandwidths):
        raise ValueError('bandwidths must be positive and finite bbox-diagonal fractions')
    target = target.detach()
    decoded = to_raw(pred, geometry, model)
    with torch.no_grad():
        source = to_raw(target, geometry, model)
        if directions is None:
            directions = torch.randn(views, 3, device=pred.device, dtype=pred.dtype)
        if (directions.ndim != 2 or directions.shape[-1] != 3 or
            not torch.isfinite(directions).all() or (directions.norm(dim=-1) < 1e-8).any()):
            raise ValueError('directions must be finite nonzero [views,3]')
        directions, frames = view_frames(directions.to(pred))
    diagonal = geometry.span.to(pred).norm()
    delta = torch.einsum('vij,nj->nvi', frames, (decoded[:, :3]-source[:, :3])/diagonal)

    def fields(raw):
        chol, radius = footprint(raw, frames)
        # Existing relative footprint floor stabilizes highly anisotropic splats.
        covariance = (chol @ chol.transpose(-1, -2)) * (radius.exp()/diagonal).square()[:, None, None, None]
        degree = model.cfg.sh_degree
        rest = raw[:, 14:].reshape(len(raw), 3, (degree+1)**2-1)
        sh = torch.cat((raw[:, 11:14, None], rest), -1)
        color = (eval_sh(degree, sh[:, None], directions[None])+.5).clamp_min(0)
        alpha = raw[:, 3].sigmoid()[:, None, None]
        amplitude = torch.cat((torch.ones_like(color[..., :1]), alpha*color, alpha*(color-1)), -1)
        return covariance, amplitude

    cov_p, amp_p = fields(decoded)
    with torch.no_grad():
        cov_t, amp_t = fields(source)
    # Tiny matrices in double precision keep log determinants and overlap stable;
    # learned features/codec remain float32. No inverse or detached XYZ path.
    eye = torch.eye(2, device=pred.device, dtype=torch.float64)
    scale_losses, geometry_losses, appearance_losses = [], [], []
    for sigma in bandwidths:
        a = cov_p.double() + sigma**2*eye
        b = cov_t.double() + sigma**2*eye
        ca, cb, cs = (torch.linalg.cholesky(c) for c in (a, b, a+b))
        logdet = lambda c: 2*c.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        whitened = torch.linalg.solve_triangular(cs, delta.double()[..., None], upper=False)
        log_k = math.log(2) + .25*(logdet(ca)+logdet(cb)) - .5*logdet(cs) - .5*whitened.square().sum((-2,-1))
        # Cauchy-Schwarz: K <= 1; tolerate roundoff near identical footprints.
        log_k = log_k.clamp_max(0)
        one_minus_k = -torch.expm1(log_k)
        # Equivalent to ap^2+at^2-2*ap*at*K, without cancellation at identity.
        terms = (amp_p.double()-amp_t.double()).square() + 2*amp_p.double()*amp_t.double()*one_minus_k[..., None]
        scale_losses.append(terms.mean())
        geometry_losses.append(terms[..., 0].mean())
        appearance_losses.append(terms[..., 1:].mean())
    loss = torch.stack(scale_losses).mean().to(pred.dtype)
    stats = {'spatial_geometry_response': float(torch.stack(geometry_losses).mean().detach()),
             'spatial_appearance_response': float(torch.stack(appearance_losses).mean().detach())}
    stats.update({f'spatial_scale_{i}_loss': float(x.detach()) for i, x in enumerate(scale_losses)})
    stats.update(position_metrics(pred, target, geometry))
    return loss, stats
