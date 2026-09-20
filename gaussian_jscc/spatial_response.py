"""Decoupled XYZ / native shape / centered appearance bootstrap (v3).

Position kernels never use predicted covariance or opacity. Unblurred shape
matching cannot hide behind the global bandwidth floor or low alpha. Targets
are training-only; receiver inputs and communication budgets are unchanged.
This experimental surrogate is NOT scene-render distortion.
"""
import math
import torch
from .data import to_raw
from .local_response import footprint, view_frames, centered_response_loss


DEFAULT_BANDWIDTHS = (.5, .125, .03125, .0078125)


def fine_position_response(decoded, source, geometry, reference):
    """Teacher-radius pseudo-Huber distance with bounded output-space slope.

    d and r are in bbox-diagonal units. rho=(sqrt(d^2+r^2)-r)/reference.
    The teacher max-axis radius controls quadratic-to-linear transition, NOT
    the gradient magnitude: ||d rho / d delta|| <= 1/reference regardless of
    how small the teacher splat is. Unlike an overlap kernel, the linear tail
    still attracts displaced points. Predicted shape/alpha cannot change rho.
    This is a 3D precision surrogate, not a camera/pixel loss, and does not give
    every relative point error equal weight. Shared-network gradients can still
    be amplified by its Jacobian; this is not a guarantee against all explosions.
    """
    diagonal = geometry.span.to(decoded).double().norm()
    delta = (decoded[:,:3].double()-source[:,:3].detach().double())/diagonal
    radius = (source[:,4:7].detach().double().amax(-1).exp()/diagonal).clamp_min(1e-12)
    squared = delta.square().sum(-1)
    # Rationalized form is exact at zero and stable for d << r.
    return (squared / ((squared+radius.square()).sqrt()+radius) / reference).mean()


def fixed_position_response(delta, bandwidths):
    """Unit-L2 isotropic kernels: 2*(1-exp(-|delta|^2/(4*sigma^2))).

    The only learned input is center displacement. Equal fixed source/predicted
    bandwidths deliberately exclude shape, rotation, alpha and color. This is
    coarse capture, not a guarantee of pixel-accurate positions.
    """
    squared = delta.double().square().sum(-1)
    return torch.stack([(-2*torch.expm1(-squared/(4*sigma*sigma))).mean()
                        for sigma in bandwidths])


def native_shape_response(decoded, source, frames):
    """Negative log normalized Gaussian overlap at coincident centers.

    No global blur. Work relative to each teacher's max axis, in float64.
    -log(overlap), rather than 1-overlap, avoids saturation for huge splats.
    The relative 1e-5 covariance floor in footprint() is numerical, not a
    bbox-sized physical floor. Scale and rotation are coupled as covariance,
    not independently weighted parameter errors; alpha cannot mask this term.
    """
    cp, rp = footprint(decoded, frames)
    with torch.no_grad():
        ct, rt = footprint(source.detach(), frames)
    cp, ct = cp.double(), ct.double()
    a = (cp @ cp.transpose(-1,-2)) * (2*(rp.double()-rt.double())).exp()[:,None,None,None]
    b = ct @ ct.transpose(-1,-2)
    # Analytic log determinants of the factors avoid forming/inverting an
    # ill-scaled 3x3 covariance. Cholesky on a+b is only 2x2 per point/view.
    logdet_a = 2*cp.diagonal(dim1=-2,dim2=-1).log().sum(-1) + 4*(rp.double()-rt.double())[:,None]
    logdet_b = 2*ct.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    cs = torch.linalg.cholesky(a+b)
    logdet_sum = 2*cs.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    return (.5*logdet_sum-.25*(logdet_a+logdet_b)-math.log(2)).clamp_min(0).mean()


@torch.no_grad()
def shape_metrics(decoded, source):
    pred_radius = decoded[:,4:7].amax(-1).exp()
    source_radius = source[:,4:7].amax(-1).exp()
    ratio = pred_radius/source_radius
    distance = (decoded[:,:3]-source[:,:3]).norm(dim=-1)
    return {'max_axis_ratio_p50':float(ratio.median()),
            'max_axis_ratio_p05':float(torch.quantile(ratio,.05)),
            'max_axis_ratio_p95':float(torch.quantile(ratio,.95)),
            'max_axis_ratio_gt10_fraction':float((ratio>10).float().mean()),
            'max_axis_ratio_lt0_1_fraction':float((ratio<.1).float().mean()),
            'decoded_max_axis_p50_world':float(pred_radius.median()),
            'source_max_axis_p50_world':float(source_radius.median()),
            'xyz_distance_over_source_radius_p50':float((distance/source_radius).median()),
            'decoded_alpha_p50':float(decoded[:,3].sigmoid().median())}


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
                          bandwidths=DEFAULT_BANDWIDTHS, directions=None, fine_weight=1.):
    """Mean of (coarse + weighted fine position), native shape and centered RGB.

    Equal weights are an explicit experimental choice, not a claim that scalar
    values imply equal gradient influence. Fine position uses a teacher-radius
    transition; render validation is still required for usable geometry.
    """
    if model.cfg.position_delivery != 'learned':
        raise ValueError('spatial-response requires learned XYZ, without a position side stream')
    if views < 1 or pred.ndim != 2 or pred.shape != target.shape or not len(pred):
        raise ValueError('spatial response needs matching nonempty [N,D] inputs and positive views')
    if not bandwidths or any(not math.isfinite(s) or s <= 0 for s in bandwidths):
        raise ValueError('bandwidths must be positive and finite bbox-diagonal fractions')
    if not math.isfinite(fine_weight) or fine_weight<0:
        raise ValueError('fine_weight must be finite and nonnegative')
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

    scale_losses = fixed_position_response(delta, bandwidths)
    coarse = scale_losses.mean()
    fine = fine_position_response(decoded,source,geometry,min(bandwidths))
    position = coarse + fine_weight*fine
    shape = native_shape_response(decoded, source, frames)
    appearance, appearance_stats = centered_response_loss(decoded, source, directions, frames, model.cfg.sh_degree)
    loss = ((position+shape+appearance)/3).to(pred.dtype)
    stats = {'spatial_position_response':float(position.detach()),
             'spatial_coarse_position_response':float(coarse.detach()),
             'spatial_fine_position_response':float(fine.detach()),
             'spatial_fine_weight':float(fine_weight),
             'spatial_fine_contribution':float((fine_weight*fine/3).detach()),
             'spatial_native_shape_response':float(shape.detach()),
             'spatial_geometry_response':float(position.detach()),
             'spatial_appearance_response':float(appearance.detach()),
             **appearance_stats, **shape_metrics(decoded,source)}
    stats.update({f'spatial_scale_{i}_loss': float(x.detach()) for i, x in enumerate(scale_losses)})
    stats.update(position_metrics(pred, target, geometry))
    return loss, stats
