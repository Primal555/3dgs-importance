"""Stage-B attribute reconstruction, without rasterization or XYZ supervision."""
import torch
from .covariance import unpack_symmetric
from .data import to_scene
from .local_response import view_frames, centered_response_loss


def attribute_objective(pred, target, geometry, model, views=4, directions=None):
    """Equal mean of physical logcov matrix MSE and centered RGB response MSE.

    This two-term mean is an explicit engineering choice, not automatic gradient
    balancing. Responses share the existing black/white local footprint objective;
    they do not model scene occlusion/perspective. XYZ is excluded from both terms.
    Returns differentiable weighted contributions for gradient diagnostics.
    """
    if model.cfg.architecture != 'learned_split_logcov':
        raise ValueError('attribute reconstruction requires logcov representation')
    if views < 1 or pred.ndim != 2 or pred.shape != target.shape or not len(pred):
        raise ValueError('expected matching nonempty [N,D] and positive views')
    target = target.detach()
    # Replace centers with constants ONLY inside the isolated-splat loss. The
    # attribute decoder itself still receives its actual predicted XYZ condition.
    centered_pred = torch.cat((torch.zeros_like(pred[:, :3]), pred[:, 3:]), -1)
    centered_target = torch.cat((torch.zeros_like(target[:, :3]), target[:, 3:]), -1)
    decoded = to_scene(centered_pred, geometry, model)
    with torch.no_grad():
        source = to_scene(centered_target, geometry, model)
        if directions is None:
            directions = torch.randn(views, 3, device=pred.device, dtype=pred.dtype)
        if directions.ndim != 2 or directions.shape[-1] != 3 or not len(directions) or \
                not torch.isfinite(directions).all() or (directions.norm(dim=-1) < 1e-8).any():
            raise ValueError('directions must be finite nonzero [views,3]')
        directions, frames = view_frames(directions.to(pred))
    residual = (pred[:, 4:10]-target[:, 4:10])*model.attr_std[1:7]
    # Frobenius^2 / 9, preserving both off-diagonal occurrences.
    shape = unpack_symmetric(residual).square().mean()
    appearance, stats = centered_response_loss(decoded, source, directions, frames, model.cfg.sh_degree)
    terms = {'shape': shape/2, 'appearance': appearance/2}
    loss = sum(terms.values()).to(pred.dtype)
    return loss, {**stats, 'attribute_logcov_mse': float(shape.detach()),
                  'attribute_response_mse': float(appearance.detach())}, terms
