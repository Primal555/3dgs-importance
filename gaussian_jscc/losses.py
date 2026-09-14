"""Physical reconstruction objectives for geometry-first JSCC.

Source parameters are frozen targets, never receiver side information. All
weights/floors live in CodecConfig and are saved with the codec. No learned
loss coefficient can suppress the difficult geometry objective.
"""

import torch
from torch.nn import functional as F

from .data import to_raw


def rotation_matrix(q):
    """Scalar-first unit quaternion -> matrix; invariant to q versus -q."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def physical_terms(pred, target, geometry, model):
    """Per-source-row losses. Inputs are normalized features, not raw PLY rows.

    Local-scale geometry uses the SOURCE covariance with a scene-relative
    floor. Shape compares covariance in the source ellipsoid's whitened frame;
    avoids eigendecomposition/backward degeneracy for spherical Gaussians.
    log1p bounds the influence of large early reconstruction errors, not their
    existence. No inverse of a predicted covariance is used.
    """
    source = to_raw(target.detach(), geometry, model).detach()
    decoded = to_raw(pred, geometry, model)
    ref_rotation = rotation_matrix(source[:, 7:11])
    rotation = rotation_matrix(decoded[:, 7:11])
    diagonal = geometry.span.to(pred).norm()
    floor = (diagonal * model.cfg.geometry_floor).clamp_min(1e-12)
    ref_scale = (source[:, 4:7].exp().square() + floor.square()).sqrt()
    delta = decoded[:, :3] - source[:, :3]
    local_delta = torch.bmm(ref_rotation.transpose(1, 2), delta[..., None]).squeeze(-1)
    position = torch.log1p((local_delta / ref_scale).square().sum(-1))

    # Covariance factor transformed into the source frame, including the SAME
    # isotropic floor for both source and prediction. Identity at exact recovery.
    transform = torch.bmm(ref_rotation.transpose(1, 2), rotation)
    factor = transform * decoded[:, None, 4:7].exp() / ref_scale[:, :, None]
    whitened = torch.bmm(factor, factor.transpose(1, 2))
    whitened = whitened + torch.diag_embed((floor / ref_scale).square())
    eye = torch.eye(3, device=pred.device, dtype=pred.dtype)
    # log-domain squared norm avoids float32 overflow at large initial errors.
    log_error = (whitened - eye).abs().clamp_min(torch.finfo(pred.dtype).tiny).log() * 2
    log_mse = torch.logsumexp(log_error.flatten(1), -1) - pred.new_tensor(9.).log()
    shape = torch.logaddexp(torch.zeros_like(log_mse), log_mse)
    terms = {
        "geometry": position,
        "shape": shape,
        "opacity": F.smooth_l1_loss(decoded[:, 3].sigmoid(), source[:, 3].sigmoid(), reduction="none"),
        "dc": F.smooth_l1_loss(pred[:, 11:14], target.detach()[:, 11:14], reduction="none").mean(-1),
    }
    terms["sh"] = (F.smooth_l1_loss(pred[:, 14:], target.detach()[:, 14:], reduction="none").mean(-1)
                   if pred.shape[1] > 14 else pred[:, 0] * 0)
    return terms


def reconstruction_loss(pred, target, geometry, model, seed=None, seed_weight=.2,
                        active=None, reduction="mean", return_terms=False):
    """Shared by warmup, packed/batched training, replay and joint allocation.

    Detached activity weighting offers no DIRECT reward for suppressing an
    auxiliary term by dropping its row. ST gradients through codec inputs still
    exist. Joint rendering, including dropped rows, remains the task objective.
    Joint losses average over all SOURCE rows, not a learned retained count.
    """
    if model.cfg.architecture == "legacy":
        # Legacy support is for historical evaluations/performance checks.
        groups = ((0, 3), (3, 4), (4, 7), (7, 11), (11, 14), (14, target.shape[1]))
        row = torch.stack([F.smooth_l1_loss(pred[:, a:b], target[:, a:b], reduction="none").mean(-1)
                           for a, b in groups if b > a]).mean(0)
        if seed is not None:
            row = row + seed_weight * F.smooth_l1_loss(seed, target[:, :3], reduction="none").mean(-1)
        terms = {"legacy": row}
    else:
        terms = physical_terms(pred, target, geometry, model)
        row = sum(getattr(model.cfg, name + "_weight") * value for name, value in terms.items())
    if active is not None:
        row = row * active.detach()
        terms = {key: value * active.detach() for key, value in terms.items()}
    if reduction == "sum":
        loss = row.sum()
    elif reduction == "mean":
        loss = row.sum() / max(1, len(row))
    elif reduction == "none":
        loss = row
    else:
        raise ValueError("unknown loss reduction")
    return (loss, terms) if return_terms else loss


def add_arguments(parser):
    parser.add_argument("--geometry-rates", type=int, nargs=4,
                        help="geometry complex symbols per tier; default floor(total/2), not a measured optimum")
    for name, default in (("geometry", 1.), ("shape", .1), ("opacity", .1), ("dc", .1), ("sh", .05)):
        parser.add_argument(f"--{name}-weight", type=float,
                            help=f"new model default {default}; retain checkpoint value if omitted")
    parser.add_argument("--geometry-floor", type=float,
                        help="minimum scale / source bbox diagonal; new model default 1e-4")
    parser.add_argument("--render-target", choices=("source", "images"), default="source",
                        help="frozen input PLY rendering (default) or original training photographs")


def config_options(args):
    defaults = dict(geometry_weight=1., shape_weight=.1, opacity_weight=.1,
                    dc_weight=.1, sh_weight=.05, geometry_floor=1e-4)
    return {"geometry_rates": tuple(args.geometry_rates or ()),
            **{name: default if getattr(args, name) is None else getattr(args, name)
               for name, default in defaults.items()}}


def configure_training(model, args):
    if model.cfg.architecture != "geometry_first":
        raise ValueError("Legacy codec cannot initialize geometry-first training. Omit --init/--codec-init "
                         "to train the upgraded architecture; old checkpoints remain evaluable.")
    if args.geometry_rates is not None and tuple(args.geometry_rates) != model.cfg.geometry_rates:
        raise ValueError("geometry-rates cannot change when initializing a checkpoint")
    # Loss weights can change for fine-tuning; architecture/rate table cannot.
    for name, value in config_options(args).items():
        if name != "geometry_rates" and getattr(args, name) is not None:
            setattr(model.cfg, name, value)
    model.cfg.__post_init__()
