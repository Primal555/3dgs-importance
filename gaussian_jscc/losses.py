"""Physical reconstruction objectives for geometry-first JSCC.

Source parameters are frozen targets, never receiver side information. All
weights/floors live in CodecConfig and are saved with the codec. No learned
loss coefficient can suppress the difficult geometry objective.
"""

import torch
from torch.nn import functional as F

from .data import to_raw


PROFILES = {
    "physical_v1": dict(geometry_weight=1., shape_weight=.1, opacity_weight=.1,
                        dc_weight=.1, sh_weight=.05, scale_weight=0., geometry_floor=1e-4),
    "balanced_v2": dict(geometry_weight=1., shape_weight=.25, opacity_weight=1.,
                        dc_weight=1., sh_weight=.25, scale_weight=1., geometry_floor=1e-4),
    "position_v3": dict(geometry_weight=10., shape_weight=.25, opacity_weight=1.,
                        dc_weight=1., sh_weight=.25, scale_weight=1., geometry_floor=1e-4,
                        position_beta=.001, position_tail_margin=.01, position_tail_weight=2.),
    "robust_v4": dict(geometry_weight=10., shape_weight=.25, opacity_weight=1.,
                      dc_weight=1., sh_weight=.25, scale_weight=1., geometry_floor=1e-4),
}


def position_v3_terms(pred, target, cfg):
    """Unclipped bbox-unit correction with a bounded derivative for every row.

    Base SmoothL1 has non-vanishing constant slope for large errors. A second
    smooth tail penalty beyond a fixed margin adds pressure on bad coordinates.
    No scene-span / tiny Gaussian scale multiplier in this gradient path. The
    reference frame is still per-axis global bbox, not a local-coordinate codec.
    """
    delta = pred[:, :3] - target[:, :3].detach()
    base = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=cfg.position_beta,
                           reduction="none").mean(-1)
    excess = (delta.abs() - cfg.position_tail_margin).clamp_min(0.)
    tail = F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=cfg.position_beta,
                           reduction="none").mean(-1)
    return base, tail


@torch.no_grad()
def initialize_position_head(model, raw, geometry):
    """Source median only initializes a trainable bias; no new decoder input."""
    if model.position_head_needs_initialization:
        center = geometry.normalize(raw[:, :3]).median(0).values.to(model.position_seed.bias)
        model.position_seed.bias.copy_((center-.5)/.25)
        model.position_head_needs_initialization = False
        print("Initialized NEW normalized affine XYZ head near source median; "
              "all other codec weights retained. Position recovery must be retrained.")


def rotation_matrix(q):
    """Scalar-first unit quaternion -> matrix; invariant to q versus -q."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def physical_terms_v1(pred, target, geometry, model):
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


def physical_terms(pred, target, geometry, model):
    """Versioned objective; evaluating old checkpoints never changes their loss.

    V2 compares log covariance analytically, without eigendecomposition or a
    predicted inverse: log(Sigma) = R diag(2 log(s)) R^T. This removes the
    bounded-collapse penalty of V1 while preserving covariance equivalences.
    A separate sorted log-scale term constrains eigenvalues without penalizing
    an equivalent permutation of ellipsoid axes. These are engineering choices,
    not literature-reproduced weights or guarantees of rendering quality.
    """
    if model.cfg.loss_profile == "physical_v1":
        return physical_terms_v1(pred, target, geometry, model)
    target = target.detach()
    source = to_raw(target, geometry, model).detach()
    decoded = to_raw(pred, geometry, model)
    ref_rotation = rotation_matrix(source[:, 7:11])
    rotation = rotation_matrix(decoded[:, 7:11])
    if model.cfg.loss_profile == "position_v3":
        base, tail = position_v3_terms(pred, target, model.cfg)
        position = base + model.cfg.position_tail_weight * tail
    else:
        floor = (geometry.span.to(pred).norm() * model.cfg.geometry_floor).clamp_min(1e-12)
        ref_scale = (source[:, 4:7].exp().square() + floor.square()).sqrt()
        delta = decoded[:, :3] - source[:, :3]
        local_delta = torch.bmm(ref_rotation.transpose(1, 2), delta[..., None]).squeeze(-1)
        position = torch.log1p((local_delta / ref_scale).square().sum(-1))

    # Use UNCLIPPED network outputs for scale/logit guards. to_raw clamps for
    # safe rendering; using its clamped values here would create dead gradients.
    attr = pred[:, 3:] * model.attr_std.to(pred) + model.attr_mean.to(pred)
    ref_attr = target[:, 3:] * model.attr_std.to(pred) + model.attr_mean.to(pred)
    log_scale, ref_log_scale = attr[:, 1:4], ref_attr[:, 1:4].detach()
    log_cov = (rotation * (2 * log_scale)[:, None, :]) @ rotation.transpose(1, 2)
    ref_log_cov = (ref_rotation * (2 * ref_log_scale)[:, None, :]) @ ref_rotation.transpose(1, 2)
    # Frobenius distance is invariant to a joint rigid rotation. SmoothL1 of
    # individual matrix entries would not have that invariance.
    shape = (log_cov - ref_log_cov).square().sum((1, 2)) / 9
    scale = F.smooth_l1_loss(log_scale.sort(-1).values, ref_log_scale.sort(-1).values,
                             reduction="none").mean(-1)
    if model.cfg.loss_profile == 'robust_v4':
        # Fixed checkpoint statistic, never a trainable loss attenuation.
        # Remove isotropic trace: scale handles size; shape handles anisotropy
        # and orientation. Radial pseudo-Huber preserves rotation invariance.
        unit=model.attr_std[1:4].detach().square().mean().sqrt().clamp_min(1.).to(pred)
        error=(log_cov-ref_log_cov)/unit
        eye=torch.eye(3,device=pred.device,dtype=pred.dtype)
        error=error-error.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*eye
        shape=2*((1+error.square().sum((1,2))/9).sqrt()-1)
        delta=(log_scale.sort(-1).values-ref_log_scale.sort(-1).values)/unit
        scale=F.smooth_l1_loss(delta,torch.zeros_like(delta),reduction='none').mean(-1)
    alpha = F.smooth_l1_loss(attr[:, 0].sigmoid(), ref_attr[:, 0].sigmoid(), reduction="none")
    logit = F.smooth_l1_loss(attr[:, 0], ref_attr[:, 0], reduction="none")
    return {
        "geometry": position, "shape": shape, "scale": scale,
        "opacity": alpha + .1 * logit,
        "dc": F.smooth_l1_loss(pred[:, 11:14], target[:, 11:14], reduction="none").mean(-1),
        "sh": (F.smooth_l1_loss(pred[:, 14:], target[:, 14:], reduction="none").mean(-1)
               if pred.shape[1] > 14 else pred[:, 0] * 0),
    }


def objective_stats(values, model, auxiliary_weight=1.):
    """Log actual scalar contributions, not a claim about gradient dominance."""
    stats = {"loss_profile": model.cfg.loss_profile, "auxiliary_weight": auxiliary_weight}
    for name in ("geometry", "shape", "scale", "opacity", "dc", "sh"):
        if name + "_loss" in values:
            stats[name + "_contribution"] = (
                (1. if name=='geometry' and model.cfg.position_head=='reference_v6' else auxiliary_weight)
                * getattr(model.cfg, name + "_weight") * values[name + "_loss"])
    if "aux_loss" in values:
        stats["aux_contribution"] = auxiliary_weight * values["aux_loss"]
        if model.cfg.position_head=='reference_v6' and 'geometry_loss' in values:
            stats['aux_contribution']+=(1-auxiliary_weight)*model.cfg.geometry_weight*values['geometry_loss']
    return stats


def position_training_inputs(model, features, q, snr):
    """Preserve original packet boundaries BEFORE flattening padded batches."""
    if model.cfg.position_head != 'reference_v6':
        return {}
    choices=q if q.is_floating_point() else None
    tiers=q.detach().argmax(-1) if choices is not None else q
    scale=model.block_geometry.supervision_scale(features[...,:3],tiers)
    result={'position_scale':scale}
    if model.cfg.geometry_clean_weight:
        z=(model.block_geometry.encode_choices(features[...,:3],choices,snr) if choices is not None else
           model.block_geometry.encode(features[...,:3],tiers,snr))
        result['clean_pred']=model.block_geometry.decode(z,tiers,snr)
    return result


def reconstruction_loss(pred, target, geometry, model, seed=None, seed_weight=.2,
                        active=None, reduction="mean", return_terms=False,
                        position_scale=None, clean_pred=None):
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
        if model.cfg.architecture == 'learned_joint':
            from .learned_objective import learned_terms
            terms = learned_terms(pred, target, model)
        else:
            terms = physical_terms(pred, target, geometry, model)
        if model.cfg.position_head == 'reference_v6':
            from .reference_geometry import reference_position_rows
            if position_scale is None:
                q=torch.ones(len(target),device=target.device,dtype=torch.long)
                if active is not None: q=(active.detach()>.5).long()
                position_scale=model.block_geometry.supervision_scale(target[:,:3],q)
            terms['geometry']=reference_position_rows(pred,target,geometry,model,position_scale)
            if clean_pred is not None:
                terms['geometry']=terms['geometry']+model.cfg.geometry_clean_weight*reference_position_rows(
                    clean_pred,target,geometry,model,position_scale)
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
    parser.add_argument("--loss-profile", choices=tuple(PROFILES), default="balanced_v2",
                        help="training objective; old checkpoints are explicitly migrated for fine-tuning")
    parser.add_argument("--geometry-rates", type=int, nargs=4,
                        help="geometry complex symbols per tier; default floor(total/2), not a measured optimum")
    parser.add_argument("--position-head", choices=("sigmoid", "normalized_affine_v3", "block_relative_v4", "block_pilot_v5", "reference_v6"),
                        help="position_v3 selects normalized_affine_v3 unless explicitly overridden")
    parser.add_argument("--geometry-group-size", type=int,
                        help="v5 deterministic retained-row group size; default 256")
    parser.add_argument('--geometry-clean-weight',type=float,
                        help='v6 persistent clean geometry constraint across all training phases; default 1')
    parser.add_argument('--individual-tiers',action='store_true',
                        help='explicit wire upgrade: stable source-row groups and per-Gaussian detail power')
    parser.add_argument("--upgrade-position-head", action="store_true",
                        help="authorize position migration; block_relative_v4 replaces the geometric payload and power layout")
    parser.add_argument("--position-beta", type=float, help="SmoothL1 transition in per-axis bbox units")
    parser.add_argument("--position-tail-margin", type=float, help="large-error margin in per-axis bbox units")
    parser.add_argument("--position-tail-weight", type=float, help="additional bounded-slope tail penalty")
    parser.add_argument("--clip-mode", choices=("global", "branch"), default="global")
    parser.add_argument("--clip-norm", type=float, default=1.)
    for name in ("geometry", "shape", "scale", "opacity", "dc", "sh"):
        parser.add_argument(f"--{name}-weight", type=float,
                            help="override profile weight; retain checkpoint value when profile is unchanged")
    parser.add_argument("--geometry-floor", type=float,
                        help="minimum scale / source bbox diagonal; new model default 1e-4")
    parser.add_argument("--render-target", choices=("source", "images"), default="source",
                        help="frozen input PLY rendering (default) or original training photographs")


def config_options(args):
    profile = args.loss_profile
    defaults = PROFILES[profile]
    return {"geometry_rates": tuple(args.geometry_rates or ()),
            "geometry_group_size": getattr(args, 'geometry_group_size', None) or 256,
            "loss_profile": profile,
            "position_head": args.position_head or ("normalized_affine_v3" if profile == "position_v3" else "sigmoid"),
            **{name: default if getattr(args, name) is None else getattr(args, name)
               for name, default in defaults.items()}}


def configure_training(model, args):
    if model.cfg.architecture == 'learned_joint':
        raise ValueError('learned_joint uses the train-learned command; historical train/route2 objectives cannot migrate it')
    if model.cfg.architecture != "geometry_first":
        raise ValueError("Legacy codec cannot initialize geometry-first training. Omit --init/--codec-init "
                         "to train the upgraded architecture; old checkpoints remain evaluable.")
    if args.geometry_rates is not None and tuple(args.geometry_rates) != model.cfg.geometry_rates:
        raise ValueError("geometry-rates cannot change when initializing a checkpoint")
    requested_head = args.position_head or ("normalized_affine_v3" if args.loss_profile == "position_v3"
                                           and model.cfg.position_head == "sigmoid" else model.cfg.position_head)
    if model.cfg.position_head != requested_head:
        if not args.upgrade_position_head:
            raise ValueError("Position head change requires --upgrade-position-head; "
                             "the old XYZ head cannot be silently reused. Other weights will be retained.")
        if requested_head in ("block_relative_v4", "block_pilot_v5", "reference_v6"):
            if getattr(args, 'geometry_group_size', None) is not None:
                model.cfg.geometry_group_size = args.geometry_group_size
            model.enable_block_geometry(requested_head)
            print("Enabled block-relative geometry payload with systematic initialization; "
                  "geometry residual heads reset; attribute weights retained, but coding/power layout changed.")
        elif requested_head == "normalized_affine_v3" and not hasattr(model, "block_geometry"):
            model.cfg.position_head = requested_head
            model.reset_position_head()
        else:
            raise ValueError("Unsupported position migration; block_relative_v4 cannot be downgraded in place")
    if not torch.isfinite(torch.tensor(args.clip_norm)) or args.clip_norm <= 0:
        raise ValueError("clip-norm must be positive and finite")
    if model.cfg.loss_profile != args.loss_profile:
        print(f"Loss profile: {model.cfg.loss_profile} -> {args.loss_profile}; "
              "resetting objective weights to the selected profile, keeping codec weights and rate layout.")
        model.cfg.loss_profile = args.loss_profile
        for name, value in PROFILES[args.loss_profile].items():
            setattr(model.cfg, name, value)
    # Loss weights can change for fine-tuning; architecture/rate table cannot.
    for name, value in config_options(args).items():
        if name not in ("geometry_rates", "loss_profile", "position_head") and getattr(args, name) is not None:
            setattr(model.cfg, name, value)
    model.cfg.__post_init__()
    if model.cfg.position_head in ('block_pilot_v5','reference_v6'):
        model.block_geometry.group_size = model.cfg.geometry_group_size
    if model.cfg.position_head == 'reference_v6' and getattr(args,'geometry_clean_weight',None) is not None:
        model.cfg.geometry_clean_weight=args.geometry_clean_weight
        model.cfg.__post_init__()
    if getattr(args,'individual_tiers',False) and not model.cfg.individual_tiers:
        if model.cfg.position_head!='reference_v6':
            raise ValueError('--individual-tiers requires --position-head reference_v6')
        model.cfg.individual_tiers=True
        model.block_geometry.individual=True
        model.cfg.__post_init__()
        print('Explicit individual-tier wire upgrade: stable source-row groups; tier-independent base; per-row enhancement layers. Retraining required.')
