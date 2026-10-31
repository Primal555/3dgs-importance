"""Fresh spatial benefit pretraining with numerically calibrated direction KL."""
from functools import partial

from . import pixel_benefit_allocation as base
from . import spatial_benefit_allocation as spatial
from .calibrated_spatial_benefit import CalibratedSpatialLearner, DIRECTION_OBJECTIVE
from .batched_spatial_window import BatchedSpatialWindow, verify_batched_execution

FAMILY = 'calibrated_spatial_benefit_allocation_v1'


def parser():
    p = spatial.parser(); p.description = __doc__
    p.set_defaults(direction_objective=DIRECTION_OBJECTIVE)
    p.add_argument('--pretrain-until', type=int,
                   help='Resume pretraining to this absolute window count, preserving the original plan')
    p.add_argument('--allocation-batch', type=int,
                   help='Execution-only batch size for batched allocation projection and spatial features')
    return p


def learner_factory(replay, device, args):
    return CalibratedSpatialLearner(replay, device, hidden=args.hidden, lr=args.benefit_lr,
        dropout=args.dropout, weight_decay=args.weight_decay, decision_weight=args.decision_weight,
        tie_fraction=args.tie_fraction, common_weight=args.common_weight, pair_weight=args.pair_weight,
        training_probe_groups=args.training_probe_groups)


def main(args, max_updates=None):
    spatial.validate_options(args)
    batch = getattr(args, 'allocation_batch', None)
    if batch is not None and (args.phase != 'allocation' or batch < 1):
        raise ValueError('--allocation-batch requires allocation phase and a positive batch size')
    target = getattr(args, 'pretrain_until', None)
    if target is not None and (not args.resume or args.phase != 'pretrain' or target < args.pretrain_windows):
        raise ValueError('--pretrain-until requires --resume --phase pretrain and cannot shorten pretraining')
    return base.main(args, max_updates, family=FAMILY, learner_factory=learner_factory,
        window_factory=BatchedSpatialWindow if batch is not None else spatial.SpatialWindow,
        preflight_fn=batched_preflight if batch is not None else spatial.preflight,
        replay_factory=partial(spatial.SpatialReplay, cache_mib=args.replay_cache_mib),
        pretrain_target=target)


def batched_preflight(model, scene, camera, args, reference):
    result = spatial.preflight(model, scene, camera, args, reference)
    result.update(verify_batched_execution(next(model.parameters()).device))
    return result


if __name__ == '__main__':
    main(parser().parse_args())
