"""Fresh spatial benefit pretraining with numerically calibrated direction KL."""
from functools import partial

from . import pixel_benefit_allocation as base
from . import spatial_benefit_allocation as spatial
from .calibrated_spatial_benefit import CalibratedSpatialLearner, DIRECTION_OBJECTIVE

FAMILY = 'calibrated_spatial_benefit_allocation_v1'


def parser():
    p = spatial.parser(); p.description = __doc__
    p.set_defaults(direction_objective=DIRECTION_OBJECTIVE)
    p.add_argument('--pretrain-until', type=int,
                   help='Resume pretraining to this absolute window count, preserving the original plan')
    return p


def learner_factory(replay, device, args):
    return CalibratedSpatialLearner(replay, device, hidden=args.hidden, lr=args.benefit_lr,
        dropout=args.dropout, weight_decay=args.weight_decay, decision_weight=args.decision_weight,
        tie_fraction=args.tie_fraction, common_weight=args.common_weight, pair_weight=args.pair_weight,
        training_probe_groups=args.training_probe_groups)


def main(args, max_updates=None):
    spatial.validate_options(args)
    target = getattr(args, 'pretrain_until', None)
    if target is not None and (not args.resume or args.phase != 'pretrain' or target < args.pretrain_windows):
        raise ValueError('--pretrain-until requires --resume --phase pretrain and cannot shorten pretraining')
    return base.main(args, max_updates, family=FAMILY, learner_factory=learner_factory,
        window_factory=spatial.SpatialWindow, preflight_fn=spatial.preflight,
        replay_factory=partial(spatial.SpatialReplay, cache_mib=args.replay_cache_mib),
        pretrain_target=target)


if __name__ == '__main__':
    main(parser().parse_args())
