"""Wide spatial positive-tier evaluator with complete historical replay."""
from functools import partial

from . import neural_correction_allocation as original
from . import calibrated_spatial_benefit_allocation as spatial
from .aligned_spatial_inputs import AlignedSpatialWindow
from .full_history_spatial_replay import FullHistorySpatialReplay
from .neural_correction_provider import NetworkCorrectionProvider
from .pixel_benefit_allocation import main as train_spatial

pretrain_window = AlignedSpatialWindow
VARIANT = 'wide_neural_full_history_v1'


def parser():
    p = original.parser()
    p.description = __doc__
    p.set_defaults(hidden=256, full_history_policy=FullHistorySpatialReplay.policy,
                   benefit_variant=VARIANT)
    return p


class FullHistoryPositiveProvider(NetworkCorrectionProvider):
    replay_class = FullHistorySpatialReplay

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.settings.get('full_history_policy') != FullHistorySpatialReplay.policy:
            raise ValueError('complete-history pretraining artifact required')

    def result_fields(self):
        return dict(super().result_fields(), benefit_variant=VARIANT,
                    benefit_replay_policy=FullHistorySpatialReplay.policy,
                    benefit_hidden=self.settings['hidden'],
                    benefit_parameters=sum(p.numel() for p in self.learner.network.parameters()),
                    benefit_history_groups=len(self.replay.training))


def pretrain(args, max_updates=None):
    spatial.spatial.validate_options(args)
    return train_spatial(args, max_updates, family=VARIANT,
        learner_factory=spatial.learner_factory, window_factory=pretrain_window,
        preflight_fn=spatial.spatial.preflight,
        replay_factory=partial(FullHistorySpatialReplay, cache_mib=args.replay_cache_mib))


def main(args, max_updates=None):
    if (getattr(args, 'full_history_policy', None) != FullHistorySpatialReplay.policy or
            getattr(args, 'benefit_variant', None) != VARIANT):
        raise ValueError('wide full-history experiment settings mismatch')
    return original.main(args, max_updates, provider_factory=FullHistoryPositiveProvider,
                         pretrain_factory=pretrain)


if __name__ == '__main__':
    main(parser().parse_args())
