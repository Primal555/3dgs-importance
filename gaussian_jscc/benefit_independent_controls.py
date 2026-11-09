"""CPU training of independent A/B/S heads with ordinary-referenced budgets."""
import argparse
from pathlib import Path

from . import benefit_response_controls as controls
from .benefit_independent_response import (VERSION, IndependentResponseNetwork,
    OrdinaryGradientBudget, GroupTaskAccumulator, positive_energy_mean)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='completed response-terms run containing data/replay')
    p.add_argument('--out', required=True, help='new independent experiment output')
    p.add_argument('--data', help='override source/data label directory')
    p.add_argument('--phase', choices=('train',), default='train')
    p.add_argument('--device', default='cpu')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--samples', type=int, default=64000)
    p.add_argument('--microbatch', type=int, default=8)
    p.add_argument('--evaluate-every-samples', type=int, default=2048)
    p.add_argument('--gradient-every-samples', type=int, default=128)
    p.add_argument('--evaluation-batch', type=int, default=8)
    p.add_argument('--training-probe', type=int, default=128)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--ordinary-gradient-ema', type=float, default=.95)
    # Collection defaults satisfy the shared runner's validation; phase=train never uses them.
    p.set_defaults(train_records=4096, validation_records=512, label_groups=64,
                   candidate_batch=64, max_collect_windows=300)
    return p


class IndependentExperiment:
    version = VERSION
    accumulator_type = GroupTaskAccumulator

    def __init__(self, decay):
        self.budget = OrdinaryGradientBudget(decay)
        self.settings = dict(architecture='independent_A_B_S_common_relative_v1',
            initialization='same source backbone; freshly seeded signed and energy heads',
            ordinary_gradient_ema=decay, group_gradient_scope='all parameters; original weighted group contributions',
            zero_ordinary_reference_policy='preserve tail until a nonzero ordinary reference is available')

    def build_network(self, state, config, groups, scales):
        energy = positive_energy_mean(groups)
        self.settings['training_positive_energy_mean'] = energy
        return IndependentResponseNetwork.from_footprint(state, config['dropout'], scales, energy)


def main(args):
    if args.phase != 'train':
        raise ValueError('independent response experiment uses existing response labels; phase must be train')
    if args.data is None:
        args.data = str(Path(args.source)/'data')
    return controls.main(args, experiment=IndependentExperiment(args.ordinary_gradient_ema))


if __name__ == '__main__':
    main(parser().parse_args())
