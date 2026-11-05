"""Aligned hybrid allocation with current-network, context-bound corrections."""
import math
from functools import partial
from . import hybrid_tier_allocation as pipeline
from .hybrid_tier_step_table import TierStepTable
from .neural_correction_history import NetworkAwareEvidence
from .neural_correction_provider import NetworkCorrectionProvider,POLICY


def parser():
    p=pipeline.parser();p.description=__doc__
    p.set_defaults(network_mode='online',neural_correction_policy=POLICY)
    p.add_argument('--tier-max-log-odds',type=float,default=1.)
    return p


def configure(options,args):
    options.neural_correction_policy=POLICY;options.tier_max_log_odds=args.tier_max_log_odds


def main(args,max_updates=None,*,provider_factory=None,pretrain_factory=None):
    if args.network_mode!='online' or not args.aligned_spatial_inputs:
        raise ValueError('this experiment requires aligned inputs and online benefit learning')
    if not math.isfinite(args.tier_max_log_odds) or args.tier_max_log_odds<=0:
        raise ValueError('positive finite tier cap required')
    return pipeline.main(args,max_updates,provider_factory=provider_factory or NetworkCorrectionProvider,
        table_factory=partial(TierStepTable,tier_max_log_odds=args.tier_max_log_odds),
        scheduler_factory=NetworkAwareEvidence,allocation_transform=configure,
        pretrain_factory=pretrain_factory)


if __name__=='__main__':main(parser().parse_args())
