"""Independent evidence-driven frozen-codec rate-distortion experiment."""
import argparse
import math
from . import budgeted_allocation as engine

FAMILY = 'evidence_allocation_v1'


def parser():
    p = engine.parser(); p.description = __doc__
    p.set_defaults(tier_lr=.4, audit_every=0, audit_samples=0)
    p.add_argument('--candidate-rounds',type=int,default=2)
    p.add_argument('--candidate-policy',choices=('coverage_demand','round_robin'),default='coverage_demand')
    p.add_argument('--evidence-memory',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--evidence-half-life',type=float,default=1000.)
    p.add_argument('--context-tv',type=float,default=.1)
    p.add_argument('--finite-feedback',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--finite-samples',type=int,default=8)
    p.add_argument('--uniform-controls',action=argparse.BooleanOptionalAction,default=True)
    p.set_defaults(evidence_experiment=True)
    return p


def validate(args):
    if args.beta is None: raise ValueError('evidence experiment requires fixed --beta')
    if (not 1 <= args.candidate_rounds <= 8 or args.finite_samples < 0
            or not math.isfinite(args.evidence_half_life) or args.evidence_half_life <= 0
            or not math.isfinite(args.context_tv) or not 0 < args.context_tv <= 1):
        raise ValueError('invalid evidence settings')
    if args.audit_every or args.audit_samples:
        raise ValueError('use finite feedback; legacy diagnostic-only audits are disabled here')


def main(args,max_updates=None):
    validate(args)
    return engine.main(args,max_updates=max_updates)


if __name__=='__main__': main(parser().parse_args())
