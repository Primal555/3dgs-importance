"""Mixed-distortion, continuous-evidence frozen-codec allocation experiment."""
import math
from . import evidence_allocation as previous
from . import budgeted_allocation as engine

FAMILY = 'calibrated_structured_allocation_v2'


def parser():
    p = previous.parser(); p.description = __doc__
    p.set_defaults(structured_evidence=True, finite_samples=32)
    p.add_argument('--distortion', choices=('mse', 'mse_ssim'), default='mse_ssim')
    p.add_argument('--structure-share', type=float, default=.2)
    p.add_argument('--lambda-s', type=float, default=None)
    p.add_argument('--max-log-odds', type=float, default=.25)
    return p


def validate(args):
    previous.validate(args)
    if (not math.isfinite(args.structure_share) or not 0 < args.structure_share < 1 or
            not math.isfinite(args.max_log_odds) or args.max_log_odds <= 0 or
            (args.lambda_s is not None and (not math.isfinite(args.lambda_s) or args.lambda_s < 0))):
        raise ValueError('invalid structured objective/update settings')
    if args.distortion == 'mse' and args.lambda_s not in (None, 0.):
        raise ValueError('MSE mode requires zero structural weight')


def main(args, max_updates=None):
    validate(args)
    return engine.main(args, max_updates=max_updates)


if __name__ == '__main__': main(parser().parse_args())
