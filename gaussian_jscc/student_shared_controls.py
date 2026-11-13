"""200-step continuation with shared four-state labels and output-controlled updates."""
from functools import partial
from . import student_keep_controls as controls
from .student_shared_feedback import SharedResidualFeedback, shared_losses
from .student_output_step import output_controlled_step

VERSION = 'student_shared_residual_controls_v1'
ARMS = ('shared_residual',)


def parser():
    p = controls.parser(ARMS, 'shared_residual')
    p.description = __doc__
    p.set_defaults(keep_draws=128)
    p.add_argument('--full-groups', type=int, default=24)
    p.add_argument('--keep-output-kl', type=float, default=2e-5)
    p.add_argument('--tier-output-kl', type=float, default=2e-5)
    return p


def main(args, max_updates=None):
    import math
    if args.full_groups < 1 or args.keep_draws-args.full_groups < 16:
        raise ValueError('keep draws must exceed full groups by at least 16')
    if any(not math.isfinite(v) or v <= 0 for v in (args.keep_output_kl, args.tier_output_kl)):
        raise ValueError('invalid output KL budget')
    return controls.main(args, max_updates, version=VERSION, arms=ARMS,
        sampler_factory=partial(SharedResidualFeedback, full_groups=args.full_groups),
        loss_builder=shared_losses, step_function=partial(output_controlled_step,
            keep_kl=args.keep_output_kl, tier_kl=args.tier_output_kl),
        plan_extra=dict(full_groups=args.full_groups, keep_output_kl=args.keep_output_kl,
            tier_output_kl=args.tier_output_kl,
            training='shared student and original Adam; paired shared-state keep/tier residuals; output-controlled displacement',
            gradient_target='fixed-context single-switch gradient sums; full-support strata and two-stage candidate/group inclusion',
            keep_sampling='104 global stratified draws + 24 retained candidate groups by default; convex mixture on retained residual',
            positive_sampling='both alternative positive states per group; conditional group weight and outer candidate inclusion once',
            cost_scope='paired exact image differences + original marginal-rate proxy; actual three-file costs for map RD',
            runtime_defaults='128 keep draws, 48 positive action draws, at most 176 training alternative states; duplicate states reused'))


if __name__ == '__main__': main(parser().parse_args())
