"""Calibrated first-order allocation with batched execution and plateau ending."""
import argparse
import json
import math
from functools import partial
from pathlib import Path
import tarfile
from . import structured_evidence_allocation as previous
from . import budgeted_allocation as engine
from .fast_first_order_execution import FastFirstOrderExecution
from .first_order_stopping import FirstOrderStopping
from .modular_experiment import json_write


def parser():
    p = previous.parser(); p.description = __doc__
    p.set_defaults(validate_every=100, fast_execution_version='batched_first_order_v1')
    p.add_argument('--candidate-batch', type=int, default=128)
    p.add_argument('--cost-protocol', choices=('legacy', 'digital_side_file_v1'), default='legacy')
    p.add_argument('--tier-max-log-odds', type=float, default=argparse.SUPPRESS,
                   help='Override the positive-tier cap; keep uses --max-log-odds')
    p.add_argument('--early-stop', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--stop-min-steps', type=int, default=1250)
    p.add_argument('--stop-patience', type=int, default=2)
    p.add_argument('--stop-relative-improvement', type=float, default=.005)
    p.add_argument('--stop-map-tolerance', type=float, default=.005)
    p.add_argument('--stop-cost-tolerance', type=float, default=.005)
    return p


def table_factory_for(args):
    cap = getattr(args, 'tier_max_log_odds', None)
    if cap is None: return None
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError('tier-max-log-odds must be finite and positive')
    from .hybrid_tier_step_table import TierStepTable
    return partial(TierStepTable, tier_max_log_odds=cap)


def review(out, result):
    out = Path(out)
    rows = [json.loads(s) for s in (out/'metrics/validation.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    final = {r['event']: r for r in rows if r['event'] in ('initial', 'selected', 'latest', 'uniform_q2', 'uniform_q3')}
    q2 = final.get('uniform_q2')
    comparisons = {}
    if q2:
        for name in ('selected', 'latest'):
            r = final[name]
            comparisons[name] = dict(source_psnr_difference=r['source_psnr']-q2['source_psnr'],
                photo_psnr_difference=r['photo_psnr']-q2['photo_psnr'],
                cost_saving=1-r['total_channel_uses']/q2['total_channel_uses'],
                rd_difference=r['rd_score']-q2['rd_score'],
                meets_q2_goal=r['source_psnr'] >= q2['source_psnr'] and r['total_channel_uses'] < q2['total_channel_uses'])
    json_write(out/'review_summary.json', dict(completion=result, rows=final, against_q2=comparisons))
    temporary = out/'review.tar.gz.tmp'
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in sorted(out.rglob('*')):
            relative = path.relative_to(out)
            if path.is_file() and relative.parts[0] in ('evaluation', 'metrics', 'charts', 'comparison',
                    'plan.json', 'training.json', 'complete.json', 'progress.json', 'execution_check.json', 'review_summary.json'):
                if '.uncheckpointed_' not in path.name:
                    archive.add(path, arcname=relative.as_posix(), recursive=False)
        for name in ('selected_tiers.npy', 'latest_tiers.npy', 'selection.json'):
            archive.add(out/'deployment'/name, arcname='deployment/'+name)
    temporary.replace(out/'review.tar.gz')


def main(args, max_updates=None):
    previous.validate(args)
    table_factory = table_factory_for(args)
    execution = FastFirstOrderExecution(args.candidate_batch)
    stopping = FirstOrderStopping(args.stop_min_steps, args.stop_patience, args.stop_relative_improvement,
                                 args.stop_map_tolerance, args.stop_cost_tolerance) if args.early_stop else None
    result = engine.main(args, max_updates=max_updates, execution=execution, stopping=stopping,
                         table_factory=table_factory)
    if result.get('status') == 'complete': review(args.out, result)
    return result


if __name__ == '__main__': main(parser().parse_args())
