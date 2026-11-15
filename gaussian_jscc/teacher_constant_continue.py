"""Continue a completed constant-LR first-order teacher in an independent directory."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import budgeted_allocation as engine
from . import fast_first_order_allocation as fast
from .budgeted_allocation_state import load_checkpoint
from .dense_rd_allocation import atomic_save
from .modular_experiment import json_write
from .multisnr_evaluation import read

VERSION = 'teacher_constant_continue_v1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--extra-steps', type=int, default=500)
    p.add_argument('--device', default='cuda')
    p.add_argument('--digital-python', required=True)
    p.add_argument('--digital-device', default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def options_for(plan, args, target):
    settings = plan['settings']
    options = fast.parser().parse_args(['--run', settings['run'], '--scene', settings['scene'],
        '--out', str(args.out), '--beta', str(settings['beta'])])
    vars(options).update(copy.deepcopy(settings))
    options.out = str(args.out.resolve()); options.device = args.device
    options.resume = True; options.dry_run = False; options.allocation_steps = target
    candidate = engine.make_plan(options, plan['version'], plan['source'], plan['spec'], plan['codec_config'])
    if 'objective_reference_anchor' in plan:
        candidate['objective_reference_anchor'] = copy.deepcopy(plan['objective_reference_anchor'])
    expected = copy.deepcopy(plan); expected['settings']['allocation_steps'] = target
    if candidate != expected:
        raise ValueError('continuation changed settings beyond allocation_steps')
    return options, candidate


def validate_source(plan, state, complete):
    s = plan['settings']
    if (plan != state['plan'] or complete.get('status') != 'complete' or
            complete['step'] != state['progress']['step'] or
            state['progress']['step'] != s['allocation_steps']):
        raise ValueError('completed full-limit source and matching latest checkpoint required')
    if (s.get('early_stop', True) or s.get('cost_protocol') != 'digital_side_file_v1' or
            s.get('beta') is None or s.get('fast_execution_version') != 'batched_first_order_v1' or
            not s.get('structured_evidence') or state.get('positive_provider') is not None or
            state['progress'].get('stopping')):
        raise ValueError('digital-cost constant-LR calibrated first-order teacher required')
    if state['progress']['pending']:
        raise ValueError('source has unfinished evaluation events')


def verify_delivery(options, args, result):
    """Resumable actual delivery of selected, latest and uniform q2 maps."""
    from comparison_baselines.digital_link import LinkConfig
    from comparison_baselines.run import summarize_trials
    from .budgeted_allocation_inputs import load_inputs
    from .budgeted_allocation_evaluation import tier_hash
    from .cli import device_for
    from .data import read_ply
    from .model_factory import build_codec
    from .multisnr_evaluation import deliver_trial
    from .perceptual_metrics import LPIPSEvaluator
    from .successive_multiscene_inputs import make_scene
    from .transport import model_id
    out = Path(options.out); root = out/'actual_delivery'
    summary = root/'summary.json'
    if summary.exists():
        if read(summary)['codec_id'] != result['codec_id']:
            raise ValueError('delivery codec identity changed')
        return
    saved, cfg, spec, old, scene_args, _ = load_inputs(options)
    device = device_for(args.device)
    model = build_codec(cfg).to(device); model.load_state_dict(saved['state_dict']); del saved
    model.eval().requires_grad_(False); model.set_training_snrs([10.])
    if model_id(model) != result['codec_id']: raise ValueError('allocation codec changed')
    raw, degree = read_ply(spec['ply'])
    scene = make_scene(spec, raw, degree, model, scene_args, root, old=old); del raw
    perceptual = LPIPSEvaluator(device) if options.lpips else None
    config = LinkConfig(device=args.digital_device)
    metrics = [json.loads(s) for s in (out/'metrics/validation.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    rows = []
    for label in ('selected', 'latest', 'uniform_q2'):
        q = (torch.full((len(scene.raw),), 2, dtype=torch.long) if label == 'uniform_q2' else
             torch.from_numpy(np.load(out/'deployment'/f'{label}_tiers.npy', allow_pickle=False).astype(np.int64)))
        if label == 'selected' and tier_hash(q) != result['tier_sha256']: raise ValueError('selected map changed')
        expected_cost = next(r['total_channel_uses'] for r in reversed(metrics) if r['event'] == label)
        for trial in range(options.trials):
            folder = root/label/f'{trial:03d}'; path = folder/'result.json'
            if path.exists():
                row = read(path)
                if row['tier_sha256'] != tier_hash(q) or row['codec_id'] != result['codec_id']:
                    raise ValueError('cached delivery map/codec changed')
            else:
                print(f'Actual side delivery: {label}, trial {trial+1}/{options.trials}', flush=True)
                row = deliver_trial(model, scene, q, options.snr, options.seed+1000+trial,
                    options.seed+100000+3*trial, config, args.digital_python, folder, perceptual=perceptual)
                row.update(method=label, trial=trial, tier_sha256=tier_hash(q))
                json_write(path, row)
            if row['total_channel_uses'] != expected_cost: raise RuntimeError('serialized cost differs from allocation')
            rows.append(row)
    json_write(summary, dict(status='complete', codec_id=result['codec_id'], trials=rows,
        results=summarize_trials(rows), cost_protocol='digital_side_file_v1', arq=False,
        quality_conditioning='successful reception of all three side files'))


def main(args, max_updates=None):
    if args.extra_steps < 1: raise ValueError('extra-steps must be positive')
    source, out = args.source.resolve(), args.out.resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('independent output directory required')
    identity = dict(source=str(source), extra_steps=args.extra_steps,
                    digital_python=str(args.digital_python), digital_device=args.digital_device)
    if args.resume:
        record = read(out/'continuation.json')
        if record['settings'] != identity: raise ValueError('continuation resume settings mismatch')
        plan = read(out/'plan.json')
        options, _ = options_for(plan, args, record['target_step'])
    else:
        if out.exists(): raise FileExistsError('use new OUT or --resume')
        path = source/'checkpoints/latest/state.pt'
        original = read(source/'plan.json'); complete = read(source/'complete.json')
        state = load_checkpoint(path); validate_source(original, state, complete)
        start = state['progress']['step']; target = start+args.extra_steps
        options, plan = options_for(original, args, target)
        with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        record = dict(version=VERSION, settings=identity, source_step=start, target_step=target,
            source_checkpoint_sha256=digest, original_plan=original,
            resume_state='latest table, evidence history, view scheduler, RNG, objective references and best checkpoint',
            selection='existing best_rd rule on calibration views; latest endpoint reported independently')
        if args.dry_run:
            print(json.dumps(record, indent=2)); return record
        best = load_checkpoint(source/'checkpoints/best_rd/state.pt')
        if best['plan'] != original or best['progress']['step'] != state['progress']['best']['step']:
            raise ValueError('source best_rd checkpoint mismatch')
        best['plan'] = plan
        atomic_save(out/'checkpoints/best_rd/state.pt', best); del best
        state['plan'] = plan
        state['progress']['initial_q'] = state['latest_q'].clone()
        state['progress']['pending'] = ['fork_initial']
        json_write(out/'plan.json', plan)
        training = read(source/'training.json')
        training.update(plan); training['continuation'] = record
        json_write(out/'training.json', training); json_write(out/'continuation.json', record)
        atomic_save(out/'checkpoints/latest/state.pt', state); del state
    if args.dry_run:
        print(json.dumps(record, indent=2)); return record
    print(f'Continue teacher {record["source_step"]} -> {record["target_step"]}; '
          f'keep LR={options.keep_lr:g}, tier LR={options.tier_lr:g}, beta={options.beta:g}.', flush=True)
    if (out/'complete.json').exists(): result = read(out/'complete.json')
    else: result = fast.main(options, max_updates=max_updates)
    if result['status'] != 'complete': return result
    verify_delivery(options, args, result)
    result.update(continuation_version=VERSION, continued_from_step=record['source_step'],
                  additional_updates=result['step']-record['source_step'], actual_side_delivery_completed=True)
    json_write(out/'complete.json', result); fast.review(out, result)
    return result


if __name__ == '__main__': main(parser().parse_args())
