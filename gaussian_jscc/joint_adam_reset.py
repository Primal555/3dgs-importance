"""30-step joint control: only change inherited vs reset channel Adam state.

The decoder-only adaptation experiment is separate and is NOT modified here.
No channel continuation, full test-set sweep, allocator or implicit LR change.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from .cli import device_for
from .data import read_ply
from .joint_stability_audit import audit_joint_step
from .modular_checkpoint import (capture_rng, restore_rng, restore_training_checkpoint,
                                 save_training_checkpoint, validate_scene)
from .modular_experiment import json_write, make_scene
from .modular_snr_controls import control_inputs
from .power_followup import encoder_digest, joint_optimizer
from .render_validation import append_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='stability suite root or a completed channel-only run')
    p.add_argument('--out', required=True, help='NEW output; short probes are never resumed from damaged joint weights')
    p.add_argument('--manifest', help='optional relocated Truck paths with unchanged scene/splits')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--top-k', type=int, default=16)
    p.add_argument('--adam-mode', choices=('reset', 'inherit'), default='reset')
    p.add_argument('--instrumented', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    return p


def build_plan(args):
    if min(args.steps, args.top_k, args.cpu_threads) < 1: raise ValueError('positive probe budgets required')
    root = run = Path(args.run).resolve()
    if not (run/'training.json').exists(): run = run/'block_continuation/block'
    out = Path(args.out).resolve()
    if any(out.is_relative_to(s) or s.is_relative_to(out) for s in (root, run)):
        raise ValueError('NEW output must be separate from source')
    path = run/'checkpoints/end_channel_bootstrap/codec.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = json.loads((run/'training.json').read_text(encoding='utf-8'))
    cfg = saved['config']; progress = saved['training']['progress']
    if (saved.get('version') != 4 or cfg.get('architecture') != 'modular_jscc_v1'
            or cfg.get('channel_variant') != 'progressive' or tuple(cfg['rates']) != (0, 8, 24, 48)
            or cfg.get('position_delivery') != 'quantized' or cfg.get('position_bits') != 16
            or cfg.get('channel_power_mode') not in ('block', 'point')
            or progress['phase'] != 'channel_bootstrap' or not progress.get('phase_complete')
            or not progress.get('freeze_semantic') or progress['completed'].get('channel_render', 0)
            or progress['step'] <= 0 or progress['step'] != progress['completed']['channel_bootstrap']
            or saved['step'] != progress['step'] or not saved['training']['optimizer']['state']):
        raise ValueError('requires completed pure channel recovery endpoint with Adam; never joint/latest')
    if json.loads(json.dumps(cfg)) != metadata['codec_config']: raise ValueError('source config/metadata mismatch')
    validate_scene(saved['training']['scene_record'], metadata['scene'])
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(version=1, source_run=str(run), checkpoint=str(path), source_sha256=digest,
        source_step=progress['step'], optimizer_policy=args.adam_mode,
        encoder_sha256=encoder_digest(saved['state_dict']), steps=args.steps, top_k=args.top_k,
        seed=args.seed, instrumented=args.instrumented, mode=cfg['channel_power_mode'],
        module_lrs={'channel_encoder': 1e-4, 'channel_decoder': 1e-4, 'semantic_decoder': 1e-5},
        objective='RGB MSE + (1-SSIM)/3 + original-y MSE, unchanged from joint audit',
        sampling='10dB AWGN; identical cumulative-step cameras, tiers and noise; replay; no clipping',
        manifest_override=str(Path(args.manifest).resolve()) if args.manifest else None)


def optimizer_summary(optimizer):
    result = {}
    for group in optimizer.param_groups:
        states = [optimizer.state[p] for p in group['params'] if p in optimizer.state]
        steps = [float(s['step']) for s in states if 'step' in s]
        result[group['name']] = dict(lr=group['lr'], betas=list(group['betas']), eps=group['eps'],
            parameters=len(group['params']), initialized_parameters=len(states),
            step_min=min(steps) if steps else 0., step_max=max(steps) if steps else 0.)
    return result


def summarize(records, plan, status, error=None):
    return dict(status=status, optimizer_policy=plan['optimizer_policy'], requested_updates=plan['steps'],
        updates=len(records), source_step=plan['source_step'], source_sha256=plan['source_sha256'],
        max_grad_norm=max((r['grad_norm'] for r in records), default=None),
        same_input_worse_steps=sum(r.get('same_input_loss_change', 0) > 0 for r in records),
        first_before=records[0]['before'] if records else None,
        first_after=records[0].get('after') if records else None,
        error=error,
        interpretation='short stability probe, not convergence or deployment quality; CUDA trajectories may diverge numerically')


def main(args):
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return plan
    out = Path(args.out).resolve()
    if out.exists(): raise FileExistsError('use NEW output; rerun short probe from original endpoint, not latest')
    out.mkdir(parents=True, exist_ok=False); json_write(out/'plan.json', plan)
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads)
    model, previous, progress, rng, scene_record = restore_training_checkpoint(plan['checkpoint'], device)
    metadata = json.loads((Path(plan['source_run'])/'training.json').read_text(encoding='utf-8'))
    original, _, spec, old, _ = control_inputs(metadata, args.manifest)
    raw, degree = read_ply(spec['ply'])
    scene = make_scene(spec, raw, degree, model, original, old, out); del raw
    validate_scene(scene_record, scene.record)
    optimizer = joint_optimizer(model, previous, 1e-4, 1e-5, reset_channel_adam=args.adam_mode == 'reset')
    progress = copy.deepcopy(progress)
    progress.update(phase='channel_render', phase_step=0, phase_complete=False, freeze_semantic=False,
        semantic_decoder_only=False, channel_denoiser_only=False, channel_neighbor_only=False,
        module_lrs=plan['module_lrs'], joint_adam_policy=args.adam_mode)
    progress['phase_budgets']['channel_render'] = args.steps
    progress['training_objective'] = dict(version='joint_adam_control_v1', channel_render_latent_weight=1.,
        render_image_objective={'kind': 'mse_ssim', 'aux_weight': 1/3})
    restore_rng(rng)

    def save(label):
        if encoder_digest(model.state_dict()) != plan['encoder_sha256']:
            raise RuntimeError('frozen semantic encoder/statistics changed')
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt', model, optimizer, progress, capture_rng(), scene.record)

    initialization = optimizer_summary(optimizer)
    if args.adam_mode == 'reset' and any(r['initialized_parameters'] for r in initialization.values()):
        raise RuntimeError('reset control unexpectedly inherited optimizer state')
    json_write(out/'optimizer_initial.json', initialization)
    json_write(out/'training.json', dict(plan=plan, codec_config=model.cfg.to_dict(), scene=scene.record,
        trainable_modules=['channel_encoder', 'channel_decoder', 'semantic_decoder'], semantic_encoder_frozen=True))
    save('initial')
    print(f'Joint Adam={args.adam_mode}; source={plan["source_step"]}; {args.steps} updates; '
          'channel LR=1e-4, semantic decoder LR=1e-5; unchanged joint objective', flush=True)
    print('Initial optimizer: '+json.dumps(initialization), flush=True)
    records = []; error = None; status = 'complete'
    try:
        for local in range(1, args.steps+1):
            started = time.perf_counter()
            row = audit_joint_step(model, optimizer, scene, plan['source_step']+local, local, args,
                                   instrumented=args.instrumented)
            row.update(adam_mode=args.adam_mode, optimizer=optimizer_summary(optimizer),
                       step_seconds=time.perf_counter()-started)
            records.append(row); append_json(out/'steps.jsonl', row)
            progress.update(step=row['step'], phase_step=local, phase_complete=local == args.steps)
            progress['completed']['channel_render'] = local
            save('latest')
            after = row['after']['loss'] if row['after'] is not None else None
            print(f'{args.adam_mode} joint {local}/{args.steps}: q={row["layout"]}, '
                  f'loss={row["before"]["loss"]:.6g}, post_loss={after}, grad={row["grad_norm"]:.6g}', flush=True)
            if row['after'] is None: raise RuntimeError('post-update measurement failed: '+row['post_update_error'])
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        status = 'probe_error'; error = f'{type(exc).__name__}: {exc}'
        json_write(out/'error.json', dict(error=error, recorded_updates=len(records)))
        print(error, flush=True)
    result = summarize(records, plan, status, error)
    result['semantic_encoder_unchanged'] = encoder_digest(model.state_dict()) == plan['encoder_sha256']
    json_write(out/'summary.json', result)
    if status == 'complete':
        save('final'); json_write(out/'complete.json', result)
    return result


if __name__ == '__main__':
    result = main(parser().parse_args())
    if result.get('status') == 'probe_error': raise SystemExit(1)
