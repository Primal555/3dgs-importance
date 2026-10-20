"""Continue pre-joint channel recovery in a new directory, retaining Adam/RNG.

Both semantic modules stay frozen. Only latent MSE is optimized; image and
color metrics remain evaluation-only. The parent's future joint stage is
explicitly disabled rather than resumed after the extra channel updates.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .modular_checkpoint import restore_training_checkpoint, save_training_checkpoint
from .modular_experiment import json_write, main as train
from .modular_joint_continue import training_options


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='Truck modular run containing end_channel_bootstrap')
    p.add_argument('--out', required=True, help='NEW output directory; original run is read-only')
    p.add_argument('--checkpoint', help='explicit completed pre-joint channel checkpoint; default: RUN/checkpoints/end_channel_bootstrap/codec.pt')
    p.add_argument('--additional-steps', type=int, default=4000)
    p.add_argument('--validate-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--device', default='cuda')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--manifest', help='optional migrated scene paths; PLY and splits must still match')
    p.add_argument('--resume', action='store_true', help='finish this extension without adding its budget again')
    p.add_argument('--dry-run', action='store_true', help='CPU checkpoint preflight only; no output or scene loading')
    return p


def main(args, max_updates=None):
    if min(args.additional_steps, args.validate_every, args.save_every, args.cpu_threads) < 1:
        raise ValueError('additional steps, intervals and threads must be positive')
    source = Path(args.run).resolve()
    out = Path(args.out).resolve()
    if source == out or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('continuation must use a NEW directory outside the original run')
    if args.resume:
        record = json.loads((out/'training.json').read_text(encoding='utf-8'))
        info = record.get('continuation', {})
        if (info.get('kind') != 'channel_bootstrap' or info.get('parent_run') != str(source) or
                info.get('additional_steps') != args.additional_steps or
                info.get('validate_every') != args.validate_every or info.get('save_every') != args.save_every or
                (args.checkpoint and str(Path(args.checkpoint).resolve()) != info.get('parent_checkpoint'))):
            raise ValueError('extension resume settings mismatch; do not add the budget twice')
        if not (out/'checkpoints/latest/codec.pt').is_file():
            raise FileNotFoundError('missing continuation latest checkpoint')
    else:
        if out.exists():
            raise FileExistsError(f'use a NEW output directory or --resume: {out}')
        record = json.loads((source/'training.json').read_text(encoding='utf-8'))
        if (record.get('codec_config', {}).get('architecture') != 'modular_jscc_v1' or
                record.get('rate_sweep') or record.get('inverse_control') or record.get('training_family') or
                record.get('clean_only') or not isinstance(record.get('scene'), dict) or
                [s.get('name') for s in record.get('selected_scenes', [])] != ['truck']):
            raise ValueError('expected a single-scene modular Truck channel-training run')
        objective = record.get('training_objective', {}).get('render_image_objective', {}).get('kind', 'mse')
        if objective not in ('mse', 'mse_ssim'):
            raise ValueError('channel continuation supports MSE/SSIM sources, not GAN/LPIPS controls')
        checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else source/'checkpoints/end_channel_bootstrap/codec.pt'
        model, optimizer, progress, rng, scene_record = restore_training_checkpoint(checkpoint, 'cpu')
        completed = progress['completed']
        if (progress.get('phase') != 'channel_bootstrap' or not progress.get('phase_complete') or
                not progress.get('channel_trained') or completed.get('channel_bootstrap', 0) < 1 or
                completed.get('channel_render') != 0 or progress['phase_step'] != completed['channel_bootstrap'] or
                progress['step'] != sum(completed.values())):
            raise ValueError('requires a completed channel_bootstrap checkpoint BEFORE any joint rendering')
        if (progress.get('phase_budgets') != record['phase_budgets'] or
                any(completed.get(p) != record['phase_budgets'][p]
                    for p in ('semantic_bootstrap', 'semantic_render', 'channel_bootstrap')) or
                progress.get('training_objective') != record.get('training_objective') or
                progress['module_lrs'] != record['module_lrs'] or
                progress.get('train_snrs') != record['train_snrs'] or scene_record != record['scene'] or
                json.loads(json.dumps(model.cfg.to_dict())) != record['codec_config']):
            raise ValueError('source checkpoint and training metadata disagree')
        if (optimizer is None or not optimizer.state or
                {g.get('name') for g in optimizer.param_groups} != {'channel_encoder', 'channel_decoder'}):
            raise ValueError('source must contain trained channel encoder/decoder Adam state')
        for group in optimizer.param_groups:
            if group['lr'] != record['module_lrs'][group['name']]:
                raise ValueError('saved Adam learning rates disagree with metadata')
        with checkpoint.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        parent_step = progress['step']
        old_budget = completed['channel_bootstrap']
        record = copy.deepcopy(record)
        record['phase_budgets'].update(channel_bootstrap=old_budget+args.additional_steps, channel_render=0)
        progress.update(phase_budgets=dict(record['phase_budgets']), phase_complete=False, freeze_semantic=True)
        progress.pop('status', None)
        record.update(source_run=str(source), source_step=parent_step, random_initialization=False,
                      source_weights_loaded=True, freeze_semantic=True,
                      initialization='pre-joint channel extension: all weights, channel Adam/RNG and phase cursor retained',
                      trainable_modules={p: ['channel_encoder', 'channel_decoder'] if p == 'channel_bootstrap' else []
                                         for p in record['phase_budgets']},
                      active_training_objective='latent MSE only; render/color/SSIM/LPIPS are evaluation-only')
        record['continuation'] = dict(kind='channel_bootstrap', parent_run=str(source),
            parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=digest,
            parent_step=parent_step, parent_channel_step=old_budget, additional_steps=args.additional_steps,
            target_step=parent_step+args.additional_steps, target_channel_step=old_budget+args.additional_steps,
            optimizer_preserved=True, rng_preserved=True, semantic_modules_frozen=True,
            joint_steps=0, validate_every=args.validate_every, save_every=args.save_every)
        if args.dry_run:
            print(json.dumps(record['continuation'], indent=2))
            return record['continuation']
        # mkdir is exclusive; a competing fresh launcher must not share this run.
        out.mkdir(parents=True, exist_ok=False)
        json_write(out/'training.json', record)
        for label in ('initial', 'latest'):
            save_training_checkpoint(out/'checkpoints'/label/'codec.pt', model, optimizer, progress, rng, scene_record)
        del model, optimizer
    info = record['continuation']
    if args.dry_run:
        print(json.dumps(info, indent=2))
        return info
    print(f'CHANNEL ONLY CONTINUATION: channel {info["parent_channel_step"]} -> {info["target_channel_step"]}; '
          f'global {info["parent_step"]} -> {info["target_step"]}; Adam/RNG retained.', flush=True)
    print('Both semantic modules FROZEN; latent MSE only; NO joint rendering updates. '
          'Rendering and color metrics are evaluation-only.', flush=True)
    print(f'Learning rates unchanged: {record["module_lrs"]}; validate/save every '
          f'{args.validate_every}/{args.save_every} global steps', flush=True)
    result = train(training_options(args, record, out), max_updates=max_updates)
    summary_path = out/'metrics/phase_summary.json'
    if summary_path.exists() and result['completed']['channel_bootstrap'] == record['phase_budgets']['channel_bootstrap']:
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary['channel_bootstrap'].update(optimizer_reset_at_boundary=False, optimizer_state_inherited=True)
        json_write(summary_path, summary)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
