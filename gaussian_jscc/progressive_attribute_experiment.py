"""Fresh progressive channel training with a frozen completed semantic model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
import tarfile
import time

import torch
from torch.nn.utils.rnn import pad_sequence

from .cli import device_for, seed_all
from .data import read_ply, to_raw
from .modular_checkpoint import capture_rng, restore_rng
from .modular_experiment import json_write
from .multiscene import manifest
from .optimization import preserved_rng
from .perceptual_metrics import LPIPSEvaluator
from .perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask
from .progressive_attribute_channel import (EIGHT_FAMILY, EIGHT_RATES, THREE_RATES, LayerConfig,
    LayeredAttributeCodec, load_semantic_slot, configure_stage, frozen_digest, state_digest)
from .progressive_attribute_evaluation import assess, layout_for
from .refinement_views import RefinementViewCycle
from .render_validation import append_json
from .semantic_attribute_codec import PRIVATE_ONLY, fit_loss_scales
from .semantic_attribute_metrics import directions_for
from .successive_multiscene_inputs import make_scene
from .training import full_scene_step

FAMILY = EIGHT_FAMILY


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--semantic-checkpoint', help='completed semantic attribute codec final.pt; required to train')
    p.add_argument('--out', required=True)
    p.add_argument('--manifest', help='relocated PLY/photo paths; scene fingerprints and camera split must match')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--three-tier-scratch', action='store_true',
                   help='8/24/48 symbols with frozen prefixes and a completed private128 semantic checkpoint')
    mode.add_argument('--three-tier-joint', action='store_true',
                   help='8/24/48 symbols with equally weighted joint channel training and frozen private128 semantics')
    p.add_argument('--bootstrap-steps', type=int,
                   help='total per layer across training scenes (500 eight-tier, 8000 three-tier)')
    p.add_argument('--render-steps', type=int, default=250, help='total per layer')
    p.add_argument('--neighbor-steps', type=int,
                   help='mixed-layout receiver-only updates (200 eight-tier, 0 three-tier)')
    p.add_argument('--bootstrap-lr', type=float, default=1e-4)
    p.add_argument('--render-lr', type=float, default=1e-5)
    p.add_argument('--neighbor-lr', type=float, default=1e-5)
    p.add_argument('--latent-weight', type=float, default=1.)
    p.add_argument('--attribute-weight', type=float, default=.01)
    p.add_argument('--ssim-weight', type=float, default=1/3)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--views-per-step', type=int, default=32)
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--save-every', type=int, help='latest checkpoint interval (50 eight-tier, 250 three-tier)')
    p.add_argument('--trials', type=int, default=4)
    p.add_argument('--snr', type=float, default=10.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--digital-python', default=str(Path.home()/'miniforge3/envs/gs_digital/bin/python'))
    p.add_argument('--digital-device', default='cuda')
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def schedule(args, layers):
    rows = []
    stages = ('joint',) if args.three_tier_joint else range(1, layers+1)
    for layer in stages:
        rows.extend([dict(stage=layer, phase='bootstrap', steps=args.bootstrap_steps, lr=args.bootstrap_lr),
                     dict(stage=layer, phase='render', steps=args.render_steps, lr=args.render_lr)])
    if args.neighbor_steps:
        rows.append(dict(stage=layers+1, phase='neighbor', steps=args.neighbor_steps, lr=args.neighbor_lr))
    return rows


def validate_settings(args):
    if min(args.bootstrap_steps, args.render_steps, args.views_per_step, args.validate_every,
           args.save_every, args.trials, args.cpu_threads, args.hidden) < 1 or args.neighbor_steps < 0:
        raise ValueError('positive training/view/interval counts required')
    for key in ('bootstrap_lr', 'render_lr', 'neighbor_lr', 'latent_weight', 'attribute_weight', 'ssim_weight'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError('positive finite '+key+' required')
    if not math.isfinite(args.snr):
        raise ValueError('finite SNR required')


def prepare(args):
    three = args.three_tier_scratch or args.three_tier_joint
    if args.bootstrap_steps is None:
        args.bootstrap_steps = 8000 if three else 500
    if args.neighbor_steps is None:
        args.neighbor_steps = 0 if three else 200
    if args.save_every is None:
        args.save_every = 250 if three else 50
    validate_settings(args)
    cfg = LayerConfig(rates=THREE_RATES if three else EIGHT_RATES, hidden=args.hidden)
    phases = schedule(args, cfg.layer_count)
    plan = dict(version=cfg.family, channel=cfg.to_dict(), schedule=phases,
        total_updates=sum(phase['steps'] for phase in phases),
        semantic_slot=dict(checkpoint=args.semantic_checkpoint, accepted=['shared128', 'shared256', 'split128', 'split256', 'split_full_geometry128']),
        settings={k: v for k, v in vars(args).items()
                  if k not in ('resume', 'dry_run', 'out', 'device', 'three_tier_scratch', 'three_tier_joint')},
        initialization='fresh channel weights; frozen completed semantic checkpoint',
        objective='source RGB MSE + ssim_weight*(1-SSIM) + latent_weight*latent_MSE + attribute_weight*attribute_loss',
        bootstrap_objective='latent_weight*latent_MSE + attribute_weight*attribute_loss',
        power='binary16 block/layer RMS over actual layer donors; fixed delivery plan; paired full-slot AWGN',
        selection='fixed final; validation reports only; tests never select weights',
        cost_protocol='digital_side_file_v1; actual LDPC4096/6144 + 16QAM; no ARQ',
        cache='frozen semantic latents in CPU RAM, rebuilt on resume; never cache noisy receiver outputs')
    if three:
        plan['semantic_slot']['accepted'] = [PRIVATE_ONLY]
        plan['bootstrap_validation_steps'] = sorted(
            {v for v in (500, 2000, 4000, args.bootstrap_steps) if v <= args.bootstrap_steps})
        plan['budget_reference'] = 'old 8/24/48 main had 8000 noisy minibatch updates for each prefix'
        plan['prefix_training'] = 'freeze the semantic model and all channel layers except the current increment'
    if args.three_tier_joint:
        plan['training_mode'] = 'joint_prefix_mean'
        plan['prefix_training'] = 'freeze the semantic model; average the three prefix gradients before each channel update'
        plan['supervision_per_tier'] = dict(bootstrap=args.bootstrap_steps, render=args.render_steps)
        plan['trainable_prefixes'] = list(cfg.rates[1:])
    if args.dry_run and not args.semantic_checkpoint:
        if not three:
            from .semantic_attribute_codec import SPLIT128_LAYOUTS
            plan['semantic_slot']['accepted'].extend(SPLIT128_LAYOUTS)
        return None, None, json.loads(json.dumps(plan))
    if not args.semantic_checkpoint:
        raise ValueError('set --semantic-checkpoint after the semantic experiment finishes')
    source = Path(args.semantic_checkpoint).resolve()
    out = Path(args.out).resolve()
    if out.is_relative_to(source.parent) or source.is_relative_to(out):
        raise ValueError('channel output must be separate from the semantic checkpoint')
    semantic, saved = load_semantic_slot(source)
    if three and semantic.arm != PRIVATE_ONLY:
        raise ValueError(f'three-tier training requires {PRIVATE_ONLY}/checkpoints/final.pt')
    if semantic.arm not in plan['semantic_slot']['accepted']:
        plan['semantic_slot']['accepted'].append(semantic.arm)
    plan['semantic_slot'].update(arm=semantic.arm, latent_dim=semantic.cfg.latent_dim,
        config=semantic.cfg.to_dict(), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        state_sha256=state_digest(semantic.state_dict()))
    specs = saved['plan']['selected_scenes']+[saved['plan']['heldout_scene']]
    if args.manifest:
        locations = {s['name']: s for s in manifest(args.manifest)}
        specs = [dict(s, **{k: locations[s['name']][k] for k in ('ply', 'source')}) for s in specs]
    train = [s for s in specs if s['role'] == 'train']
    if not train or any(row['steps'] % len(train) for row in plan['schedule']):
        raise ValueError('each phase budget must be divisible by the training scene count')
    for spec in specs:
        record = saved['scene_records'][spec['name']]
        if spec['role'] == 'train' and len(record['train_view_names']) < args.views_per_step:
            raise ValueError('insufficient source training views')
        names = record['train_view_names']+record['validation_view_names']+record['test_view_names']
        if len(names) != len(set(names)):
            raise ValueError('source camera splits overlap')
        for key in ('ply', 'source'):
            if not Path(spec[key]).exists():
                raise FileNotFoundError(spec[key])
    plan['selected_scenes'] = specs
    plan['scene_records'] = saved['scene_records']
    plan['camera_policy'] = 'inherit semantic train/validation/test identities exactly; train rotates independently per phase'
    return semantic, saved, json.loads(json.dumps(plan))


@torch.no_grad()
def cache_scene(model, scene):
    device = next(model.parameters()).device
    model.eval(); model.set_scene_geometry(scene.geometry)
    batches, blocks = [], []
    for f, ids in zip(scene.groups, scene.group_ids):
        f, active = f.to(device), (ids >= 0).to(device)
        y = model.semantic.encode(f, active)
        data = torch.cat((f, y), -1).cpu()
        batches.append(data)
        blocks.extend(data[i, :int(valid.sum())] for i, valid in enumerate(ids >= 0))
    return dict(batches=batches, blocks=blocks)


def sample_blocks(blocks, count, visit, seed):
    selected, previous_epoch, permutation = [], None, None
    for position in range((visit-1)*count, visit*count):
        epoch, offset = divmod(position, len(blocks))
        if epoch != previous_epoch:
            permutation = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(seed+epoch*100003))
            previous_epoch = epoch
        selected.append(blocks[int(permutation[offset])])
    data = pad_sequence(selected, batch_first=True)
    active = torch.arange(data.shape[1])[None] < torch.tensor([len(b) for b in selected])[:, None]
    return data, active


def batch_objective(model, scene, data, q, cameras, scales, args):
    from .semantic_attribute_codec import attribute_objective
    f, y, active = data[..., :59], data[..., 59:], q > 0
    pred, yh, wire = model.forward_cached(f, y, q, args.snr)
    latent = (yh[active]-y[active]).square().mean()
    attrs, terms = attribute_objective(pred[active], f[active], scene.geometry, model,
        directions_for(scene, f, active, cameras), scales)
    auxiliary = args.latent_weight*latent+args.attribute_weight*attrs
    terms = dict(terms, latent=latent, attributes=attrs, transmitted_power=wire['transmitted_power'])
    return to_raw(pred[active], scene.geometry, model), auxiliary, terms


def train_update(model, scene, cache, phase, visit, cameras, scales, args, index):
    device = next(model.parameters()).device
    model.train(); model.set_scene_geometry(scene.geometry)
    layer = phase['stage']
    if phase['phase'] == 'bootstrap':
        data, active = sample_blocks(cache['blocks'], scene.record['blocks_per_batch'], visit,
                                     args.seed+index*1009+layer*100003)
        q = active.long()*layer
        _, loss, terms = batch_objective(model, scene, data.to(device), q.to(device), cameras, scales, args)
        loss.backward()
        return loss.detach(), {k+'_loss': float(v.detach()) for k, v in terms.items()}
    layouts = layout_for(scene, 'mixed' if phase['phase'] == 'neighbor' else layer,
                         args.seed+visit*1009+index, layers=model.layer_config.layer_count)
    objective = PerceptualImageObjective('mse_ssim', args.ssim_weight, device)
    task = PerceptualMultiViewRenderTask(cameras, scene.reference, scene.degree, scene.white_background, objective)
    def forward(data, q):
        return batch_objective(model, scene, data, q, cameras, scales, args)
    # Auxiliary already contains its explicit latent/attribute weights.
    return full_scene_step(model, list(zip(cache['batches'], layouts)), scene.geometry, args.snr,
        'awgn', task, attr_weight=1., mode='replay', profile=True, batch_forward=forward)


def save(path, model, optimizer, progress, plan, scales):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(dict(family=plan['version'], plan=plan, state=model.state_dict(),
        optimizer=optimizer.state_dict() if optimizer else None, progress=progress,
        rng=capture_rng(), loss_scales=scales), temporary)
    temporary.replace(path)


def write_review(out):
    out = Path(out)
    rows = []
    for path in sorted((out/'evaluation').rglob('summary.json')):
        value = json.loads(path.read_text(encoding='utf8'))
        rows.extend(dict(r, evaluation=str(path.parent.relative_to(out/'evaluation'))) for r in value['rows'])
    delivered = [r for r in rows if 'actual_delivery' in r]
    delivery_summary = []
    for scene, layer in sorted({(r['scene'], str(r['layer'])) for r in delivered}):
        trials = [r['actual_delivery'] for r in delivered if r['scene'] == scene and str(r['layer']) == layer]
        successes = [t for t in trials if t['status'] == 'success']
        metrics = ({k: sum(t['metrics'][k] for t in successes)/len(successes) for k in successes[0]['metrics']}
                   if successes else None)
        delivery_summary.append(dict(scene=scene, layer=layer, trials=len(trials),
            success_rate=len(successes)/len(trials), success_conditioned_metrics=metrics,
            mean_sent_channel_uses=sum(t['total_channel_uses'] for t in trials)/len(trials)))
    plan_file = out/'plan.json'
    plan = json.loads(plan_file.read_text(encoding='utf8')) if plan_file.is_file() else {}
    review = dict(version=plan.get('version', FAMILY), rows=rows, actual_delivery=delivery_summary,
        cost_unit='complex_channel_use_per_delivery', selection='fixed_final')
    if 'training_mode' in plan:
        review.update(training_mode=plan['training_mode'], supervision_per_tier=plan['supervision_per_tier'])
    json_write(out/'review_summary.json', review)
    temporary = out/'review.tmp.tar.gz'
    with tarfile.open(temporary, 'w:gz') as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.png', '.md'):
                archive.add(p, arcname=str(p.relative_to(out)))
    temporary.replace(out/'review.tar.gz')


def main(args, max_updates=None):
    semantic, source, plan = prepare(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return plan
    out = Path(args.out).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError('use a new OUT or --resume')
    if args.resume and (not (out/'plan.json').exists() or
            json.loads((out/'plan.json').read_text(encoding='utf8')) != plan):
        raise ValueError('resume requires the same semantic checkpoint and experiment settings')
    if args.resume and (out/'complete.json').exists():
        write_review(out); return json.loads((out/'complete.json').read_text(encoding='utf8'))
    if not Path(args.digital_python).is_file():
        raise FileNotFoundError('actual final side delivery requires --digital-python: '+args.digital_python)
    device = device_for(args.device)
    torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    model = LayeredAttributeCodec(semantic, LayerConfig(**plan['channel'])).to(device)
    layers = model.layer_config.layer_count
    out.mkdir(parents=True, exist_ok=True); json_write(out/'plan.json', plan)
    (out/'metrics').mkdir(exist_ok=True)
    progress = dict(step=0, phase_index=0, phase_step=0, cycles={}, evaluated=[], frozen=None)
    saved = None
    if args.resume:
        saved = torch.load(out/'checkpoints/latest.pt', map_location='cpu', weights_only=True)
        if saved['family'] != plan['version'] or saved['plan'] != plan:
            raise ValueError('checkpoint identity mismatch')
        model.load_state_dict(saved['state']); progress = saved['progress']
        log = out/'metrics/loss.jsonl'
        if log.exists():
            lines = [line for line in log.read_text().splitlines() if json.loads(line)['step'] <= progress['step']]
            log.write_text(''.join(line+'\n' for line in lines), encoding='utf8')
    specs, scenes, train_raws = plan['selected_scenes'], [], []
    for spec in specs:
        old = source['scene_records'][spec['name']]
        scene_args = argparse.Namespace(**source['plan']['settings'])
        for key in ('resolution', 'images', 'white_background', 'blocks_per_batch', 'views_per_step'):
            setattr(scene_args, key, old[key])
        scene_args.validation_views = len(old['validation_view_names'])
        scene_args.test_views = len(old['test_view_names']); scene_args.net_bits_per_use = 2.
        raw, degree = read_ply(spec['ply'])
        if spec['role'] == 'train':
            train_raws.append(raw)
        scene = make_scene(spec, raw, degree, model, scene_args, out, old=old)
        scenes.append(scene)
    scales = fit_loss_scales(train_raws); del train_raws
    if saved and scales != saved['loss_scales']:
        raise ValueError('training-only loss scales changed')
    json_write(out/'attribute_statistics.json', scales)
    caches = {}
    for scene in scenes:
        print(f'Caching frozen {model.semantic.arm} latent: {scene.name}', flush=True)
        caches[scene.name] = cache_scene(model, scene)
    from .progressive_attribute_preflight import verify_execution
    print(f'Checking {layers} prefixes and received-file decoding on {device}', flush=True)
    json_write(out/'execution_check.json', verify_execution(model, scenes[0],
               caches[scenes[0].name]['batches'], args, out))
    from comparison_baselines.run import _execute
    _execute(args.digital_python, 'comparison_baselines.digital_link',
             ['self-test', '--device', args.digital_device, '--snr-db', 30], out/'digital_preflight.log')
    train_scenes = [s for s in scenes if s.spec['role'] == 'train']
    cycles = {}
    tier_phases = {}
    for pi, phase in enumerate(plan['schedule']):
        joint = phase['stage'] == 'joint'
        tiers = range(1, layers+1) if joint else (phase['stage'],)
        tier_phases[pi] = []
        for tier in tiers:
            # Match the frozen run's camera cycle seed for each tier/objective pair.
            cycle_phase = 2*(tier-1)+pi if joint else (2*layers if args.three_tier_joint else pi)
            tier_phases[pi].append((cycle_phase, dict(phase, stage=tier)))
            for scene in train_scenes:
                key = f'{cycle_phase}/{scene.name}'
                cycles[key] = RefinementViewCycle([str(c.image_name) for c in scene.cameras], args.seed,
                    scene.name, cycle_phase, progress['cycles'].get(key))
    perceptual = LPIPSEvaluator(device) if args.lpips else None
    if saved:
        restore_rng(saved['rng'])
    executed = 0
    optimizer = None
    def checkpoint(label='latest'):
        progress['cycles'] = {k: v.state_dict() for k, v in cycles.items()}
        save(out/'checkpoints'/f'{label}.pt', model, optimizer, progress, plan, scales)
    def validation(label, layers):
        if label in progress['evaluated']:
            return
        if args.three_tier_scratch or args.three_tier_joint:
            print(f'Validation {label} at global step {progress["step"]}', flush=True)
        with preserved_rng(device):
            for scene in train_scenes:
                rows = assess(model, scene, caches[scene.name]['batches'], args, out/'evaluation'/label/scene.name,
                              scope='validation', layers=layers, perceptual=perceptual)
                if (args.three_tier_scratch or args.three_tier_joint) and rows:
                    awgn = {j: fmean(r['source_psnr'] for r in rows if r['path'] == 'awgn' and r['layer'] == j)
                            for j in layers}
                    details = ', '.join(
                        f'{j if j == "mixed" else model.layer_config.rates[j]}={value:.3f} dB'
                        for j, value in awgn.items())
                    gain = ''.join(f' | paired {model.layer_config.rates[a]}->{model.layer_config.rates[b]} '
                                   f'{awgn[b]-awgn[a]:+.3f} dB' for a, b in zip(layers, layers[1:]))
                    print(f'{label} {scene.name} | AWGN mean PSNR {details}{gain}', flush=True)
        progress['evaluated'].append(label); checkpoint()
    for pi in range(progress['phase_index'], len(plan['schedule'])):
        phase = plan['schedule'][pi]
        joint = phase['stage'] == 'joint'
        tasks = tier_phases[pi]
        optimizer = configure_stage(model, phase['stage'], phase['lr'])
        if saved and saved['optimizer'] is not None and pi == progress['phase_index']:
            optimizer.load_state_dict(saved['optimizer'])
        saved = None
        if progress['frozen'] is None:
            progress['frozen'] = frozen_digest(model)
        if progress['frozen'] != frozen_digest(model):
            raise RuntimeError('frozen modules changed at resume/stage transition')
        for step in range(progress['phase_step']+1, phase['steps']+1):
            index = (step-1) % len(train_scenes); scene = train_scenes[index]
            visit = (step-1)//len(train_scenes)+1
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            results = {}
            for cycle_phase, task in tasks:
                tier = task['stage']
                cycle = cycles[f'{cycle_phase}/{scene.name}']
                cameras = [scene.cameras[i] for i in cycle.next(args.views_per_step)]
                loss, metrics = train_update(model, scene, caches[scene.name], task, visit, cameras, scales, args, index)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'nonfinite training loss at layer {tier}')
                results[tier] = dict(loss=float(loss), views=[str(c.image_name) for c in cameras],
                                     coverage=cycle.coverage(), **metrics)
            if joint:
                # Each tier computes its own full-scene VJP. Reuse .grad to avoid retaining three render graphs.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(len(tasks))
                metrics = dict(loss=fmean(r['loss'] for r in results.values()), tier_metrics=results)
            else:
                metrics = results[phase['stage']]
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                 float('inf'), error_if_nonfinite=True)
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            progress['step'] += 1; progress['phase_step'] = step
            row = dict(step=progress['step'], phase=phase['phase'], layer=phase['stage'], phase_step=step,
                scene=scene.name, grad_norm=float(norm), seconds=time.perf_counter()-started, **metrics)
            append_json(out/'metrics/loss.jsonl', row)
            if step == 1 or step % 10 == 0:
                label = 'joint' if joint else f'layer {phase["stage"]}'
                detail = ', '.join(f'{model.layer_config.rates[tier]}={r["loss"]:.6g}'
                                   for tier, r in results.items()) if joint else ''
                print(f'{label} {phase["phase"]} {step}/{phase["steps"]}: {scene.name}, loss={row["loss"]:.7g}, '
                      f'views={len(cameras)}, sec={row["seconds"]:.2f}'+(f' | {detail}' if detail else ''), flush=True)
            if step % args.save_every == 0 or step == phase['steps']:
                if frozen_digest(model) != progress['frozen']:
                    raise RuntimeError('frozen semantic/prior channel layers changed')
                checkpoint()
            if phase['phase'] != 'bootstrap' and step % args.validate_every == 0 and step != phase['steps']:
                validation(f'monitor_{progress["step"]:06d}',
                           ['mixed'] if phase['phase'] == 'neighbor' else list(results))
            if phase['phase'] == 'bootstrap' and step in plan.get('bootstrap_validation_steps', ()):
                label = 'joint' if joint else f'layer_{phase["stage"]:02d}'
                validation(f'{label}_bootstrap_{step:05d}', list(results))
            executed += 1
            if max_updates is not None and executed >= max_updates:
                checkpoint(); return dict(status='interrupted', step=progress['step'])
        if frozen_digest(model) != progress['frozen']:
            raise RuntimeError('completed phase changed frozen modules')
        if phase['phase'] != 'bootstrap':
            stage = phase['stage']
            label = 'joint_render' if joint else f'layer_{stage:02d}'
            checkpoint(label)
            checked = ['mixed'] if phase['phase'] == 'neighbor' else [task['stage'] for _, task in tasks]
            if args.three_tier_scratch and phase['phase'] == 'render' and stage > 1:
                checked = [stage-1, stage]
            validation(label, checked)
        progress.update(phase_index=pi+1, phase_step=0, frozen=None)
        optimizer = None; checkpoint()
    if state_digest(model.semantic.state_dict()) != plan['semantic_slot']['state_sha256']:
        raise RuntimeError('semantic model changed')
    checkpoint('final')
    with preserved_rng(device):
        for scene in scenes:
            assess(model, scene, caches[scene.name]['batches'], args, out/'evaluation/final'/scene.name,
                scope='test', layers=list(range(1, layers+1))+['mixed'], perceptual=perceptual,
                actual=True, core_comparison=args.neighbor_steps > 0)
    complete = dict(status='complete', version=plan['version'], step=progress['step'], semantic_frozen_verified=True,
                    codec_id=model.identity(), rates=list(model.layer_config.rates), selection='fixed_final')
    if args.three_tier_joint:
        complete.update(training_mode=plan['training_mode'], supervision_per_tier=plan['supervision_per_tier'])
    json_write(out/'complete.json', complete); write_review(out)
    print(json.dumps(complete), flush=True)
    return complete


if __name__ == '__main__':
    main(parser().parse_args())
