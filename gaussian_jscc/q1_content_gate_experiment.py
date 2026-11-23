"""Paired q1 continuation and a training-scene mean-gate intervention."""
import argparse
import copy
import json
import math
from pathlib import Path
from statistics import fmean
import tarfile
import time

import torch

from comparison_baselines.run import _execute
from .cli import device_for, seed_all
from .data import read_ply
from .modular_experiment import json_write
from .multiscene import manifest
from .perceptual_metrics import LPIPSEvaluator
from .progressive_attribute_channel import (THREE_FAMILY, THREE_RATES, LayerConfig, LayeredAttributeCodec,
    configure_stage, frozen_digest, load_semantic_slot, state_digest)
from .progressive_attribute_evaluation import assess
from .progressive_attribute_experiment import cache_scene, save, train_update
from .progressive_attribute_preflight import verify_execution
from .progressive_attribute_supervision import WEIGHTS, fit_color_scales
from .q1_content_gate import GatedQ1Encoder
from .refinement_views import RefinementViewCycle
from .render_validation import append_json
from .semantic_attribute_codec import PRIVATE_ONLY, fit_loss_scales
from .successive_multiscene_inputs import make_scene


FAMILY = 'q1_content_gate_controls_v1'
ARMS = ('baseline', 'content_gate')
SNRS = (4., 10., 16.)
SCENES = ('truck', 'playroom')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='completed layer_01.pt')
    p.add_argument('--semantic-checkpoint', help='relocated copy of the original semantic final.pt')
    p.add_argument('--manifest', help='relocated scene paths; original camera/PLY fingerprints still apply')
    p.add_argument('--out', required=True)
    p.add_argument('--bootstrap-steps', type=int, default=1000)
    p.add_argument('--render-steps', type=int, default=500)
    p.add_argument('--views-per-step', type=int, default=32)
    p.add_argument('--trials', type=int, default=4)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--digital-python', default=str(Path.home()/'miniforge3/envs/gs_digital/bin/python'))
    p.add_argument('--digital-device', default='cuda')
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    return p


def prepare(args):
    counts = (args.bootstrap_steps, args.render_steps, args.views_per_step, args.trials,
              args.save_every, args.cpu_threads)
    if min(counts) < 1 or any(n % len(SCENES) for n in counts[:2]) or args.seed < 0:
        raise ValueError('positive counts and even phase budgets are required for the two training scenes')
    path = Path(args.source).resolve()
    out = Path(args.out).resolve()
    if path.is_relative_to(out) or out.is_relative_to(path.parent):
        raise ValueError('use an output directory separate from the source checkpoint')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    source_plan, progress = saved['plan'], saved['progress']
    phases = source_plan['schedule'][:2]
    if (saved['family'] != THREE_FAMILY or len(phases) != 2 or
            [(p['stage'], p['phase']) for p in phases] != [(1, 'bootstrap'), (1, 'render')] or
            progress['step'] != sum(p['steps'] for p in phases) or progress['phase_index'] != 1 or
            progress['phase_step'] != phases[1]['steps']):
        raise ValueError('expected the fixed q1 endpoint from independent 8/24/48 prefix training')
    cfg = LayerConfig(**source_plan['channel'])
    if cfg.rates != THREE_RATES or bool(saved['state']['neighbor_enabled']):
        raise ValueError('expected 8/24/48 prefixes with a q1 endpoint before mixed neighbor training')
    semantic_path = Path(args.semantic_checkpoint or source_plan['semantic_slot']['checkpoint']).resolve()
    if semantic_path.is_relative_to(out) or out.is_relative_to(semantic_path.parent):
        raise ValueError('use an output directory separate from the semantic checkpoint')
    semantic, semantic_saved = load_semantic_slot(semantic_path)
    semantic_id = state_digest(semantic.state_dict())
    if (semantic.arm != PRIVATE_ONLY or semantic.cfg.latent_dim != 128 or
            semantic_id != source_plan['semantic_slot']['state_sha256'] or
            json.dumps(semantic.cfg.to_dict(), sort_keys=True) !=
            json.dumps(source_plan['semantic_slot']['config'], sort_keys=True)):
        raise ValueError('semantic checkpoint does not match the frozen private128 source')
    specs = {s['name']: s for s in semantic_saved['plan']['selected_scenes'] if s['role'] == 'train'}
    specs = [dict(specs[name]) for name in SCENES]
    if args.manifest:
        locations = {s['name']: s for s in manifest(args.manifest)}
        for spec in specs:
            spec.update({k: locations[spec['name']][k] for k in ('ply', 'source')})
    args.grouped_attribute_supervision = True
    args.ssim_weight = 1/3
    plan = dict(version=FAMILY, source_checkpoint=str(path), source_step=progress['step'],
        source_state_sha256=state_digest(saved['state']), semantic_checkpoint=str(semantic_path),
        semantic_state_sha256=semantic_id, selected_scenes=specs,
        settings={k: getattr(args, k) for k in ('bootstrap_steps', 'render_steps', 'views_per_step',
            'trials', 'save_every', 'seed', 'lpips', 'ssim_weight', 'device', 'digital_device', 'cpu_threads')},
        channel=cfg.to_dict(), symbols_per_gaussian=8, snrs_db=SNRS, arms=ARMS,
        schedule=[dict(stage=1, phase='bootstrap', steps=args.bootstrap_steps, lr=3e-5),
                  dict(stage=1, phase='render', steps=args.render_steps, lr=1e-5)],
        attribute_weights=WEIGHTS[1], latent_loss_weight=0.,
        gate=dict(width=32, initial=1., coefficient_range=[.5, 1.5], location='q1 transmitter input'),
        mean_gate='point-weighted over Truck and Playroom training latents separately for each SNR',
        cost_protocol='digital_side_file_v1', selection='fixed_final')
    base = LayeredAttributeCodec(semantic, cfg)
    base.load_state_dict(saved['state'], strict=True)
    if state_digest(base.semantic.state_dict()) != semantic_id:
        raise ValueError('source channel checkpoint and semantic checkpoint have different semantic weights')
    return base, semantic_saved, json.loads(json.dumps(plan))


def load_scenes(base, source, specs, args, out):
    scenes = []
    for spec in specs:
        old = source['scene_records'][spec['name']]
        train, val, test = (old[k] for k in ('train_view_names', 'validation_view_names', 'test_view_names'))
        if len(train) < args.views_per_step or not val or not test or len(set(train+val+test)) != len(train+val+test):
            raise ValueError(f'insufficient or overlapping camera splits for {spec["name"]}')
        options = argparse.Namespace(**source['plan']['settings'])
        for key in ('resolution', 'images', 'white_background', 'blocks_per_batch', 'views_per_step'):
            setattr(options, key, old[key])
        options.validation_views = len(val)
        options.test_views = len(test)
        options.net_bits_per_use = 2.
        raw, degree = read_ply(spec['ply'])
        scenes.append(make_scene(spec, raw, degree, base, options, out, old=old))
    return scenes


def at_snr(args, snr, *, trials=None):
    return argparse.Namespace(**(vars(args) | dict(snr=snr, trials=args.trials if trials is None else trials)))


def score(model, scenes, caches, args, root, perceptual, *, scope, snrs, actual=False, trials=None):
    rows = []
    for snr in snrs:
        for scene in scenes:
            rows.extend(assess(model, scene, caches[scene.name]['batches'], at_snr(args, snr, trials=trials),
                root/f'{snr:g}db'/scene.name, scope=scope, layers=[1], perceptual=perceptual, actual=actual))
    return rows


@torch.no_grad()
def check_initial(base, gated, scene, cached, args, out):
    device = next(base.parameters()).device
    data = cached[0][:2].to(device)
    q = (scene.group_ids[0][:2] >= 0).to(device, torch.long)
    errors = {}
    for model in (base, gated):
        model.eval()
        model.set_scene_geometry(scene.geometry)
    for snr in SNRS:
        symbols_a, scales_a = base.encode(data[..., 59:], q, snr)
        symbols_b, scales_b = gated.encode(data[..., 59:], q, snr)
        torch.testing.assert_close(symbols_a, symbols_b, rtol=0, atol=0)
        torch.testing.assert_close(scales_a, scales_b, rtol=0, atol=0)
        pred = [m.forward_cached(data[..., :59], data[..., 59:], q, snr,
                generator=torch.Generator(device=device).manual_seed(args.seed))[0] for m in (base, gated)]
        torch.testing.assert_close(*pred, rtol=0, atol=0)
        errors[f'{snr:g}'] = float((pred[0]-pred[1]).abs().max())
    packet = verify_execution(gated, scene, cached, at_snr(args, 10.), out)
    return dict(status='passed', initial_reconstruction_max_error=errors, initial_packet_check=packet,
        gate_parameters=sum(p.numel() for p in gated.encoders[0].gate.parameters()))


def train_arm(model, scenes, caches, args, root, plan, scales, perceptual):
    device = next(model.parameters()).device
    checkpoint = root/'checkpoints/latest.pt'
    log = root/'metrics/loss.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    progress = dict(step=0, phase_index=0, phase_step=0, cycles={})
    saved = None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['plan'] != plan or saved['loss_scales'] != scales:
            raise ValueError('continuation checkpoint settings or training normalization changed')
        model.load_state_dict(saved['state'], strict=True)
        progress = saved['progress']
    if log.exists():
        keep = [line for line in log.read_text(encoding='utf8').splitlines()
                if line.endswith('}') and json.loads(line)['step'] <= progress['step']]
        pending = log.with_suffix('.tmp')
        pending.write_text(''.join(line+'\n' for line in keep), encoding='utf8')
        pending.replace(log)
    for pi in range(progress['phase_index'], len(plan['schedule'])):
        phase = plan['schedule'][pi]
        optimizer = configure_stage(model, 1, phase['lr'])
        fixed = frozen_digest(model)
        if saved and saved['optimizer'] is not None:
            optimizer.load_state_dict(saved['optimizer'])
        saved = None
        cycles = {s.name: RefinementViewCycle([str(c.image_name) for c in s.cameras], args.seed,
            s.name, pi, progress['cycles'].get(s.name)) for s in scenes}
        milestones = {phase['steps']}
        interval = 500 if pi == 0 else 250
        milestones.update(range(interval, phase['steps'], interval))

        def monitor(step):
            score(model, scenes, caches, args, root/'validation'/f'{phase["phase"]}_{step:05d}', perceptual,
                  scope='validation', snrs=(4., 10.), trials=1)

        if progress['phase_step'] in milestones:
            monitor(progress['phase_step'])
        for step in range(progress['phase_step']+1, phase['steps']+1):
            index = (step-1) % len(scenes)
            visit = (step-1)//len(scenes)+1
            scene = scenes[index]
            snr = SNRS[(visit-1+index+pi) % len(SNRS)]
            cycle = cycles[scene.name]
            views = [scene.cameras[i] for i in cycle.next(args.views_per_step)]
            noise_seed = args.seed+1000000+progress['step']
            torch.manual_seed(noise_seed)
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            loss, metrics = train_update(model, scene, caches[scene.name], phase, visit, views, scales,
                                         at_snr(args, snr), index)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite q1 comparison loss')
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                  float('inf'), error_if_nonfinite=True)
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            progress['step'] += 1
            progress['phase_step'] = step
            row = dict(step=progress['step'], phase=phase['phase'], phase_step=step, scene=scene.name,
                snr_db=snr, noise_seed=noise_seed, views=[str(c.image_name) for c in views], loss=float(loss),
                grad_norm=float(norm), **metrics)
            row['seconds'] = time.perf_counter()-started
            append_json(log, row)
            if step == 1 or step % 10 == 0:
                print(f'{root.name} {phase["phase"]} {step}/{phase["steps"]} {scene.name} '
                      f'{snr:g}dB loss={row["loss"]:.7g} sec={row["seconds"]:.2f}', flush=True)
            if step % args.save_every == 0 or step in milestones:
                if frozen_digest(model) != fixed:
                    raise RuntimeError('a frozen channel or semantic parameter changed')
                progress['cycles'] = {k: c.state_dict() for k, c in cycles.items()}
                save(checkpoint, model, optimizer, progress, plan, scales)
            if step in milestones:
                monitor(step)
        progress.update(phase_index=pi+1, phase_step=0, cycles={})
        save(checkpoint, model, None, progress, plan, scales)
    if state_digest(model.semantic.state_dict()) != plan['semantic_state_sha256']:
        raise RuntimeError('frozen semantic weights changed')
    save(root/'checkpoints/final.pt', model, None, progress, plan, scales)


@torch.no_grad()
def mean_gates(model, scenes, caches, path):
    identity = model.identity()
    if path.is_file():
        saved = json.loads(path.read_text(encoding='utf8'))
        if saved['codec_id'] != identity:
            raise ValueError('mean gate belongs to a different checkpoint')
        return saved['means']
    encoder = model.encoders[0]
    device = next(model.parameters()).device
    means = {}
    for snr in SNRS:
        total = torch.zeros(model.cfg.latent_dim, dtype=torch.float64, device=device)
        count = 0
        for scene in scenes:
            for data, ids in zip(caches[scene.name]['batches'], scene.group_ids):
                latent = data[..., 59:][ids >= 0].to(device)
                total += encoder.gate(latent, snr).sum(0, dtype=torch.float64)
                count += len(latent)
        means[f'{snr:g}'] = (total/count).float().tolist()
    json_write(path, dict(codec_id=identity, means=means, fitted_scenes=[s.name for s in scenes]))
    return means


def write_review(out, plan, rows):
    report = []
    for scene in SCENES:
        for snr in SNRS:
            noiseless = {}
            for noise in ('finite_noiseless', 'awgn'):
                arms = {}
                for arm, arm_rows in rows.items():
                    group = [row for row in arm_rows if row['scene'] == scene and row['snr_db'] == snr
                             and row['path'] == noise]
                    quality = [k for k in group[0] if k.startswith(('source_', 'photo_'))]
                    delivered = [r['actual_delivery'] for r in group if 'actual_delivery' in r]
                    successes = [r['metrics'] for r in delivered if r['status'] == 'success']
                    arms[arm] = dict(trials=len(group), **{k: fmean(r[k] for r in group) for k in quality},
                        attribute_mse={k: fmean(r['attribute_errors']['mse'][k] for r in group) for k in WEIGHTS[1]},
                        transmitted_power=fmean(r['diagnostics']['transmitted_power'] for r in group),
                        channel_uses=fmean(r['accounting']['total_channel_uses'] for r in group),
                        actual_trials=len(delivered),
                        actual_success_rate=len(successes)/len(delivered) if delivered else None,
                        actual_success_conditioned_quality={k: fmean(r[k] for r in successes) for k in quality}
                            if successes else None)
                    if noise == 'finite_noiseless':
                        noiseless[arm] = arms[arm]['source_psnr']
                    else:
                        arms[arm]['source_psnr_noise_loss_db'] = noiseless[arm]-arms[arm]['source_psnr']
                gated = arms['content_gate']['source_psnr']
                power_a, power_b = (arms[arm]['transmitted_power'] for arm in ARMS)
                report.append(dict(scene=scene, snr_db=snr, path=noise, arms=arms,
                    gate_minus_baseline_db=gated-arms['baseline']['source_psnr'],
                    gate_minus_mean_db=gated-arms['mean_gate']['source_psnr'],
                    gate_minus_baseline_power_db=10*math.log10(power_b/power_a) if min(power_a, power_b) > 0 else None))
    json_write(out/'review_summary.json', dict(version=FAMILY, selection='fixed_final',
        source_step=plan['source_step'], cost_unit='complex_channel_use_per_delivery', results=report))
    temp = out/'review.tmp.tar.gz'
    with tarfile.open(temp, 'w:gz') as archive:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.png'):
                archive.add(p, arcname=str(p.relative_to(out)))
    temp.replace(out/'review.tar.gz')


def main(args):
    out = Path(args.out).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError('use a new output or --resume')
    base, source, plan = prepare(args)
    if args.resume and (not (out/'plan.json').is_file() or
            json.loads((out/'plan.json').read_text(encoding='utf8')) != plan):
        raise ValueError('resume requires the original source and experiment settings')
    if (out/'complete.json').is_file():
        return json.loads((out/'complete.json').read_text(encoding='utf8'))
    if not Path(args.digital_python).is_file():
        raise FileNotFoundError(f'digital link interpreter not found: {args.digital_python}')
    device = device_for(args.device)
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    gated = copy.deepcopy(base)
    gated.encoders[0] = GatedQ1Encoder(gated.encoders[0], gated.cfg.latent_dim)
    base.to(device)
    gated.to(device)
    out.mkdir(parents=True, exist_ok=True)
    json_write(out/'plan.json', plan)
    scenes = load_scenes(base, source, plan['selected_scenes'], args, out)
    train_raws = [s.raw for s in scenes]
    scales = fit_loss_scales(train_raws) | fit_color_scales(train_raws)
    json_write(out/'attribute_statistics.json', scales)
    caches = {s.name: cache_scene(base, s) for s in scenes}
    json_write(out/'execution_check.json', check_initial(base, gated, scenes[0],
        caches[scenes[0].name]['batches'], args, out))
    _execute(args.digital_python, 'comparison_baselines.digital_link',
             ['self-test', '--device', args.digital_device, '--snr-db', 30], out/'digital_preflight.log')
    perceptual = LPIPSEvaluator(device) if args.lpips else None
    score(base, scenes, caches, args, out/'initial', perceptual, scope='validation', snrs=SNRS, trials=1)
    models = dict(zip(ARMS, (base, gated)))
    for arm, model in models.items():
        train_arm(model, scenes, caches, args, out/arm, plan, scales, perceptual)
    rows = {arm: score(model, scenes, caches, args, out/arm/'test', perceptual,
                       scope='test', snrs=SNRS, actual=True) for arm, model in models.items()}
    means = mean_gates(gated, scenes, caches, out/'mean_gates.json')
    rows['mean_gate'] = []
    for snr in SNRS:
        gated.encoders[0].mean_gate = torch.tensor(means[f'{snr:g}'], device=device)
        rows['mean_gate'].extend(score(gated, scenes, caches, args, out/'mean_gate'/'test', perceptual,
                                      scope='test', snrs=(snr,)))
    gated.encoders[0].mean_gate = None
    write_review(out, plan, rows)
    complete = dict(status='complete', version=FAMILY, arms=ARMS,
        steps_per_arm=args.bootstrap_steps+args.render_steps, semantic_frozen_verified=True,
        selection='fixed_final')
    json_write(out/'complete.json', complete)
    print(json.dumps(complete), flush=True)
    return complete


if __name__ == '__main__':
    main(parser().parse_args())
