"""Adapt shared private128 semantics and one progressive channel tier at a time."""
import argparse
import json
import math
from pathlib import Path
from statistics import fmean

import torch

from .cli import device_for, seed_all
from .data import read_ply, to_raw
from .modular_checkpoint import restore_rng
from .modular_config import ModularCodecConfig
from .modular_experiment import json_write
from .multiscene import manifest
from .optimization import preserved_rng
from .perceptual_metrics import LPIPSEvaluator
from .progressive_attribute_channel import (THREE_FAMILY, LayerConfig, LayeredAttributeCodec,
                                           configure_stage, frozen_digest)
from .progressive_attribute_evaluation import assess, layout_for
from .progressive_attribute_experiment import cache_scene, save, write_review
from .refinement_views import RefinementViewCycle
from .render_objective import MultiViewRenderTask
from .render_validation import append_json
from .semantic_attribute_codec import PRIVATE_ONLY, SemanticAttributeCodec
from .successive_multiscene_inputs import make_scene
from .training import full_scene_step


TIERS = (1, 2, 3)
LAYOUTS = (*TIERS, 'mixed')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='completed three-tier channel run')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', help='override scene paths, retaining the checkpoint splits')
    p.add_argument('--steps-per-scene', type=int, default=1500,
                   help='total per training scene, divided across consecutive q1, q2 and q3 phases')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--validate-every', type=int, default=250, help='updates per training scene')
    p.add_argument('--save-every', type=int, default=100, help='global updates')
    p.add_argument('--device', default='cuda')
    p.add_argument('--digital-python', help='defaults to the source run interpreter')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p


def prepare(args):
    run, out = args.run.resolve(), args.out.resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('joint output must be separate from the source run')
    if min(args.steps_per_scene, args.validate_every, args.save_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('positive step counts and finite positive learning rate required')
    if args.steps_per_scene < len(TIERS):
        raise ValueError('each scene needs at least one update per tier')
    saved = torch.load(run/'checkpoints/final.pt', map_location='cpu', weights_only=True)
    parent = saved['plan']
    complete = json.loads((run/'complete.json').read_text(encoding='utf8'))
    slot = parent['semantic_slot']
    if (saved['family'] != THREE_FAMILY or slot['arm'] != PRIVATE_ONLY or
            complete['status'] != 'complete' or saved['progress']['step'] != complete['step'] or
            complete['step'] != parent['total_updates'] or complete['rates'] != [0, 8, 24, 48] or
            saved['state']['neighbor_enabled'].item()):
        raise ValueError('expected a completed private128 three-tier backbone with no neighbor stage')
    semantic = SemanticAttributeCodec(slot['arm'], ModularCodecConfig.from_dict(slot['config']))
    model = LayeredAttributeCodec(semantic, LayerConfig(**parent['channel']))
    model.load_state_dict(saved['state'], strict=True)
    if model.identity() != complete['codec_id']:
        raise ValueError('source final weights differ from completed evaluation')
    specs = parent['selected_scenes']
    if args.manifest:
        locations = {s['name']: s for s in manifest(args.manifest)}
        specs = [dict(s, **{k: locations[s['name']][k] for k in ('source', 'ply')}) for s in specs]
    settings = parent['settings']
    evaluation = argparse.Namespace(**{k: settings[k] for k in
        ('snr', 'seed', 'trials', 'lpips', 'digital_device', 'cpu_threads', 'views_per_step')},
        digital_python=args.digital_python or settings['digital_python'], grouped_attribute_supervision=False,
        record_symbol_diagnostics=parent.get('record_symbol_diagnostics', False))
    scene_count = sum(s['role'] == 'train' for s in specs)
    per_tier, remainder = divmod(args.steps_per_scene, len(TIERS))
    schedule = []
    start = 0
    for tier in TIERS:
        visits = per_tier + int(tier <= remainder)
        end = start + visits*scene_count
        schedule.append(dict(tier=tier, steps_per_scene=visits, start_step=start, end_step=end))
        start = end
    plan = dict(version='progressive_attribute_tier_semantic_v1', source_run=str(run),
        source_step=complete['step'], source_codec_id=complete['codec_id'], channel=parent['channel'],
        semantic_slot=slot, selected_scenes=specs, scene_records=parent['scene_records'],
        steps_per_scene=args.steps_per_scene, total_updates=start, schedule=schedule, evaluation_layouts=list(LAYOUTS),
        trainable='shared semantic encoder and decoder plus the active tier channel encoder and decoder',
        optimizer='new Adam at each tier; inherit all model weights across tiers',
        lr=args.lr, validate_every_per_scene=args.validate_every, save_every=args.save_every,
        objective='source PLY render RGB MSE + MSE(recovered latent, stop_gradient(current semantic latent))',
        evaluation=vars(evaluation), device=args.device, cost_protocol=parent['cost_protocol'])
    return model, json.loads(json.dumps(plan)), evaluation


def update(model, scene, tier, visit, cameras, settings):
    model.train()
    model.set_scene_geometry(scene.geometry)
    layouts = layout_for(scene, tier, settings.seed+1009*visit, layers=3)
    task = MultiViewRenderTask(cameras, scene.reference, scene.degree, scene.white_background)

    def forward(features, q):
        active = q > 0
        y = model.semantic.encode(features, active)
        pred, recovered, _ = model.forward_cached(features, y, q, settings.snr)
        target = y[active].detach()
        mse = (recovered[active]-target).square().mean()
        return to_raw(pred[active], scene.geometry, model), mse, dict(latent=mse, target_power=target.square().mean())

    loss, metrics = full_scene_step(model, list(zip(scene.groups, layouts)), scene.geometry, settings.snr,
        'awgn', task, attr_weight=1., mode='replay', profile=True, batch_forward=forward)
    metrics['latent_target_power'] = metrics.pop('target_power_loss')
    metrics['latent_nmse'] = metrics['latent_loss']/max(metrics['latent_target_power'], 1e-12)
    return float(loss), metrics


def read_endpoint(root, scene, codec_id, settings, record):
    data = json.loads((root/'evaluation/final'/scene/'summary.json').read_text(encoding='utf8'))
    views = record['test_view_names']
    if data['scope'] != 'test' or [v['view'] for v in data['clean']['views']] != views:
        raise ValueError(f'{scene}: clean evaluation camera mismatch')
    rows = {}
    for row in data['rows']:
        key = (row['layer'], row['path'], row['trial'])
        if (key in rows or row['codec_id'] != codec_id or row['scene'] != scene or row['scope'] != 'test'
                or row['snr_db'] != settings.snr or row['seed'] != settings.seed+1000+row['trial']
                or [v['view'] for v in row['views']] != views):
            raise ValueError(f'{scene}: mismatched endpoint row {key}')
        if row['path'] == 'awgn' and row.get('actual_delivery', {}).get('status') not in ('success', 'failed'):
            raise ValueError(f'{scene}: missing actual delivery outcome for {key}')
        rows[key] = row
    expected = {(tier, path, t) for tier in LAYOUTS for path, trials in
                [('finite_noiseless', 1), ('awgn', settings.trials)] for t in range(trials)}
    if rows.keys() != expected:
        raise ValueError(f'{scene}: incomplete endpoint trials')
    return data['clean'], rows


def photo_comparison(before, after, trials):
    clean0, initial = before
    clean1, final = after
    def change(x, y):
        return dict(before=x, after=y, gain_db=y-x)
    comparison = dict(clean=change(clean0['photo_psnr'], clean1['photo_psnr']), layouts={})
    for tier in LAYOUTS:
        key = (tier, 'finite_noiseless', 0)
        finite = change(initial[key]['photo_psnr'], final[key]['photo_psnr'])
        pairs = [(initial[tier, 'awgn', t]['actual_delivery'], final[tier, 'awgn', t]['actual_delivery'])
                 for t in range(trials)]
        success = [(a['metrics']['photo_psnr'], b['metrics']['photo_psnr']) for a, b in pairs
                   if a['status'] == b['status'] == 'success']
        awgn = gap = None
        if success:
            awgn = change(fmean(a for a, _ in success), fmean(b for _, b in success))
            gap = dict(before=finite['before']-awgn['before'], after=finite['after']-awgn['after'])
            gap['reduction_db'] = gap['before']-gap['after']
        comparison['layouts'][str(tier)] = dict(finite_noiseless=finite, awgn=awgn, noise_penalty_db=gap,
                                                paired_successes=len(success), attempted=trials)
    return comparison


def main(args):
    run, out = args.run.resolve(), args.out.resolve()
    model, plan, settings = prepare(args)
    records = plan['scene_records']
    before = {s['name']: read_endpoint(run, s['name'], plan['source_codec_id'], settings, records[s['name']])
              for s in plan['selected_scenes']}
    if args.dry_run:
        budget = ', '.join(f"q{p['tier']}={p['steps_per_scene']}" for p in plan['schedule'])
        print(f"Verified source step {plan['source_step']}, codec {plan['source_codec_id']}, "
              f"{len(before)} paired test baselines; {budget} per scene, {plan['total_updates']} new updates.")
        return plan
    if not Path(settings.digital_python).is_file():
        raise FileNotFoundError(settings.digital_python)
    device = device_for(args.device)
    if args.resume:
        if json.loads((out/'plan.json').read_text(encoding='utf8')) != plan:
            raise ValueError('resume requires the same source, scenes and joint settings')
        if (out/'complete.json').exists():
            write_review(out)
            return
    else:
        out.mkdir(parents=True, exist_ok=False)
        json_write(out/'plan.json', plan)
    torch.set_num_threads(settings.cpu_threads)
    seed_all(settings.seed)
    model.to(device)
    progress = dict(step=0, tier=1, cycles={})
    latest = out/'checkpoints/latest.pt'
    saved = None
    log = out/'metrics/loss.jsonl'
    if args.resume and latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if saved['plan'] != plan:
            raise ValueError('checkpoint plan mismatch')
        model.load_state_dict(saved['state'])
        progress = saved['progress']
        if log.exists():
            lines = log.read_text(encoding='utf8').splitlines()[:progress['step']]
            if len(lines) != progress['step']:
                raise ValueError('training log shorter than the saved checkpoint')
            temporary = log.with_suffix('.tmp')
            temporary.write_text(''.join(line+'\n' for line in lines), encoding='utf8')
            temporary.replace(log)
    log.parent.mkdir(exist_ok=True)
    scenes = []
    for spec in plan['selected_scenes']:
        old = records[spec['name']]
        scene_args = argparse.Namespace(**old, validation_views=len(old['validation_view_names']),
            test_views=len(old['test_view_names']), net_bits_per_use=2.)
        raw, degree = read_ply(spec['ply'])
        scenes.append(make_scene(spec, raw, degree, model, scene_args, out, old=old))
        del raw
    train = [s for s in scenes if s.spec['role'] == 'train']
    cycles = {s.name: RefinementViewCycle([str(c.image_name) for c in s.cameras], settings.seed, s.name,
              'tier_semantic', progress['cycles'].get(s.name)) for s in train}
    perceptual = LPIPSEvaluator(device) if settings.lpips else None
    if saved:
        restore_rng(saved['rng'])

    def checkpoint(label, optimizer):
        if bool(model.neighbor_enabled) or frozen_digest(model) != progress['frozen_sha256']:
            raise RuntimeError(f"q{progress['tier']}: inactive channel layers or fixed statistics changed")
        progress['cycles'] = {k: v.state_dict() for k, v in cycles.items()}
        save(out/'checkpoints'/f'{label}.pt', model, optimizer, progress, plan, {})

    def evaluate(label, scenes_to_test, final=False):
        with preserved_rng(device):
            for scene in scenes_to_test:
                cached = cache_scene(model, scene)['batches']
                assess(model, scene, cached, settings, out/'evaluation'/label/scene.name,
                       scope='test' if final else 'validation', layers=LAYOUTS if final else TIERS,
                       perceptual=perceptual, actual=final)
                del cached

    interval = len(train)*args.validate_every
    for phase in plan['schedule'][progress['tier']-1:]:
        tier, start, end = phase['tier'], phase['start_step'], phase['end_step']
        optimizer = configure_stage(model, tier, args.lr, train_semantic=True)
        fixed = frozen_digest(model)
        if saved:
            if not start <= progress['step'] <= end or fixed != progress['frozen_sha256']:
                raise ValueError('resume checkpoint does not match the tier or its frozen weights')
            optimizer.load_state_dict(saved['optimizer'])
            saved = None
        else:
            progress.update(tier=tier, frozen_sha256=fixed)
            checkpoint('latest', optimizer)
        print(f"q{tier} semantic adaptation: train semantic endpoints and channel increment {tier}; "
              f"global steps {start+1}..{end}, {phase['steps_per_scene']} updates per scene, "
              f"completed {progress['step']}/{plan['total_updates']}.", flush=True)
        for step in range(progress['step'], end):
            phase_step = step-start
            if phase_step and phase_step % interval == 0:
                evaluate(f'monitor_{step:06d}', train)
            visit, index = divmod(step, len(train))
            scene = train[index]
            cameras = [scene.cameras[i] for i in cycles[scene.name].next(settings.views_per_step)]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = update(model, scene, tier, visit+1, cameras, settings)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            progress['step'] = step+1
            append_json(log, dict(step=step+1, scene=scene.name, scene_step=visit+1, layout=tier,
                        tier_step=phase_step+1, loss=loss, grad_norm=float(norm),
                        views=[str(c.image_name) for c in cameras], **metrics))
            if phase_step == 0 or (step+1) % 10 == 0:
                print(f"q{tier} {phase_step+1}/{end-start} {scene.name} loss={loss:.7g}", flush=True)
            if (step+1) % args.save_every == 0 or (phase_step+1) % interval == 0 or step+1 == end:
                checkpoint('latest', optimizer)
        checkpoint(f'q{tier}_end', optimizer)
        evaluate(f'q{tier}_end', train)
    checkpoint('final', optimizer)
    evaluate('final', scenes, final=True)
    identity = model.identity()
    comparison = {s.name: photo_comparison(before[s.name],
        read_endpoint(out, s.name, identity, settings, records[s.name]), settings.trials) for s in scenes}
    json_write(out/'photo_comparison.json', comparison)
    json_write(out/'complete.json', dict(status='complete', version=plan['version'], step=progress['step'],
               source_step=plan['source_step'], codec_id=identity, rates=list(model.layer_config.rates)))
    write_review(out)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main(parser().parse_args())
