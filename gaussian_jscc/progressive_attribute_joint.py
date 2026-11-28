"""Render adaptation for a completed private128 progressive channel."""
import argparse
import json
import math
from copy import deepcopy
from dataclasses import replace
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
from .perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask
from .position_delivery import delivered_positions
from .progressive_attribute_alternation import alternating_schedule, refinement_noise_states, validation_response
from .progressive_attribute_channel import (THREE_FAMILY, LayerConfig, LayeredAttributeCodec,
                                           configure_stage, frozen_digest, load_semantic_slot, state_digest)
from .progressive_attribute_evaluation import assess, layout_for
from .progressive_attribute_experiment import cache_scene, clean_response_loss, save, write_review
from .refinement_views import RefinementViewCycle
from .render_objective import MultiViewRenderTask
from .render_validation import append_json
from .semantic_attribute_codec import PRIVATE_ONLY, SemanticAttributeCodec, attribute_objective
from .semantic_attribute_metrics import directions_for
from .successive_multiscene_inputs import make_scene
from .training import full_scene_step


TIERS = (1, 2, 3)
LAYOUTS = (*TIERS, 'mixed')
REFINEMENT_FAMILY = 'progressive_attribute_fixed_semantics_refinement_v1'


def rotation_channel_tiers(layout):
    return TIERS if layout == 'mixed' else (layout,)


def rotation_visit(visit, finite_every):
    if finite_every:
        group, offset = divmod(visit, finite_every+1)
        if offset == finite_every:
            return 'finite_noiseless', group
        visit = group*finite_every+offset
    return 'awgn', visit


def set_rotation_layout(model, layout):
    """Keep the optimizer's parameter groups fixed along the layout cycle."""
    model.train()
    model.semantic.semantic_encoder.eval().requires_grad_(False)
    model.semantic.semantic_decoder.train().requires_grad_(True)
    active = rotation_channel_tiers(layout)
    for tier, (encoder, decoder) in enumerate(zip(model.encoders, model.decoders), 1):
        enabled = tier in active
        encoder.train(enabled).requires_grad_(enabled)
        decoder.train(enabled).requires_grad_(enabled)
    model.neighbor.eval()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='completed three-tier channel run')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', help='override scene paths, retaining the checkpoint splits')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--alternate-tiers', action='store_true',
                      help='alternate two rotation and fixed-channel refinement passes, then refine mixed')
    mode.add_argument('--rotate-tiers', action='store_true',
                   help='cycle q1/q2/q3/mixed each scene; freeze the semantic encoder')
    mode.add_argument('--refine-tiers', action='store_true',
                   help='refine q1, q2, q3 and mixed in order with both semantic endpoints frozen')
    p.add_argument('--steps-per-scene', type=int,
                   help='AWGN updates per scene (default: 1000 rotating/refining, 1500 adapting)')
    p.add_argument('--finite-noiseless-every', type=int, metavar='N',
                   help='add one finite noiseless rotating update per scene after every N AWGN updates')
    p.add_argument('--attribute-weight', type=float,
                   help='weight of equally averaged source PLY attribute errors in rotation or refinement')
    p.add_argument('--semantic-anchor-weight', type=float,
                   help='weight of matching the starting clean semantic attributes on finite noiseless updates')
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
    rotating = args.rotate_tiers or args.alternate_tiers
    if args.finite_noiseless_every is None:
        args.finite_noiseless_every = 5 if args.alternate_tiers else 0
    if args.semantic_anchor_weight is None:
        args.semantic_anchor_weight = .02 if args.alternate_tiers else 0.
    if args.steps_per_scene is None:
        args.steps_per_scene = 1000 if rotating or args.refine_tiers else 1500
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('joint output must be separate from the source run')
    if min(args.steps_per_scene, args.validate_every, args.save_every) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('positive step counts and finite positive learning rate required')
    if not args.rotate_tiers and args.steps_per_scene < len(LAYOUTS if args.refine_tiers else TIERS):
        raise ValueError('each scene needs at least one update per stage')
    saved = torch.load(run/'checkpoints/final.pt', map_location='cpu', weights_only=True)
    parent = saved['plan']
    response = parent.get('local_response') if rotating or args.refine_tiers else None
    if args.alternate_tiers and not response:
        raise ValueError('alternating continuation expects the completed local-response channel')
    if response:
        # Scratch stores the weights in settings; rotation records them for the following refinement.
        inherited = parent.get('render_weights', parent.get('settings'))
        if args.attribute_weight is None:
            args.attribute_weight = inherited['attribute_weight']
    elif args.attribute_weight is None:
        args.attribute_weight = 0.
    weights = (args.attribute_weight, args.semantic_anchor_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights) or args.finite_noiseless_every < 0:
        raise ValueError('rotation weights and finite noiseless interval must be nonnegative')
    if not rotating and (args.finite_noiseless_every or args.semantic_anchor_weight):
        raise ValueError('finite noiseless updates and the semantic anchor require --rotate-tiers')
    if args.attribute_weight and not (rotating or args.refine_tiers):
        raise ValueError('attribute supervision requires rotation or refinement')
    augmented_rotation = rotating and (args.finite_noiseless_every or any(weights))
    finite_steps = args.steps_per_scene // args.finite_noiseless_every if args.finite_noiseless_every else 0
    if (args.finite_noiseless_every or args.semantic_anchor_weight) and not finite_steps:
        raise ValueError('the budget must include a finite noiseless update to enable it or use a semantic anchor')
    complete = json.loads((run/'complete.json').read_text(encoding='utf8'))
    slot = parent['semantic_slot']
    from_rotation = saved['family'] in ('progressive_attribute_rotating_semantic_v1',
                                        'progressive_attribute_rotating_semantic_v2')
    if ((saved['family'] != THREE_FAMILY and not (args.refine_tiers and from_rotation)) or
            slot['arm'] != PRIVATE_ONLY or
            complete['status'] != 'complete' or saved['progress']['step'] != complete['step'] or
            complete['step'] != parent['total_updates'] or complete['rates'] != [0, 8, 24, 48] or
            saved['state']['neighbor_enabled'].item()):
        raise ValueError('expected a completed private128 three-tier source with no neighbor stage')
    semantic = SemanticAttributeCodec(slot['arm'], ModularCodecConfig.from_dict(slot['config']))
    model = LayeredAttributeCodec(semantic, LayerConfig(**parent['channel']))
    model.load_state_dict(saved['state'], strict=True)
    if model.identity() != complete['codec_id']:
        raise ValueError('source final weights differ from completed evaluation')
    if args.refine_tiers or args.alternate_tiers:
        model.layer_config = replace(model.layer_config, neighbor_mixed_only=True)
    specs = parent['selected_scenes']
    if args.manifest:
        locations = {s['name']: s for s in manifest(args.manifest)}
        specs = [dict(s, **{k: locations[s['name']][k] for k in ('source', 'ply')}) for s in specs]
    settings = parent['evaluation'] if from_rotation else parent['settings']
    evaluation = argparse.Namespace(**{k: settings[k] for k in
        ('snr', 'seed', 'trials', 'lpips', 'digital_device', 'cpu_threads', 'views_per_step')},
        digital_python=args.digital_python or settings['digital_python'], grouped_attribute_supervision=False,
        record_symbol_diagnostics=settings.get('record_symbol_diagnostics',
                                               parent.get('record_symbol_diagnostics', False)))
    scene_count = sum(s['role'] == 'train' for s in specs)
    phase_layouts = LAYOUTS if args.refine_tiers else TIERS
    per_tier, remainder = divmod(args.steps_per_scene, len(phase_layouts))
    schedule = []
    start = 0
    for tier, layout in enumerate(phase_layouts, 1):
        visits = per_tier + int(tier <= remainder)
        end = start + visits*scene_count
        schedule.append(dict(tier=tier, steps_per_scene=visits, start_step=start, end_step=end))
        if args.refine_tiers:
            name = 'mixed' if layout == 'mixed' else f'q{tier}'
            trainable = 'neighbor refiner' if layout == 'mixed' else f'channel encoder and decoder layer {tier}'
            schedule[-1].update(layout=layout, name=name, trainable=trainable)
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
    if args.refine_tiers:
        plan.update(version=REFINEMENT_FAMILY, channel=model.layer_config.to_dict(),
            trainable='active channel layer in fixed tier stages; receiver-only neighbor refiner in mixed',
            optimizer='new Adam at each stage; inherit all model weights across stages',
            mixed_layout='uniform random positive tiers per Gaussian; refine only heterogeneous blocks',
            objective='source PLY render RGB MSE + MSE(recovered latent, fixed semantic encoder latent)')
    if rotating:
        cycles, partial = divmod(args.steps_per_scene, len(LAYOUTS))
        plan.pop('schedule')
        plan.update(version='progressive_attribute_rotating_semantic_v1',
            rotation=[dict(layout=layout, steps_per_scene=cycles+int(index < partial),
                           trainable_channel_tiers=list(rotation_channel_tiers(layout)))
                      for index, layout in enumerate(LAYOUTS)],
            trainable='shared semantic decoder and the channel layers active in the current layout',
            optimizer='one Adam throughout rotation; skipped parameters retain their own moments and step counters',
            mixed_layout='uniform random positive tiers per Gaussian, independently resampled each mixed update',
            objective='source PLY render RGB MSE + MSE(recovered latent, fixed semantic encoder latent)')
        if augmented_rotation:
            finite_cycles, finite_partial = divmod(finite_steps, len(LAYOUTS))
            plan.update(version='progressive_attribute_rotating_semantic_v2',
                total_updates=(args.steps_per_scene+finite_steps)*scene_count,
                steps_per_scene=args.steps_per_scene+finite_steps,
                awgn_steps_per_scene=args.steps_per_scene, finite_noiseless_steps_per_scene=finite_steps,
                finite_noiseless_every=args.finite_noiseless_every)
            for index, row in enumerate(plan['rotation']):
                finite = finite_cycles+int(index < finite_partial)
                row.update(awgn_steps_per_scene=row['steps_per_scene'], finite_noiseless_steps_per_scene=finite)
                row['steps_per_scene'] += finite
    if response:
        plan['render_weights'] = dict(attribute_weight=args.attribute_weight, ssim_weight=inherited['ssim_weight'],
                                      latent_weight=inherited['latent_weight'])
        plan['local_response'] = {k: response[k] for k in ('weight', 'target', 'directions', 'reduction')}
        plan['local_response']['teacher_state_sha256'] = slot['state_sha256']
        plan['local_response']['phases'] = ['awgn', 'finite_noiseless'] if finite_steps else ['awgn']
        plan['objective'] = ('source PLY render RGB MSE + ssim_weight*(1-SSIM) + '
                             'latent_weight*MSE(recovered latent, fixed semantic encoder latent) + '
                             'local_response_weight*local_response_loss')
    if any(weights):
        scales = {key: saved.get('loss_scales', {}).get(key, 0.)
                  for key in ('shape', 'opacity', 'dc', 'sh', 'direction_rgb')}
        if any(not math.isfinite(scale) or scale <= 0 for scale in scales.values()):
            raise ValueError('attribute supervision requires valid loss scales from the source checkpoint')
        plan['attribute_supervision'] = dict(
            terms=list(scales), scales=scales, reduction='equal mean of the normalized terms',
            attribute_weight=args.attribute_weight, target='source PLY')
        if rotating:
            plan['attribute_supervision'].update(
                semantic_anchor_weight=args.semantic_anchor_weight,
                semantic_anchor_target='fixed starting semantic decoder at the clean semantic latent',
                semantic_anchor_path='current clean semantic decode on finite noiseless training steps')
            plan['objective'] += ' + attribute_weight*source attribute loss (both channel conditions)'
            plan['objective'] += ' + semantic_anchor_weight*clean semantic attribute loss (finite noiseless only)'
        else:
            plan['objective'] += ' + attribute_weight*source attribute loss'
    if args.alternate_tiers:
        phases = alternating_schedule(args.steps_per_scene, finite_steps, scene_count)
        plan.update(version='progressive_attribute_alternating_v1', channel=model.layer_config.to_dict(),
                    alternating_schedule=phases, total_updates=phases[-1]['end_step'],
                    steps_per_scene=phases[-1]['end_step']//scene_count,
                    awgn_steps_per_scene=2*args.steps_per_scene, joint_awgn_steps_per_scene=args.steps_per_scene,
                    refinement_awgn_steps_per_scene=args.steps_per_scene,
                    trainable='joint: semantic decoder and active tiers; q1/q2/q3: active increment; mixed finish: neighbor only',
                    optimizer='persistent separate Adam states for joint, q1, q2, q3 and mixed',
                    sampling='continue joint and sequential refinement camera, AWGN, layout and response streams by reference step')
    return model, json.loads(json.dumps(plan)), evaluation


def update(model, scene, tier, visit, cameras, settings, *, path='awgn', supervision=None, teacher=None,
           render_weights=None, response_weight=0., response_seed=None):
    model.set_scene_geometry(scene.geometry)
    if teacher is not None:
        teacher.set_scene_geometry(scene.geometry)
    layouts = layout_for(scene, tier, settings.seed+1009*visit, layers=3)
    if render_weights:
        device = next(model.parameters()).device
        objective = PerceptualImageObjective('mse_ssim', render_weights['ssim_weight'], device)
        task = PerceptualMultiViewRenderTask(cameras, scene.reference, scene.degree, scene.white_background, objective)
        centers = torch.stack([camera.camera_center for camera in scene.cameras]).to(device)
    else:
        task = MultiViewRenderTask(cameras, scene.reference, scene.degree, scene.white_background)

    def forward(features, q):
        active = q > 0
        y = model.semantic.encode(features, active)
        pred, recovered, _ = model.forward_cached(features, y, q, settings.snr,
                                                  noiseless=path == 'finite_noiseless')
        target = y[active].detach()
        mse = (recovered[active]-target).square().mean()
        aux = render_weights['latent_weight']*mse if render_weights else mse
        metrics = dict(latent=mse, target_power=target.square().mean())
        if supervision and (supervision['attribute_weight'] or teacher is not None):
            scales = supervision['scales']
            directions = directions_for(scene, features, active, cameras)
            if supervision['attribute_weight']:
                attrs, terms = attribute_objective(pred[active], features[active], scene.geometry,
                                                    model, directions, scales)
                aux = aux + supervision['attribute_weight']*attrs
                metrics.update(terms, attributes=attrs)
        if teacher is not None:
            xyz = delivered_positions(features[..., :3], q, model.cfg)
            with torch.no_grad():
                fixed = teacher.decode(y, active, xyz)
            if response_weight:
                response = clean_response_loss(model, scene, pred[active], fixed[active], centers, response_seed)
                aux = aux + response_weight*response
                metrics['local_response'] = response
            if path == 'finite_noiseless' and supervision and supervision.get('semantic_anchor_weight'):
                clean = model.semantic.decode(y, active, xyz)
                anchor, terms = attribute_objective(clean[active], fixed[active], scene.geometry,
                                                     model, directions, scales)
                aux = aux + supervision['semantic_anchor_weight']*anchor
                metrics.update({f'semantic_anchor_{key}': value for key, value in terms.items()})
                metrics['semantic_anchor'] = anchor
        return to_raw(pred[active], scene.geometry, model), aux, metrics

    loss, metrics = full_scene_step(model, list(zip(scene.groups, layouts)), scene.geometry, settings.snr,
        path, task, attr_weight=1., mode='replay', profile=True, batch_forward=forward)
    if render_weights:
        metrics.update({key: task.stats[key] for key in ('image_mse', 'image_ssim_loss', 'image_aux_weighted')})
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
        if args.alternate_tiers:
            print(json.dumps(plan['alternating_schedule'], indent=2))
            budget = f"{plan['steps_per_scene']} updates"
        elif args.rotate_tiers:
            print(json.dumps(plan['rotation'], indent=2))
            budget = f"{plan['steps_per_scene']} updates"
        elif args.refine_tiers:
            budget = ', '.join(f"{p['name']}={p['steps_per_scene']}" for p in plan['schedule'])
        else:
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
    supervision = plan.get('attribute_supervision')
    response = plan.get('local_response')
    teacher = None
    if response:
        teacher = load_semantic_slot(plan['semantic_slot']['checkpoint'])[0]
        if state_digest(teacher.state_dict()) != response['teacher_state_sha256']:
            raise ValueError('original clean semantic teacher weights do not match the channel experiment')
        teacher.to(device)
    elif supervision and supervision.get('semantic_anchor_weight'):
        teacher = deepcopy(model.semantic).requires_grad_(False).eval()
    progress = dict(step=0, cycles={})
    if not args.rotate_tiers:
        progress['tier'] = 1
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
    cycle_key = 'rotating_semantic' if args.rotate_tiers or args.alternate_tiers else ('tier_refine' if args.refine_tiers else 'tier_semantic')
    cycle_specs = [(s.name, s, cycle_key) for s in train]
    if args.finite_noiseless_every:
        cycle_specs += [(f'{s.name}:finite_noiseless', s, f'{cycle_key}_finite_noiseless') for s in train]
    cycles = {key: RefinementViewCycle([str(c.image_name) for c in s.cameras], settings.seed, s.name,
              mode, progress['cycles'].get(key)) for key, s, mode in cycle_specs}
    if args.alternate_tiers:
        for scene in train:
            for index, lane in enumerate(('q1', 'q2', 'q3', 'mixed')):
                key = f'{scene.name}:{lane}'
                state = progress['cycles'].get(key)
                cycle = RefinementViewCycle([str(c.image_name) for c in scene.cameras], settings.seed, scene.name,
                                            'tier_refine', state)
                if state is None:
                    for _ in range(index*(args.steps_per_scene//4)):
                        cycle.next(settings.views_per_step)
                cycles[key] = cycle
    perceptual = LPIPSEvaluator(device) if settings.lpips else None
    if saved:
        restore_rng(saved['rng'])

    def fixed_digest():
        if args.rotate_tiers or (args.alternate_tiers and progress.get('lane') == 'joint'):
            trainable = ('semantic.semantic_decoder.', 'encoders.', 'decoders.')
            return state_digest({k: v for k, v in model.state_dict().items() if not k.startswith(trainable)})
        return frozen_digest(model)

    def checkpoint(label, optimizer):
        mixed_refinement = (args.refine_tiers and progress['tier'] == len(LAYOUTS)) or progress.get('lane') == 'mixed'
        if bool(model.neighbor_enabled) != mixed_refinement or fixed_digest() != progress['frozen_sha256']:
            raise RuntimeError('frozen model weights or statistics changed')
        progress['cycles'] = {k: v.state_dict() for k, v in cycles.items()}
        save(out/'checkpoints'/f'{label}.pt', model, optimizer, progress, plan,
             supervision['scales'] if supervision else {})

    def evaluate(label, scenes_to_test, final=False):
        with preserved_rng(device):
            for scene in scenes_to_test:
                cached = cache_scene(model, scene)['batches']
                assess(model, scene, cached, settings, out/'evaluation'/label/scene.name,
                       scope='test' if final else 'validation',
                       layers=LAYOUTS if final or args.rotate_tiers or args.refine_tiers or args.alternate_tiers else TIERS,
                       perceptual=perceptual, actual=final, detailed_attributes=True)
                if args.alternate_tiers and not final and (label.endswith('_end') or label == 'monitor_000000'):
                    json_write(out/'evaluation'/label/scene.name/'local_response_validation.json',
                               validation_response(model, teacher, scene, cached, settings))
                del cached

    def train_step(optimizer, step, layout, *, path=None, path_visit=None, reference_step=None, **phase_fields):
        sample_step = step if reference_step is None else reference_step
        visit, index = divmod(sample_step, len(train))
        scene = train[index]
        noiseless = path == 'finite_noiseless'
        key = f'{scene.name}:{path}' if noiseless else scene.name
        lane = progress.get('lane')
        if args.alternate_tiers and lane != 'joint':
            key = f'{scene.name}:{lane}'
        cycle = cycles[key]
        cameras = [scene.cameras[i] for i in cycle.next(settings.views_per_step)]
        optimizer.zero_grad(set_to_none=True)
        if args.rotate_tiers or lane == 'joint':
            set_rotation_layout(model, layout)
        else:
            model.train()
        # Preserve the AWGN mixed seeds when inserting the noiseless visits.
        seed_visit = visit if path_visit is None else path_visit
        if noiseless:
            seed_visit += args.steps_per_scene
        layout_visit = seed_visit*len(train)+index+1 if layout == 'mixed' else seed_visit+1
        update_options = {}
        if path is not None:
            phase_fields.update(path=path, path_scene_step=path_visit+1)
            update_options.update(path=path, supervision=supervision, teacher=teacher if noiseless else None)
        elif (args.refine_tiers or args.alternate_tiers) and supervision is not None:
            update_options['supervision'] = supervision
            if args.alternate_tiers:
                phase_fields['path'] = 'awgn'
        if response:
            update_options.update(teacher=teacher, render_weights=plan['render_weights'],
                                  response_weight=response['weight'], response_seed=settings.seed+900001*(sample_step+1))
        loss, metrics = update(model, scene, layout, layout_visit, cameras, settings, **update_options)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        optimizer.step()
        progress['step'] = step+1
        append_json(log, dict(step=step+1, scene=scene.name, scene_step=step//len(train)+1, layout=layout,
                    loss=loss, grad_norm=float(norm), views=[str(c.image_name) for c in cameras],
                    **phase_fields, **metrics))
        return scene.name, loss

    interval = len(train)*args.validate_every
    if args.alternate_tiers:
        if not saved:
            progress.update(phase=0, paused_adam={}, noise_states=refinement_noise_states(
                train, args.steps_per_scene//4, device))
        phases = plan['alternating_schedule']
        for i in range(progress['phase'], len(phases)):
            phase = phases[i]
            lane, name = phase['lane'], phase['name']
            joint = lane == 'joint'
            if joint:
                model.neighbor.requires_grad_(False)
                set_rotation_layout(model, 'mixed')
                optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
            else:
                tier = 4 if lane == 'mixed' else int(lane[1:])
                optimizer = configure_stage(model, tier, args.lr)
            start, end = phase['start_step'], phase['end_step']
            if saved:
                if (progress['lane'] != lane or not start <= progress['step'] <= end
                        or fixed_digest() != progress['frozen_sha256']):
                    raise ValueError('alternating checkpoint phase or frozen state mismatch')
                optimizer.load_state_dict(saved['optimizer'])
                saved = None
            else:
                progress.update(phase=i, lane=lane)
                progress['frozen_sha256'] = fixed_digest()
                if lane in progress['paused_adam']:
                    optimizer.load_state_dict(progress['paused_adam'].pop(lane))
                state = progress['noise_states'][lane]
                if device.type == 'cuda':
                    torch.cuda.set_rng_state(state, device)
                else:
                    torch.set_rng_state(state)
                checkpoint('latest', optimizer)
            # A later joint pass changes the semantic decoder seen by frozen-stage evaluations.
            for scene in scenes:
                scene._progressive_clean_metrics = {}
            print(f"{name}: {start}..{end}, completed {progress['step']}", flush=True)
            for step in range(progress['step'], end):
                if step == 0 or (step > start and (step-start) % interval == 0):
                    evaluate(f'monitor_{step:06d}', train)
                reference = phase['reference_step']+step-start
                options = {}
                if joint:
                    path, visit = rotation_visit(reference//len(train), args.finite_noiseless_every)
                    layout = LAYOUTS[visit % len(LAYOUTS)]
                    options.update(path=path, path_visit=visit)
                else:
                    layout = 'mixed' if lane == 'mixed' else tier
                scene_name, loss = train_step(optimizer, step, layout, reference_step=reference,
                    phase=name, reference_update=reference+1, **options)
                if step == start or (step+1) % 10 == 0:
                    print(f"{name} {step+1}/{plan['total_updates']} {scene_name} {layout=} {loss=:.7g}", flush=True)
                if (step+1) % args.save_every == 0 or step+1 == end:
                    checkpoint('latest', optimizer)
            checkpoint(name+'_end', optimizer)
            evaluate(name+'_end', train)
            if i < len(phases)-1:
                progress['noise_states'][lane] = torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state()
                if any(p['lane'] == lane for p in phases[i+1:]):
                    progress['paused_adam'][lane] = optimizer.state_dict()
    if args.rotate_tiers:
        set_rotation_layout(model, 'mixed')
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
        if saved:
            if fixed_digest() != progress['frozen_sha256']:
                raise ValueError('resume checkpoint does not match the frozen encoder and statistics')
            optimizer.load_state_dict(saved['optimizer'])
            saved = None
        else:
            progress['frozen_sha256'] = fixed_digest()
            checkpoint('latest', optimizer)
        print(f"Rotating q1/q2/q3/mixed from step {progress['step']}/{plan['total_updates']}; "
              "semantic encoder frozen; mixed trains all three channel tiers.", flush=True)
        augmented = 'awgn_steps_per_scene' in plan
        if augmented:
            print(f"Per scene: {plan['awgn_steps_per_scene']} AWGN + "
                  f"{plan['finite_noiseless_steps_per_scene']} finite noiseless updates; "
                  f"attribute weight {args.attribute_weight:g}, clean semantic anchor "
                  f"weight {args.semantic_anchor_weight:g}.", flush=True)
        for step in range(progress['step'], plan['total_updates']):
            if step % interval == 0:
                evaluate(f'monitor_{step:06d}', train)
            path, visit = rotation_visit(step//len(train), args.finite_noiseless_every)
            layout = LAYOUTS[visit % len(LAYOUTS)]
            options = dict(path=path, path_visit=visit) if augmented else {}
            scene, loss = train_step(optimizer, step, layout, **options)
            if step == 0 or (step+1) % 10 == 0:
                print(f"step {step+1}/{plan['total_updates']} {scene} layout={layout} path={path} "
                      f"loss={loss:.7g}", flush=True)
            if (step+1) % args.save_every == 0 or (step+1) % interval == 0 or step+1 == plan['total_updates']:
                checkpoint('latest', optimizer)

    for phase in plan.get('schedule', ())[progress.get('tier', 1)-1:]:
        tier, start, end = phase['tier'], phase['start_step'], phase['end_step']
        layout = phase.get('layout', tier)
        stage_name = phase.get('name', f'q{tier}')
        optimizer = configure_stage(model, tier, args.lr, train_semantic=not args.refine_tiers)
        fixed = fixed_digest()
        if saved:
            if not start <= progress['step'] <= end or fixed != progress['frozen_sha256']:
                raise ValueError('resume checkpoint does not match the tier or its frozen weights')
            optimizer.load_state_dict(saved['optimizer'])
            saved = None
        else:
            progress.update(tier=tier, frozen_sha256=fixed)
            checkpoint('latest', optimizer)
        description = (f"{stage_name} refinement: train {phase['trainable']}" if args.refine_tiers else
                       f"q{tier} semantic adaptation: train semantic endpoints and channel increment {tier}")
        print(f"{description}; "
              f"global steps {start+1}..{end}, {phase['steps_per_scene']} updates per scene, "
              f"completed {progress['step']}/{plan['total_updates']}.", flush=True)
        for step in range(progress['step'], end):
            phase_step = step-start
            if phase_step and phase_step % interval == 0:
                evaluate(f'monitor_{step:06d}', train)
            scene, loss = train_step(optimizer, step, layout, tier_step=phase_step+1)
            if phase_step == 0 or (step+1) % 10 == 0:
                print(f"{stage_name} {phase_step+1}/{end-start} {scene} loss={loss:.7g}", flush=True)
            if (step+1) % args.save_every == 0 or (phase_step+1) % interval == 0 or step+1 == end:
                checkpoint('latest', optimizer)
        checkpoint(f'{stage_name}_end', optimizer)
        evaluate(f'{stage_name}_end', train)
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
