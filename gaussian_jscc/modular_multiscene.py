"""Scene-balanced four-module SSIM training, random weights unless exact resume.

No legacy codec initializer, no allocation learning. Budgets are PER SCENE;
one step means one real shared-Adam update, not a sweep or microbatch.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from .cli import device_for, seed_all
from .data import read_ply
from .modular_checkpoint import (capture_rng, restore_rng, restore_training_checkpoint,
                                 save_training_checkpoint, validate_scene)
from .modular_codec import ModularGaussianCodec
from .modular_config import ModularCodecConfig
from .modular_evaluation import evaluate_paths
from .modular_experiment import json_write, repair_logs, tier_batches
from .modular_training import (MODULES, PHASE_MODULES, bootstrap_step, color_bootstrap_step,
                               configure_phase, finish_update, objective_settings, render_step)
from .multiscene import Scene, manifest, shared_statistics
from .perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask
from .render_validation import append_json

PHASES = tuple(PHASE_MODULES)
FAMILY = 'modular_multiscene_ssim_v1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, help='existing pruned-scene manifest; no raw-PLY fallback')
    p.add_argument('--out', required=True)
    p.add_argument('--scenes', nargs='+', default=['train', 'truck', 'playroom'])
    p.add_argument('--codec-config', default=str(Path(__file__).resolve().parents[1]/'configs/modular_multiscene_ssim.json'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', help='exact continuation in original out directory; never weight-only initialization')
    for phase, steps in zip(PHASES, (5000, 1000, 2000, 2000)):
        p.add_argument('--'+phase.replace('_', '-')+'-steps', type=int, default=steps, help='updates PER SCENE')
    p.add_argument('--semantic-lr', type=float, default=1e-4)
    p.add_argument('--channel-lr', type=float, default=1e-4)
    p.add_argument('--joint-semantic-decoder-lr', type=float, default=1e-5)
    p.add_argument('--snr', type=float, default=10.)
    p.add_argument('--image-aux-weight', type=float, default=1/3)
    p.add_argument('--channel-latent-weight', type=float, default=1.)
    p.add_argument('--semantic-color-every', type=int, default=4)
    p.add_argument('--semantic-color-directions', type=int, default=8)
    p.add_argument('--blocks-per-batch', type=int, default=64)
    p.add_argument('--resolution', type=int, default=2)
    p.add_argument('--images', default='images')
    p.add_argument('--white-background', action='store_true')
    p.add_argument('--views-per-step', type=int, default=2)
    p.add_argument('--validation-views', type=int, default=4)
    p.add_argument('--validate-every', type=int, default=500, help='completed balanced rounds in each phase')
    p.add_argument('--save-every', type=int, default=500, help='completed balanced rounds in each phase')
    p.add_argument('--trials', type=int, default=2)
    p.add_argument('--net-bits-per-use', type=float, default=2., help='ASSUMED net digital bits/use for overhead conversion, not payload bits')
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true', help='validate config/path selection only; no CUDA or output creation')
    return p


def phase_lrs(args, phase):
    return dict(semantic_encoder=args.semantic_lr, semantic_decoder=(
        args.joint_semantic_decoder_lr if phase == 'channel_render' else args.semantic_lr),
        channel_encoder=args.channel_lr, channel_decoder=args.channel_lr)


def prepare_inputs(args):
    if not args.scenes or len(set(args.scenes)) != len(args.scenes):
        raise ValueError('training scene names must be unique and nonempty')
    by_name = {s['name']: s for s in manifest(args.manifest)}
    selected = []
    for name in args.scenes:
        if name not in by_name or by_name[name]['role'] != 'train':
            raise ValueError(f'{name}: select existing train-role scenes only; heldout never fits statistics')
        spec = dict(by_name[name], prior='none')
        for key in ('ply', 'source'):
            if not Path(spec[key]).exists(): raise FileNotFoundError(spec[key])
        selected.append(spec)
    cfg = ModularCodecConfig.from_dict(json.loads(Path(args.codec_config).read_text(encoding='utf-8')))
    if cfg.experimental_multirate or len(cfg.rates) != 4:
        raise ValueError('this backbone uses three positive channel prefixes')
    return selected, cfg


def validate_options(args):
    counts = {p: getattr(args, p+'_steps') for p in PHASES}
    if min(counts.values()) < 0 or not sum(counts.values()):
        raise ValueError('stage budgets must be nonnegative with at least one nonempty stage')
    if min(args.blocks_per_batch, args.views_per_step, args.validation_views, args.validate_every,
           args.save_every, args.cpu_threads, args.trials) < 1:
        raise ValueError('batch sizes, view counts, intervals, threads and trials must be positive')
    if args.semantic_color_every < 0 or args.semantic_color_directions < 2 or args.semantic_color_directions % 2:
        raise ValueError('invalid color schedule')
    for value in (args.semantic_lr, args.channel_lr, args.joint_semantic_decoder_lr,
                  args.net_bits_per_use, args.image_aux_weight, args.channel_latent_weight):
        if not math.isfinite(value) or value <= 0: raise ValueError('rates and objective weights must be finite and positive')
    if not math.isfinite(args.snr): raise ValueError('SNR must be finite')
    return counts


def make_scene(spec, raw, degree, model, args, out):
    from .rendering import load_cameras, RenderReference
    from .render_objective import split_cameras
    s = Scene(spec, raw, degree, model, args, out, with_mask=False, cameras=False)
    all_cameras = load_cameras(spec['source'], args.resolution, args.white_background, args.images, 'train')
    names = [str(c.image_name) for c in all_cameras]
    if len(set(names)) != len(names): raise ValueError(f'{s.name}: duplicate camera names')
    s.cameras, s.val_cameras, _, _ = split_cameras(all_cameras, args.validation_views)
    if len(s.cameras) < args.views_per_step: raise ValueError(f'{s.name}: insufficient training cameras')
    s.reference = RenderReference(s.raw, degree, args.white_background, 'source')
    s.white_background, s.views_per_step = args.white_background, args.views_per_step
    s.net_bits_per_use = args.net_bits_per_use
    s.record = dict(fingerprint=s.fingerprint, source_gaussians=len(raw), sh_degree=degree,
        geometry=s.geometry.to_dict(), train_view_names=[str(c.image_name) for c in s.cameras],
        validation_view_names=[str(c.image_name) for c in s.val_cameras],
        resolution=args.resolution, images=args.images, white_background=args.white_background,
        blocks_per_batch=args.blocks_per_batch, views_per_step=args.views_per_step)
    s.color_centers = torch.stack([torch.as_tensor(c.camera_center).detach().cpu() for c in s.cameras])
    return s


def settings(args, specs, cfg, counts):
    keys = ('semantic_lr','channel_lr','joint_semantic_decoder_lr','snr','image_aux_weight',
            'channel_latent_weight','semantic_color_every','semantic_color_directions','blocks_per_batch',
            'resolution','images','white_background','views_per_step','validation_views','validate_every',
            'save_every','trials','net_bits_per_use','lpips','cpu_threads','seed')
    # JSON metadata uses lists; tensor checkpoints must carry exactly the same
    # representation, not tuples that compare unequal after a round trip.
    return json.loads(json.dumps(dict(training_family=FAMILY, scene_names=[s['name'] for s in specs],
                codec_config=cfg.to_dict(), phase_budgets=counts, **{k:getattr(args,k) for k in keys})))


def main(args, max_updates=None):
    counts = validate_options(args)
    specs, cfg = prepare_inputs(args)
    protocol = settings(args, specs, cfg, counts)
    if args.dry_run:
        print(json.dumps(dict(protocol, selected_scenes=specs, random_initialization=not bool(args.resume)), indent=2))
        return protocol
    out = Path(args.out).resolve()
    if not args.resume and out.exists(): raise FileExistsError(f'use a NEW output directory: {out}')
    if args.resume and not out.exists(): raise FileNotFoundError('exact resume requires original output directory')
    if args.resume and not Path(args.resume).resolve().is_relative_to(out/'checkpoints'):
        raise ValueError('exact resume checkpoint must belong to the original output directory')
    device = device_for(args.device)
    torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    # Heldout scenes are not opened, including for moments/SH degree inference.
    loaded = [read_ply(s['ply']) for s in specs]
    if any(degree != cfg.sh_degree for _,degree in loaded):
        raise ValueError('all selected PLY SH degrees must match codec config')
    resumed_rng = saved_scenes = None
    if args.resume:
        previous = json.loads((out/'training.json').read_text(encoding='utf-8'))
        if previous.get('protocol') != protocol: raise ValueError('exact resume protocol/objective/settings mismatch')
        model, optimizer, progress, resumed_rng, saved_scenes = restore_training_checkpoint(args.resume, device)
        if progress.get('training_family') != FAMILY or model.cfg != cfg:
            raise ValueError('expected matching modular multi-scene checkpoint')
        if progress.get('protocol') != protocol: raise ValueError('checkpoint protocol mismatch')
    else:
        model = ModularGaussianCodec(cfg).to(device)
        mean, std = shared_statistics([raw for raw,_ in loaded])
        model.attr_mean.copy_(mean.to(device)); model.attr_std.copy_(std.to(device))
        optimizer = None
        progress = dict(training_family=FAMILY, protocol=protocol, step=0, phase=PHASES[0], phase_step=0,
            completed={p:0 for p in PHASES}, scene_completed={s['name']:{p:0 for p in PHASES} for s in specs},
            next_scene=0, channel_trained=False, phase_complete=False, train_snrs=[args.snr],
            module_lrs=phase_lrs(args,PHASES[0]))
    model.set_training_snrs([args.snr])
    scenes = [make_scene(spec,raw,degree,model,args,out) for spec,(raw,degree) in zip(specs,loaded)]
    del loaded
    scene_records = {s.name:s.record for s in scenes}
    if saved_scenes is not None: validate_scene(saved_scenes,scene_records)
    if not args.resume:
        for s in scenes: json_write(s.out/'scene.json',dict(s.record,**s.spec))
    (out/'metrics').mkdir(parents=True,exist_ok=True)
    objective = PerceptualImageObjective('mse_ssim', args.image_aux_weight, device)
    objectives = dict(objective_settings(args.channel_latent_weight), render_image_objective=objective.protocol())
    progress['training_objective'] = objectives
    if args.resume:
        repair_logs(out, progress['step'])
        for s in scenes: repair_logs(s.out, progress['step'])
        # Macro rows have the same global-step clock.
        from .modular_multiscene_report import repair_macro_logs
        repair_macro_logs(out, progress['step'])
        restore_rng(resumed_rng)
    else:
        json_write(out/'training.json', dict(protocol=protocol, codec_config=cfg.to_dict(),
            manifest=str(Path(args.manifest).resolve()), selected_scenes=specs,
            random_initialization=True, source_weights_loaded=False, initializer=None,
            step_budgets_scope='per_scene', step_definition='one real shared-Adam update on one scene',
            phase_budgets=counts, total_optimizer_updates=len(scenes)*sum(counts.values()),
            statistics_scenes=[s.name for s in scenes], statistics_weighting='equal scene mass',
            scene_records=scene_records, phase_module_lrs={p:phase_lrs(args,p) for p in PHASES},
            trainable_modules=PHASE_MODULES, training_objective=objectives,
            allocation_training=False, render_backward='replay', clip_mode='none',
            mixed_layout='fixed random positive tiers at evaluation; NOT learned allocation',
            model_cost='shared model excluded from wire rate; net_bits_per_use is an explicit overhead assumption'))
    print(f'Random four-module backbone; {len(scenes)} scenes; budgets PER SCENE: {counts}; '
          f'total real updates={len(scenes)*sum(counts.values())}', flush=True)

    def save(label, status='running'):
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,capture_rng(),scene_records)
        json_write(out/'progress.json',dict(progress,status=status))

    def evaluate(phase, images=False, test=False):
        from .rendering import load_cameras
        from .modular_multiscene_report import macro_metrics
        measured = {}
        for s in scenes:
            s.channel_trained = progress['channel_trained']
            original_cameras = s.val_cameras
            try:
                if test:
                    s.val_cameras = load_cameras(s.spec['source'],args.resolution,args.white_background,args.images,'test')
                    if not s.val_cameras: raise ValueError(f'{s.name}: no test views; never substitute training views')
                print(f'Evaluate scene={s.name}; global step={progress["step"]}; phase={phase}',flush=True)
                measured[s.name] = evaluate_paths(model,s,step=progress['step'],phase=phase,snrs=[args.snr],
                    trials=args.trials,seed=args.seed,save_images=images,out=s.out,
                    clean_only=not progress['channel_trained'],color_metrics=True,lpips_metrics=args.lpips)
            finally:
                s.val_cameras = original_cameras
        for row in macro_metrics(measured):
            append_json(out/'metrics'/('macro_test.jsonl' if test else 'macro_validation.jsonl'),row)

    if not args.resume:
        save('initial'); evaluate('initial',images=True)
    updated = 0
    for phase in PHASES:
        target = counts[phase]*len(scenes)
        if progress['completed'][phase] >= target:
            if target and progress['phase'] == phase and not (out/'checkpoints'/('end_'+phase)/'codec.pt').exists():
                save('end_'+phase)
            continue
        continuing = progress['phase'] == phase and optimizer is not None and not progress['phase_complete']
        lrs = phase_lrs(args,phase)
        if not continuing:
            optimizer = configure_phase(model,phase,lrs)
            progress['next_scene'] = 0
        progress.update(phase=phase,phase_step=progress['completed'][phase],module_lrs=lrs,phase_complete=False)
        model.train()
        while progress['completed'][phase] < target:
            index = progress['next_scene']; s = scenes[index]
            visit = progress['scene_completed'][s.name][phase]+1
            layout = 'clean' if phase.startswith('semantic') else ('1','2','3','mixed')[(visit-1)%4]
            qs = tier_batches(s,'3' if layout == 'clean' else layout)
            # Geometry is runtime metadata. All four network parameter sets and
            # the optimizer/moments remain shared across scene switches.
            model.set_scene_geometry(s.geometry)
            optimizer.zero_grad(set_to_none=True)
            before = {n:p.detach().clone() for n,p in model.named_parameters()}
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
            start = time.perf_counter(); views = []
            if phase.endswith('bootstrap'):
                group = random.randrange(len(s.groups))
                features,q = s.groups[group].to(device),qs[group].to(device)
                if phase == 'semantic_bootstrap' and args.semantic_color_every and visit%args.semantic_color_every == 0:
                    loss,stats = color_bootstrap_step(model,features,q,s.geometry,s.color_centers,args.semantic_color_directions)
                else:
                    loss,stats = bootstrap_step(model,features,q,s.geometry,args.snr,'awgn',phase)
            else:
                views = random.sample(range(len(s.cameras)),s.views_per_step)
                task = PerceptualMultiViewRenderTask([s.cameras[i] for i in views],s.reference,s.degree,
                                                    s.white_background,objective)
                loss,stats = render_step(model,list(zip(s.groups,qs)),s.geometry,args.snr,'awgn',task,phase,
                                         channel_latent_weight=args.channel_latent_weight)
                stats.update(task.stats)
            stats.update(finish_update(model,optimizer,before))
            if device.type == 'cuda': torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-start
            progress['step'] += 1; progress['completed'][phase] += 1
            progress['phase_step'] = progress['completed'][phase]
            progress['scene_completed'][s.name][phase] = visit
            progress['next_scene'] = (index+1)%len(scenes)
            progress['phase_complete'] = progress['completed'][phase] == target
            if phase.startswith('channel'): progress['channel_trained'] = True
            row = dict(stats,step=progress['step'],phase=phase,phase_step=progress['phase_step'],
                scene=s.name,scene_step=visit,layout=layout,loss=float(loss),
                actual_snr=args.snr if phase.startswith('channel') else None,
                module_lrs={g['name']:g['lr'] for g in optimizer.param_groups},
                frozen_parameters_unchanged=True,step_seconds=elapsed,training_view_indices=views,
                optimizer_reset_at_scene_switch=False,
                peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
            append_json(out/'metrics/loss.jsonl',row)
            if visit == 1 or visit%10 == 0 or visit == counts[phase]:
                print(f'{s.name} {phase} {visit}/{counts[phase]} (global {progress["step"]}): '
                      f'loss={float(loss):.6g}, q={layout}, grad={stats["grad_norm"]:.4g}, sec={elapsed:.2f}',flush=True)
            balanced = progress['next_scene'] == 0
            # Save first: interruption during slow validation loses no trained round.
            if balanced and (visit%args.save_every == 0 or progress['phase_complete']): save('latest')
            if balanced and visit%args.validate_every == 0: evaluate(phase)
            updated += 1
            if max_updates is not None and updated >= max_updates:
                save('latest',status='interrupted'); return progress
        if counts[phase]%args.validate_every: evaluate(phase+'_end')
        save('end_'+phase); save('latest')
    evaluate('final',images=True); evaluate('test',test=True)
    progress['status'] = 'complete'
    save('final',status='complete'); save('latest',status='complete')
    from .modular_multiscene_report import plot_run
    plot_run(out)
    json_write(out/'complete.json',dict(progress,status='complete',cuda_render_verified=device.type=='cuda'))
    return progress


if __name__ == '__main__': main(parser().parse_args())
