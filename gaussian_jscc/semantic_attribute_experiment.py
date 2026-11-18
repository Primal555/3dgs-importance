"""Four random-initialized semantic-only controls, rotating views and exact resume."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time

import torch
from torch.nn.utils.rnn import pad_sequence

from .cli import device_for, seed_all
from .data import read_ply, to_raw
from .modular_checkpoint import capture_rng, restore_rng, validate_scene
from .modular_config import ModularCodecConfig
from .modular_experiment import json_write
from .multiscene import manifest, shared_statistics
from .optimization import preserved_rng
from .perceptual_metrics import LPIPSEvaluator, image_metric_protocol
from .perceptual_objective import PerceptualImageObjective, PerceptualMultiViewRenderTask
from .refinement_views import RefinementViewCycle, refinement_split, make_refinement_scene
from .render_validation import append_json
from .semantic_attribute_codec import ARMS, TRAINABLE_ARMS, SemanticAttributeCodec, make_config, fit_loss_scales, attribute_objective
from .semantic_attribute_metrics import evaluate, latent_diagnostics, directions_for, write_plots, diagnostic_plots
from .successive_multiscene_inputs import make_scene
from .training import full_scene_step

FAMILY = 'semantic_attribute_controls_v1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='existing shared codec run: read scene metadata only')
    p.add_argument('--out', required=True)
    p.add_argument('--manifest', help='explicit replacement scene paths, preserving source identity')
    p.add_argument('--heldout-manifest', default='configs/multiscene_tandt_db.json')
    p.add_argument('--scenes', nargs='+', default=['truck', 'playroom'])
    p.add_argument('--heldout-scene', default='drjohnson')
    p.add_argument('--arms', nargs='+', choices=TRAINABLE_ARMS, default=list(ARMS))
    p.add_argument('--bootstrap-steps', type=int, default=3000, help='TOTAL updates per arm across training scenes')
    p.add_argument('--render-steps', type=int, default=1000, help='TOTAL updates per arm across training scenes')
    p.add_argument('--bootstrap-lr', type=float, default=1e-4)
    p.add_argument('--render-lr', type=float, default=1e-5)
    p.add_argument('--attribute-weight', type=float, default=.01)
    p.add_argument('--ssim-weight', type=float, default=1/3)
    p.add_argument('--views-per-step', type=int, default=32)
    p.add_argument('--validation-views', type=int, default=32)
    p.add_argument('--test-views', type=int, default=32)
    p.add_argument('--blocks-per-batch', type=int, default=16)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--resolution', type=int)
    p.add_argument('--images')
    p.add_argument('--white-background', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def prepare(args):
    source_path = Path(args.run).resolve()/'training.json'
    source = read(source_path)
    if Path(args.out).resolve().is_relative_to(Path(args.run).resolve()):
        raise ValueError('use an output outside the source run')
    if not args.scenes or len(set(args.scenes)) != len(args.scenes) or len(set(args.arms)) != len(args.arms):
        raise ValueError('unique scene/arm names required')
    if args.heldout_scene in args.scenes:
        raise ValueError('heldout scene overlaps training')
    if min(args.bootstrap_steps, args.render_steps) < 1 or any(
            n % len(args.scenes) for n in (args.bootstrap_steps, args.render_steps)):
        raise ValueError('positive stage budgets must be divisible by training scene count')
    if min(args.views_per_step, args.validation_views, args.test_views, args.blocks_per_batch,
           args.validate_every, args.save_every, args.cpu_threads) < 1:
        raise ValueError('positive view/batch/interval counts required')
    for k in ('bootstrap_lr', 'render_lr', 'attribute_weight', 'ssim_weight'):
        if not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0:
            raise ValueError('positive finite '+k+' required')
    cfg_source = source.get('codec_config', {})
    settings = source.get('settings', source.get('scene', source))
    for key, default in [('resolution', 2), ('images', 'images'), ('white_background', False)]:
        if getattr(args, key) is None:
            setattr(args, key, settings.get(key, default))
    # Scene's legacy meter is never used/reported in this continuous reconstruction study.
    args.net_bits_per_use = 2.
    lookup = {s['name']: s for s in source['selected_scenes']}
    override = {s['name']: s for s in manifest(args.manifest)} if args.manifest else lookup
    specs = []
    for name in args.scenes:
        if name not in lookup or name not in override or override[name]['role'] != 'train':
            raise ValueError('source training scene missing: '+name)
        specs.append(dict(override[name], prior='none'))
    held_lookup = {s['name']: s for s in manifest(args.heldout_manifest)}
    held = dict(held_lookup[args.heldout_scene], prior='none')
    if held['role'] != 'heldout':
        raise ValueError('heldout-role scene required')
    for spec in specs+[held]:
        for key in ('ply', 'source'):
            if not Path(spec[key]).exists():
                raise FileNotFoundError(f'{spec["name"]}: {spec[key]}')
    for key in ('ply', 'source'):
        if len({str(Path(s[key]).resolve()).casefold() for s in specs+[held]}) != len(specs)+1:
            raise ValueError('training/heldout paths overlap')
    records = source.get('scene_records', {})
    if not records and 'scene' in source and len(lookup) == 1:
        records = {next(iter(lookup)): source['scene']}
    splits = {s['name']: refinement_split(records[s['name']], args.validation_views, args.views_per_step)
              for s in specs}
    # Preserve recorded test cameras exactly, including scenes with fewer than
    # the requested maximum (Playroom has 29). Never repeat cameras to make 32.
    if any(not records[s['name']]['test_view_names'] or
           len(records[s['name']]['test_view_names']) > args.test_views for s in specs):
        raise ValueError('--test-views must cover all recorded source test cameras')
    cfg = make_config(128, args.hidden, cfg_source.get('block_size', 256))
    for key in ('context_centers', 'context_neighbors', 'context_radii', 'morton_bits'):
        if key in cfg_source:
            setattr(cfg, key, cfg_source[key])
    cfg.__post_init__()
    options = {k: v for k, v in vars(args).items()
               if k not in ('resume', 'dry_run', 'out', 'device', 'net_bits_per_use')}
    plan = dict(version=FAMILY, settings=options, selected_scenes=specs, heldout_scene=held,
        source_metadata_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        config_template=cfg.to_dict(), camera_splits=splits,
        budgets=dict(bootstrap_total=args.bootstrap_steps, render_total=args.render_steps,
            updates_per_arm=args.bootstrap_steps+args.render_steps,
            bootstrap_per_scene=args.bootstrap_steps//len(specs), render_per_scene=args.render_steps//len(specs)),
        initialization='random weights; no source weights loaded',
        latent='continuous real values, no quantization/power normalization/channel',
        source_reference='fixed source PLY; original XYZ; no source attribute optimization',
        view_protocol='distinct shuffled cycles per scene/phase; mean over 32 views by default; CPU target cache',
        selection='fixed bootstrap/final endpoints; validation reports only; test only at endpoints',
        objective=dict(attribute='equal mean of five independently scaled attribute errors',
            shape='matrix log covariance MSE, invariant to quaternion sign',
            render='source RGB MSE + ssim_weight*(1-SSIM) + attribute_weight*attribute_loss',
            statistics='equal scene mass on training PLYs only; fixed across arms'),
        communication_cost=None, cost_scope='semantic-only experiment; no channel use or digital delivery measured',
        metric_protocol=image_metric_protocol(args.lpips),
        capacity_control='same hidden width/context config; parameter counts reported, not parameter matched')
    return cfg, specs, held, records, json.loads(json.dumps(plan))


def save_checkpoint(path, model, optimizer, progress, records, plan):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(dict(family=FAMILY, arm=model.arm, config=model.cfg.to_dict(),
        model=model.state_dict(), optimizer=optimizer.state_dict(), progress=progress,
        rng=capture_rng(), scene_records=records, plan=plan), temporary)
    temporary.replace(path)


def bootstrap_batch(scene, count, visit, seed):
    # Counter-based balanced shuffled block cycles, independent of model/evaluation RNG.
    n = len(scene.blocks)
    selected = []
    for offset in range(count):
        epoch, index = divmod((visit-1)*count+offset, n)
        g = torch.Generator().manual_seed(seed+epoch*100003)
        # A batch generally spans only one permutation; cache it within this call.
        if not selected or epoch != last_epoch:
            permutation = torch.randperm(n, generator=g).tolist()
            last_epoch = epoch
        selected.append(permutation[index])
    f = pad_sequence([scene.blocks[i] for i in selected], batch_first=True)
    active = torch.arange(f.shape[1])[None] < torch.tensor([len(scene.blocks[i]) for i in selected])[:, None]
    return f, active


def render_update(model, scene, cameras, scales, args):
    model.train(); model.set_scene_geometry(scene.geometry)
    objective = PerceptualImageObjective('mse_ssim', args.ssim_weight, next(model.parameters()).device)
    task = PerceptualMultiViewRenderTask(cameras, scene.reference, scene.degree, scene.white_background, objective)
    centers = torch.stack([c.camera_center.cpu() for c in cameras])

    def forward(f, q):
        from .local_response import camera_directions
        active = q > 0
        pred = model(f, active)
        dirs = camera_directions(scene.geometry.denormalize(f[active][:, :3]), centers.to(f))
        loss, terms = attribute_objective(pred[active], f[active], scene.geometry, model, dirs, scales)
        return to_raw(pred[active], scene.geometry, model), loss, terms

    batches = [(f, (ids >= 0).long()) for f, ids in zip(scene.groups, scene.group_ids)]
    loss, stats = full_scene_step(model, batches, scene.geometry, 0., 'none', task,
        attr_weight=args.attribute_weight, mode='replay', profile=True, batch_forward=forward)
    return loss, dict(stats, **task.stats)


def review(out):
    out = Path(out)
    rows = []
    for path in sorted(out.glob('*/evaluation/*.json')):
        record = read(path)
        semantic = next(r for r in record['rows'] if r['variant'] == 'semantic')
        rows.append({k: v for k, v in dict(record, **semantic).items()
                     if k not in ('rows', 'views', 'attributes')})
    json_write(out/'review_summary.json', dict(version=FAMILY, rows=rows,
        completed_arms=[p.parent.name for p in out.glob('*/complete.json')],
        architectures={p.parent.name: read(p) for p in out.glob('*/architecture.json')},
        comparison='match scene/scope/endpoint; latent dimensions are not channel costs'))
    write_plots(out/'plots', rows)
    temporary = out/'review.tmp.tar.gz'
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.png', '.md'):
                archive.add(path, arcname=str(path.relative_to(out)))
    temporary.replace(out/'review.tar.gz')


def main(args, max_updates=None):
    cfg, specs, held, source_records, plan = prepare(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return plan
    out = Path(args.out).resolve()
    if out.exists() and not args.resume:
        raise FileExistsError('use a new output or --resume')
    if args.resume and (not (out/'plan.json').exists() or read(out/'plan.json') != plan):
        raise ValueError('resume requires identical experiment settings')
    out.mkdir(parents=True, exist_ok=True); json_write(out/'plan.json', plan)
    device = device_for(args.device)
    torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    raws = [read_ply(s['ply']) for s in specs]
    if any(degree != 3 for _, degree in raws):
        raise ValueError('SH degree 3 required')
    mean, std = shared_statistics([raw for raw, _ in raws])
    scales = fit_loss_scales([raw for raw, _ in raws])
    json_write(out/'attribute_statistics.json', dict(mean=mean.tolist(), std=std.tolist(), loss_scales=scales))
    template = SemanticAttributeCodec('shared128', cfg).to(device)
    template.attr_mean.copy_(mean.to(device)); template.attr_std.copy_(std.to(device))
    scenes = [make_refinement_scene(spec, raw, degree, template, args, out,
        dict(source_records[spec['name']], blocks_per_batch=args.blocks_per_batch),
        plan['camera_splits'][spec['name']])
        for spec, (raw, degree) in zip(specs, raws)]
    del raws
    raw, degree = read_ply(held['ply'])
    held_scene = make_scene(held, raw, degree, template, args, out)
    del raw, template
    records = {s.name: s.record for s in scenes+[held_scene]}
    if len({s.fingerprint for s in scenes+[held_scene]}) != len(scenes)+1:
        raise ValueError('scene PLY content aliases')
    json_write(out/'scene_records.json', records)
    # LPIPS loads once and never participates in training gradients.
    perceptual = LPIPSEvaluator(device) if args.lpips else None
    executed = 0
    for arm in args.arms:
        folder = out/arm
        if (folder/'complete.json').exists():
            continue
        (folder/'metrics').mkdir(parents=True, exist_ok=True)
        seed_all(args.seed)
        arm_cfg = ModularCodecConfig.from_dict(dict(cfg.to_dict(), latent_dim=int(arm[-3:])))
        model = SemanticAttributeCodec(arm, arm_cfg).to(device)
        model.attr_mean.copy_(mean.to(device)); model.attr_std.copy_(std.to(device))
        progress = dict(step=0, phase='bootstrap', cycles={}, evaluated=[])
        saved = None
        if (folder/'checkpoints/latest.pt').exists():
            if not args.resume:
                raise ValueError('checkpoint exists without --resume')
            saved = torch.load(folder/'checkpoints/latest.pt', map_location='cpu', weights_only=True)
            if saved['family'] != FAMILY or saved['arm'] != arm or saved['plan'] != plan:
                raise ValueError('checkpoint identity mismatch')
            validate_scene(saved['scene_records'], records)
            model.load_state_dict(saved['model']); progress = saved['progress']
            if not torch.equal(model.attr_mean.cpu(), mean) or not torch.equal(model.attr_std.cpu(), std):
                raise ValueError('training attribute statistics changed')
        optimizer = torch.optim.Adam(model.parameters(), lr=(args.bootstrap_lr if progress['phase'] == 'bootstrap' else args.render_lr))
        if saved is not None:
            optimizer.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
            path = folder/'metrics/loss.jsonl'
            if path.exists():
                lines = [line for line in path.read_text().splitlines() if json.loads(line)['step'] <= progress['step']]
                path.write_text(''.join(line+'\n' for line in lines), encoding='utf8')
        json_write(folder/'architecture.json', dict(arm=arm, config=model.cfg.to_dict(),
            latent_slices=model.slices, parameters=sum(p.numel() for p in model.parameters()),
            modules={k: sum(p.numel() for p in m.parameters()) for k, m in model.named_children()}))
        cycles = {}
        for phase in ('bootstrap', 'render'):
            for scene in scenes:
                key = phase+'/'+scene.name
                cycles[key] = RefinementViewCycle([str(c.image_name) for c in scene.cameras],
                    args.seed, scene.name, 1 if phase == 'bootstrap' else 2, progress['cycles'].get(key))

        def save(label='latest'):
            progress['cycles'] = {k: c.state_dict() for k, c in cycles.items()}
            save_checkpoint(folder/'checkpoints'/f'{label}.pt', model, optimizer, progress, records, plan)

        def assessment(endpoint, final=False):
            key = str(progress['step'])+'/'+endpoint
            if key in progress['evaluated']:
                return
            with preserved_rng(device):
                for scope, candidates in [('validation', scenes), ('test', scenes+[held_scene])]:
                    if scope == 'test' and endpoint not in ('bootstrap_end', 'final'):
                        continue
                    for scene in candidates:
                        cameras = scene.val_cameras if scope == 'validation' else scene.test_cameras
                        dest = folder/'evaluation'/f'{progress["step"]:06d}_{scope}_{scene.name}'
                        record = evaluate(model, scene, cameras, perceptual, dest,
                                          interventions=final and scope == 'test',
                                          images=scope == 'test')
                        record.update(arm=arm, step=progress['step'], endpoint=endpoint, scope=scope)
                        json_write(dest.with_suffix('.json'), record)
                        r = record['rows'][0]
                        print(f'{arm} {scope} {scene.name} step={progress["step"]}: '
                              f'source PSNR={r["source_psnr"]:.4f}, photo PSNR={r["photo_psnr"]:.4f}, '
                              f'SSIM={r["photo_ssim"]:.5f}', flush=True)
                        if final and scope == 'test':
                            diagnostics = latent_diagnostics(model, scene, scales, args.seed)
                            json_write(folder/'diagnostics'/f'{scene.name}.json', diagnostics)
                            diagnostic_plots(folder/'diagnostics'/scene.name, diagnostics)
            progress['evaluated'].append(key); save()

        total = args.bootstrap_steps+args.render_steps
        # If interrupted during endpoint evaluation, replay only that evaluation.
        if progress['step'] == args.bootstrap_steps:
            assessment('bootstrap_end')
        for step in range(progress['step']+1, total+1):
            phase = 'bootstrap' if step <= args.bootstrap_steps else 'render'
            if phase != progress['phase']:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.render_lr)
                progress['phase'] = phase
            phase_step = step if phase == 'bootstrap' else step-args.bootstrap_steps
            index = (phase_step-1) % len(scenes); scene = scenes[index]
            visit = (phase_step-1)//len(scenes)+1
            cycle = cycles[phase+'/'+scene.name]
            camera_ids = cycle.next(args.views_per_step)
            cameras = [scene.cameras[i] for i in camera_ids]
            model.train(); model.set_scene_geometry(scene.geometry)
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            if phase == 'bootstrap':
                features, active = bootstrap_batch(scene, args.blocks_per_batch, visit, args.seed+index*1009)
                features, active = features.to(device), active.to(device)
                pred = model(features, active)
                dirs = directions_for(scene, features, active, cameras)
                loss, terms = attribute_objective(pred[active], features[active], scene.geometry, model, dirs, scales)
                loss.backward()
                stats = {k+'_loss': float(v.detach()) for k, v in terms.items()}
            else:
                loss, stats = render_update(model, scene, cameras, scales, args)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite semantic loss')
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            progress['step'] = step
            record = dict(step=step, phase=phase, scene=scene.name, scene_visit=visit,
                loss=float(loss.detach()), grad_norm=float(norm), seconds=time.perf_counter()-started,
                cameras=[str(c.image_name) for c in cameras], view_coverage=cycle.coverage(),
                lr=optimizer.param_groups[0]['lr'], **stats)
            if device.type == 'cuda':
                record['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/2**20
            append_json(folder/'metrics/loss.jsonl', record)
            if step == 1 or step % 10 == 0:
                print(f'{arm} {phase} {phase_step}/{args.bootstrap_steps if phase == "bootstrap" else args.render_steps}: '
                      f'{scene.name}, loss={float(loss.detach()):.7g}, views={len(cameras)}, '
                      f'coverage={cycle.coverage()["covered"]}/{len(scene.cameras)}, sec={record["seconds"]:.2f}', flush=True)
            if step % args.save_every == 0 or step in (args.bootstrap_steps, total):
                save()
            if step == args.bootstrap_steps:
                save('bootstrap_end'); assessment('bootstrap_end')
            elif step != total and step % args.validate_every == 0:
                assessment('monitor')
            executed += 1
            if max_updates is not None and executed >= max_updates:
                save(); return dict(status='interrupted', arm=arm, step=step)
        save('final'); assessment('final', final=True)
        json_write(folder/'complete.json', dict(status='complete', arm=arm, step=total,
            selection='fixed_final', initialized_from_source_weights=False,
            training_scenes=args.scenes, heldout_scene=args.heldout_scene,
            view_coverage={k: c.coverage() for k, c in cycles.items()}))
        del model, optimizer
        review(out)
    json_write(out/'complete.json', dict(status='complete', arms=args.arms,
        updates_per_arm=args.bootstrap_steps+args.render_steps))
    review(out)


if __name__ == '__main__':
    main(parser().parse_args())
