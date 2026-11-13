"""Continue a shared successive codec across SNR, with a frozen semantic encoder."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import tarfile
import time
import statistics

import torch

from .channel_normalization_metrics import cache_teacher
from .cli import device_for, seed_all
from .data import read_ply
from .model_factory import build_codec
from .modular_checkpoint import (capture_rng, restore_rng, save_training_checkpoint,
                                 restore_training_checkpoint, validate_scene)
from .modular_experiment import json_write, repair_logs
from .modular_training import finish_update
from .multisnr_evaluation import evaluate_matrix, actual_delivery, read
from .multisnr_training import schedule, tiers, optimizer_for, cached_render_step
from .optimization import preserved_rng
from .perceptual_metrics import LPIPSEvaluator
from .render_objective import MultiViewRenderTask
from .render_validation import append_json
from .successive_joint import encoder_digest
from .successive_multiscene_inputs import make_scene
from .successive_training import loss_for_batch
from .successive_transfer import load_source
from .transport import model_id

FAMILY = 'successive_multisnr_v1'
PHASES = ('channel_bootstrap', 'channel_render')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--out', required=True)
    p.add_argument('--scenes', nargs='+', help='default: all training scenes in the completed shared run')
    p.add_argument('--heldout-scenes', nargs='*', default=['room'])
    p.add_argument('--snr-range', nargs=2, type=float, default=[5.,15.])
    p.add_argument('--eval-snrs', nargs='+', type=float, default=[5.,7.5,10.,12.5,15.])
    for name, default in [('snr-bins',5), ('channel-steps-per-scene',1000), ('joint-steps-per-scene',250),
                          ('save-every',50), ('trials',4), ('cpu-threads',4), ('seed',42),
                          ('response-directions',8), ('color-every',4), ('color-stat-start',250),
                          ('color-start',500), ('noiseless-every',4)]:
        p.add_argument('--'+name, type=int, default=default)
    for name, default in [('channel-lr',1e-4), ('joint-channel-lr',1e-5), ('decoder-lr',1e-5),
                          ('latent-weight',1.), ('response-weight',.1), ('noiseless-weight',.25),
                          ('color-relative-gradient',.25)]:
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--digital-python', required=True, help='gs_digital interpreter for final actual side-file tests')
    p.add_argument('--digital-device', default='cuda')
    p.add_argument('--delivery-scenes', nargs='+', default=['truck'])
    p.add_argument('--delivery-layouts', nargs='+', choices=('1','2','3','mixed'), default=['2'])
    p.add_argument('--resume', action='store_true'); p.add_argument('--dry-run', action='store_true')
    return p


def subset_digest(model, prefix):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(prefix): h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def inputs(args):
    source, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out): raise ValueError('source/output must be separate')
    saved, metadata, cfg, identity = load_source(source)
    lookup = {s['name']: s for s in metadata['selected_scenes']}
    names = args.scenes or list(lookup)
    if len(set(names)) != len(names) or not set(names) <= set(lookup):
        raise ValueError('select unique source training scenes')
    specs = [dict(lookup[name], prior='none') for name in names]
    held = metadata.get('heldout_scene', [])
    if isinstance(held, dict): held = [held]
    held_lookup = {s['name']: s for s in held}
    if len(set(args.heldout_scenes)) != len(args.heldout_scenes) or not set(args.heldout_scenes) <= set(held_lookup):
        raise ValueError('heldout scenes must belong to the original source evaluation set; use --heldout-scenes for none')
    heldouts = [dict(held_lookup[n], prior='none') for n in args.heldout_scenes]
    if set(args.heldout_scenes) & set(identity['training_scene_history']): raise ValueError('heldout used in codec training')
    for spec in specs+heldouts:
        for key in ('ply','source'):
            if not Path(spec[key]).exists(): raise FileNotFoundError(f'{spec["name"]}: {spec[key]}')
    if not set(args.delivery_scenes) <= {s['name'] for s in specs+heldouts}:
        raise ValueError('delivery scene must be part of this experiment')
    if len(set(args.delivery_scenes)) != len(args.delivery_scenes) or len(set(args.delivery_layouts)) != len(args.delivery_layouts):
        raise ValueError('duplicate delivery scenes/layouts')
    if not Path(args.digital_python).is_file(): raise FileNotFoundError(args.digital_python)
    low, high = args.snr_range
    schedule(1, args.seed, low, high, args.snr_bins)
    if low > 10 or high < 10: raise ValueError('continuation range must include source 10 dB')
    if (len(set(args.eval_snrs)) != len(args.eval_snrs) or 10. not in args.eval_snrs or
            any(not math.isfinite(x) or not low <= x <= high for x in args.eval_snrs)):
        raise ValueError('unique finite in-range evaluation SNRs including 10 dB required')
    args.eval_snrs = sorted(args.eval_snrs)
    integer_keys = ('channel_steps_per_scene','joint_steps_per_scene','save_every','trials',
                    'cpu_threads','response_directions','color_every','noiseless_every')
    if min(getattr(args,k) for k in integer_keys) < 1: raise ValueError('positive budgets required')
    if args.response_directions%2 or not 0 <= args.color_stat_start <= args.color_start:
        raise ValueError('even color directions and ordered nonnegative color warmup required')
    if any(not math.isfinite(getattr(args,k)) or getattr(args,k)<=0 for k in
           ('channel_lr','joint_channel_lr','decoder_lr','latent_weight','response_weight',
            'noiseless_weight','color_relative_gradient')): raise ValueError('positive finite weights/LRs required')
    # Scene preparation retains the source split and execution shape exactly.
    old = metadata['settings']
    for key in ('blocks_per_batch','views_per_step','resolution','images','white_background'):
        setattr(args, key, old[key])
    args.validation_views = old['validation_views']; args.test_views = old['test_views']
    args.net_bits_per_use = 2.  # Scene legacy constructor only; every reported cost uses digital_side_file_v1.
    settings = {k:v for k,v in vars(args).items() if k not in ('run','out','resume','dry_run','net_bits_per_use')}
    plan = dict(version=FAMILY, **identity, source_run=str(source), settings=settings,
        selected_scenes=specs, heldout_scene=heldouts, codec_config=cfg.to_dict(),
        total_updates=len(specs)*(args.channel_steps_per_scene+args.joint_steps_per_scene),
        phase_updates={p: len(specs)*n for p,n in zip(PHASES,[args.channel_steps_per_scene,args.joint_steps_per_scene])},
        snr_sampling='uniform dB, shuffled SNR-stratum x layout cycles per scene',
        cost_protocol='digital_side_file_v1', selection='fixed final update',
        semantic_encoder_frozen=True, attribute_statistics_frozen=True, allocation_training=False,
        actual_delivery_scope=dict(scenes=args.delivery_scenes, layouts=args.delivery_layouts, snrs=args.eval_snrs, trials=args.trials))
    return saved, metadata, cfg, specs, heldouts, json.loads(json.dumps(plan))


def write_review(out):
    out = Path(out)
    # Checkpoints, cached features and temporary analogue payloads stay off the review bundle.
    with tarfile.open(out/'review.tar.gz', 'w:gz') as archive:
        for root in ('evaluation','actual_delivery','metrics'):
            for path in sorted((out/root).rglob('*')):
                if path.is_file() and path.suffix in ('.json','.jsonl','.csv','.png','.log'):
                    archive.add(path, arcname=path.relative_to(out))
        for name in ('plan.json','training.json','complete.json','review_summary.json','progress.json'):
            if (out/name).exists(): archive.add(out/name, arcname=name)


def summarize(out, plan):
    rows = []
    sources = {}
    for label in ('initial','channel_end','final'):
        for path in sorted((Path(out)/'evaluation'/label).glob('*/summary.json')):
            source = read(path)['rows']
            sources[f'{label}/{path.parent.name}'] = read(path.parent/'source_photo.json')['mean']
            keys = sorted({(r['path'],r['snr_db'],r['layout']) for r in source})
            for kind, snr, layout in keys:
                selected = [r for r in source if (r['path'],r['snr_db'],r['layout']) == (kind,snr,layout)]
                row = dict(endpoint=label,scene=selected[0]['scene'],scope=selected[0]['scope'],
                    path=kind,snr_db=snr,condition_snr_db=selected[0]['condition_snr_db'],layout=layout,
                    trials=len(selected),total_channel_uses=selected[0]['total_channel_uses'])
                for key in ('source_psnr','photo_psnr','source_ssim','photo_ssim','source_lpips','photo_lpips'):
                    if key in selected[0]:
                        values = [r[key] for r in selected]
                        row[key] = statistics.mean(values)
                        row[key+'_std_trials'] = statistics.pstdev(values)
                for key in ('latent_mse','latent_nmse','latent_target_power'):
                    row[key] = statistics.mean(r['diagnostics'][key] for r in selected)
                rows.append(row)
    initial = {(r['scene'],r['scope'],r['path'],r['snr_db'],r['layout']):r for r in rows if r['endpoint']=='initial'}
    for row in rows:
        if row['endpoint'] != 'final': continue
        base = initial[(row['scene'],row['scope'],row['path'],row['snr_db'],row['layout'])]
        row['change_from_initial'] = {k:row[k]-base[k] for k in
            ('source_psnr','photo_psnr','source_ssim','photo_ssim','latent_mse','latent_nmse')}
    report = dict(status='complete', matrix=rows, snr_range=plan['settings']['snr_range'],
        source_ply_vs_photos=sources,
        cost_protocol='digital_side_file_v1', quality_conditioning='reliable-side assumption; actual link results reported separately',
        actual_delivery={p.parent.name:read(p)['results'] for p in (Path(out)/'actual_delivery').glob('*/summary.json')})
    json_write(Path(out)/'review_summary.json', report)
    return report


def main(args, max_updates=None):
    saved, metadata, cfg, specs, heldouts, plan = inputs(args)
    if args.dry_run: print(json.dumps(plan, indent=2)); return plan
    out = Path(args.out).resolve(); latest = out/'checkpoints/latest/codec.pt'
    if out.exists() and not args.resume: raise FileExistsError('use new OUT or --resume')
    if args.resume and (not latest.exists() or read(out/'plan.json') != plan):
        raise ValueError('resume checkpoint missing or plan/source changed')
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    if args.resume:
        model, optimizer, progress, rng, old_records = restore_training_checkpoint(latest, device)
    else:
        model = build_codec(cfg).to(device); model.load_state_dict(saved['state_dict'])
        optimizer, lrs = optimizer_for(model, PHASES[0], args)
        progress = dict(training_family=FAMILY, step=0, phase=PHASES[0], phase_step=0,
            successive_joint=True, freeze_semantic=True, train_semantic_encoder=False,
            channel_trained=True, module_lrs=lrs, train_snrs=args.snr_range,
            phase_budgets=plan['phase_updates'], completed={p:0 for p in PHASES},
            scene_completed={p:{s['name']:0 for s in specs} for p in PHASES},
            initial_done=False, channel_end_done=False, final_done=False, next_scene=0,
            successive_phase='cooperative', color_state={}, phase_complete=False,
            encoder_sha256=encoder_digest(model.state_dict()), decoder_sha256=subset_digest(model,'semantic_decoder.'),
            training_objective=dict(version=FAMILY))
        rng = capture_rng(); old_records = None
    del saved
    model.set_training_snrs(args.snr_range)
    scenes = []
    for spec in specs:
        raw, degree = read_ply(spec['ply'])
        scenes.append(make_scene(spec,raw,degree,model,args,out,old=metadata['scene_records'][spec['name']]))
    records = {s.name:s.record for s in scenes}
    if len({s.fingerprint for s in scenes}) != len(scenes): raise ValueError('training scene aliases')
    if old_records is not None: validate_scene(old_records, records)
    else:
        json_write(out/'plan.json', plan)
        json_write(out/'training.json', dict(plan,scene_records=records,
            trainable_modules={PHASES[0]:['channel_encoder','channel_decoder'],
                               PHASES[1]:['channel_encoder','channel_decoder','semantic_decoder']},
            optimizer_reset='new Adam at entry and phase transition; shared across scenes'))
    for s in scenes: json_write(s.out/'scene.json',dict(s.record,**s.spec))
    if args.resume: repair_logs(out, progress['step'])
    (out/'metrics').mkdir(parents=True, exist_ok=True)
    restore_rng(rng)
    perceptual = None
    cache = {}; packed_cache = {}

    def save(label='latest'):
        if encoder_digest(model.state_dict()) != progress['encoder_sha256']:
            raise RuntimeError('frozen semantic encoder/statistics changed')
        if progress['phase'] == PHASES[0] and subset_digest(model,'semantic_decoder.') != progress['decoder_sha256']:
            raise RuntimeError('semantic decoder changed during channel-only training')
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,capture_rng(),records)
        json_write(out/'progress.json',dict(progress,status='running'))

    def evaluate(label):
        nonlocal perceptual
        if args.lpips and perceptual is None:
            with preserved_rng(device): perceptual = LPIPSEvaluator(device)
        scope = 'validation' if label == 'channel_end' else 'test'
        with preserved_rng(device):
            for scene in scenes:
                evaluate_matrix(model,scene,args,out,label,scope=scope,fixed_condition=label=='initial',perceptual=perceptual)
            if scope == 'test':
                for spec in heldouts:
                    raw, degree = read_ply(spec['ply'])
                    s = make_scene(spec,raw,degree,model,args,out)
                    if s.fingerprint in {x.fingerprint for x in scenes}: raise ValueError('heldout/training PLY alias')
                    record = dict(s.record,**s.spec)
                    if (s.out/'scene.json').exists(): validate_scene(read(s.out/'scene.json'),record)
                    else: json_write(s.out/'scene.json',record)
                    evaluate_matrix(model,s,args,out,label,fixed_condition=label=='initial',perceptual=perceptual)
                    del s
        progress[label+'_done'] = True; save()

    if not args.resume: save('initial'); save()
    if not progress.get('digital_preflight_done'):
        from comparison_baselines.run import _execute
        _execute(args.digital_python,'comparison_baselines.digital_link',
            ['self-test','--device',args.digital_device,'--snr-db',30],out/'metrics/digital_preflight.log')
        progress['digital_preflight_done'] = True; save()
    if not progress['initial_done']: evaluate('initial')
    updated = 0
    while progress['step'] < plan['total_updates']:
        phase = progress['phase']
        if progress['completed'][phase] == plan['phase_updates'][phase]:
            if not progress['channel_end_done']: evaluate('channel_end')
            save('channel_end')
            phase = PHASES[1]
            optimizer, lrs = optimizer_for(model,phase,args)  # Reset Adam at the actual objective/scope transition.
            progress.update(phase=phase,phase_step=0,freeze_semantic=False,module_lrs=lrs,next_scene=0)
            save()
        scene_index = progress['next_scene']; s = scenes[scene_index]
        visit = progress['scene_completed'][phase][s.name]+1; step = progress['step']+1
        # Different scene/phase counters, with complete SNR-layout coverage within each.
        snr, layout, stratum = schedule(visit,args.seed+20011*scene_index+(500009 if phase==PHASES[1] else 0),
                                       *args.snr_range,args.snr_bins)
        if s.name not in cache:
            print(f'Caching fixed semantic latent: {s.name}',flush=True)
            with preserved_rng(device): cache[s.name] = cache_teacher(model,s)
        model.train(); model.semantic_encoder.eval(); model.set_scene_geometry(s.geometry)
        if phase == PHASES[0]: model.semantic_decoder.eval()
        optimizer.zero_grad(set_to_none=True)
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        if device.type == 'cuda': torch.cuda.synchronize(device)
        start = time.perf_counter()
        if phase == PHASES[0]:
            index = random.Random(args.seed+100003*step).randrange(len(cache[s.name]))
            batch = cache[s.name][index]
            y, xyz = batch['y'].to(device), batch['xyz'].to(device)
            q = tiers(batch['active'],layout,args.seed+300007*step).to(device)
            centers = torch.stack([torch.as_tensor(c.camera_center).detach().cpu() for c in s.cameras]).to(device)
            loss, stats = loss_for_batch(model,model.semantic_decoder,y,q,xyz,s.geometry,centers,args,progress,snr=snr)
            loss.backward(); stats['batch_index'] = index
        else:
            if s.name not in packed_cache:
                packed_cache[s.name] = [torch.cat((f,c['y']),-1) for f,c in zip(s.groups,cache[s.name])]
            qs = [tiers(ids>=0,layout,args.seed+300007*step+i) for i,ids in enumerate(s.group_ids)]
            indices = random.Random(args.seed+400009*step).sample(range(len(s.cameras)),args.views_per_step)
            task = MultiViewRenderTask([s.cameras[i] for i in indices],s.reference,s.degree,s.white_background)
            loss, stats = cached_render_step(model,list(zip(packed_cache[s.name],qs)),s.geometry,snr,task,args.latent_weight)
            stats['training_view_indices'] = indices
        stats.update(finish_update(model,optimizer,before))
        if device.type == 'cuda': torch.cuda.synchronize(device)
        progress.update(step=step,phase_step=progress['phase_step']+1,next_scene=(scene_index+1)%len(scenes))
        progress['completed'][phase] += 1; progress['scene_completed'][phase][s.name] = visit
        row = dict(stats,step=step,phase=phase,scene=s.name,scene_step=visit,snr_db=snr,
            tx_condition_snr_db=snr,rx_condition_snr_db=snr,snr_stratum=stratum,layout=layout,
            loss=float(loss.detach()),step_seconds=time.perf_counter()-start,module_lrs=progress['module_lrs'])
        append_json(out/'metrics/loss.jsonl',row)
        if visit == 1 or visit%10 == 0:
            print(f'{s.name} {phase} {visit}: SNR={snr:.3f}, q={layout}, y={stats["latent_mse"]:.6g}, '
                  f'loss={row["loss"]:.6g}, sec={row["step_seconds"]:.2f}',flush=True)
        if progress['next_scene'] == 0 and (visit%args.save_every == 0 or
                progress['completed'][phase] == plan['phase_updates'][phase]): save()
        updated += 1
        if max_updates is not None and updated >= max_updates: save(); return dict(progress,status='interrupted')
    progress['phase_complete'] = True
    if not progress['final_done']: evaluate('final')
    save('final'); save()
    # Actual digital delivery follows the codec test matrix and is independently resumable per trial.
    if args.lpips and perceptual is None:
        with preserved_rng(device): perceptual = LPIPSEvaluator(device)
    for name in args.delivery_scenes:
        existing = next((s for s in scenes if s.name==name),None)
        if existing is None:
            spec = next(s for s in heldouts if s['name']==name); raw,degree=read_ply(spec['ply'])
            existing = make_scene(spec,raw,degree,model,args,out)
        actual_delivery(model,existing,args,out,perceptual=perceptual)
    result = dict(status='complete',step=progress['step'],phase_completed=progress['completed'],
        scene_completed=progress['scene_completed'],frozen_semantic_encoder_verified=True,
        frozen_attribute_statistics_verified=True,selection='fixed_final',codec_id=model_id(model),
        train_snrs=args.snr_range,actual_side_delivery_completed=True)
    json_write(out/'complete.json',result); json_write(out/'progress.json',dict(progress,status='complete'))
    summarize(out,plan); write_review(out)
    print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':
    main(parser().parse_args())
