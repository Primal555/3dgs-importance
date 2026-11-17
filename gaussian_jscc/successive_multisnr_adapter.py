"""Two-scene SNR adaptation with a frozen 10 dB backbone and 32-view supervision."""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from . import successive_q3_refinement as base
from .channel_normalization_metrics import cache_teacher
from .cli import device_for, seed_all
from .data import read_ply
from .model_factory import build_codec
from .modular_checkpoint import capture_rng, restore_rng, save_training_checkpoint, validate_scene
from .modular_experiment import json_write, repair_logs
from .multisnr_adapter_training import train_step
from .multisnr_evaluation import evaluate_matrix, actual_delivery, read
from .multisnr_training import tiers
from .optimization import preserved_rng, clip_codec_gradients
from .perceptual_metrics import LPIPSEvaluator
from .refinement_views import refinement_split, make_refinement_scene, RefinementViewCycle
from .render_validation import append_json
from .snr_adapter import adapter_parameter, configure_optimizer
from .successive_multiscene_inputs import make_scene
from .successive_multisnr import write_review
from .transport import model_id

FAMILY = 'successive_multisnr_adapter_v1'


def parser():
    p = base.parser()
    p.description = __doc__
    p.set_defaults(scenes=['truck', 'playroom'], steps_per_scene=100, validate_every=25)
    p.add_argument('--views-per-step', type=int, default=32)
    p.add_argument('--validation-views', type=int, default=32)
    p.add_argument('--adapter-hidden', type=int, default=64)
    p.add_argument('--eval-snrs', nargs='+', type=float, default=[5., 7.5, 10., 12.5, 15.])
    p.add_argument('--monitor-trials', type=int, default=2)
    p.add_argument('--noiseless-every', type=int, default=4)
    p.add_argument('--noiseless-weight', type=float, default=.25)
    p.add_argument('--response-weight', type=float, default=.1)
    p.add_argument('--response-directions', type=int, default=8)
    return p


def frozen_digest(state):
    return base.frozen_digest({k: v for k, v in state.items() if not adapter_parameter(k)}, ())


def inputs(args):
    requested = dict(views_per_step=args.views_per_step, validation_views=args.validation_views,
                     eval_snrs=args.eval_snrs)
    saved, metadata, source_cfg, specs, heldouts, plan = base.inputs(args)
    for k, v in requested.items():
        setattr(args, k, v)
    if source_cfg.channel_snr_adapter_hidden:
        raise ValueError('initialize adapters from the original fixed-10dB shared codec')
    if min(args.adapter_hidden, args.monitor_trials, args.noiseless_every, args.response_directions) < 1:
        raise ValueError('positive adapter, trial and supervision settings required')
    if args.response_directions % 2 or any(not math.isfinite(v) or v < 0 for v in (args.noiseless_weight, args.response_weight)):
        raise ValueError('even response directions and finite nonnegative auxiliary weights required')
    if (len(set(args.eval_snrs)) != len(args.eval_snrs) or 10. not in args.eval_snrs or
            any(not math.isfinite(s) or not 5. <= s <= 15. for s in args.eval_snrs)):
        raise ValueError('unique evaluation SNRs in [5,15], including 10 dB, required')
    args.eval_snrs = sorted(args.eval_snrs)
    args.snr_range = [5., 15.]
    args.eval_layouts = ['1', '2', '3', 'mixed']
    args.eval_paths = ['clean', 'finite_noiseless', 'awgn']
    args.delivery_layouts = ['2']
    args.side_link_policy = 'snr_qpsk_half_v1'
    cfg = replace(source_cfg, channel_snr_adapter_hidden=args.adapter_hidden)
    splits = {s['name']: refinement_split(metadata['scene_records'][s['name']],
                  args.validation_views, args.views_per_step) for s in specs}
    settings = {k: v for k, v in vars(args).items() if k not in ('run','out','resume','dry_run','net_bits_per_use')}
    plan.update(version=FAMILY, settings=settings, codec_config=cfg.to_dict(),
        total_updates=len(specs)*args.steps_per_scene,
        full_scene_training_recoveries=3*len(specs)*args.steps_per_scene,
        training_render_view_uses=3*len(specs)*args.steps_per_scene*args.views_per_step,
        frozen_state_sha256=frozen_digest(saved['state_dict']), camera_splits=splits,
        trainable_prefixes=['*.snr_adapter.*'],
        objective='mean of three SNR source RGB MSE + latent MSE; per-scene rotating noiseless source auxiliary',
        optimizer_initialization='fresh Adam; SNR adapters only; all original parameters frozen',
        selection='fixed final update; calibration/test reporting only',
        source_supervision='original PLY render and attributes; semantic latent cached',
        condition='original condition fixed at 10dB plus zero-initialized A(s)-A(10); shared at both ends',
        snr_sampling='one sample in each of three equal-width dB strata in [5,15] per update',
        layout_sampling='independent per-scene shuffled cycles of q1/q2/q3/mixed',
        noise='same standard normal draws across the three SNRs; scaled by actual SNR; new seed every scene visit',
        side_link_policy=args.side_link_policy,
        actual_delivery_scope=dict(scenes=args.delivery_scenes, layouts=['2'], snrs=args.eval_snrs, trials=args.trials),
        validation_protocol='source validation plus training-side held-out views; source test unchanged')
    return saved, metadata, source_cfg, cfg, specs, heldouts, json.loads(json.dumps(plan))


@torch.no_grad()
def initialization_probe(model, source, scene, seed):
    """Compare real receiver paths under matched noise before optimizer training."""
    device = next(model.parameters()).device
    model.eval(); source.eval()
    model.set_scene_geometry(scene.geometry); source.set_scene_geometry(scene.geometry)
    model.set_training_snrs([5., 15.]); source.set_training_snrs([10.])
    features = scene.groups[0][:2].to(device)
    active = scene.group_ids[0][:2].to(device) >= 0
    rows = []
    with preserved_rng(device):
        for layout in ('1', '2', '3', 'mixed'):
            q = tiers(active, layout, seed)
            for snr in (5., 10., 15.):
                for kind in ('none', 'awgn'):
                    torch.manual_seed(seed)
                    a = source.forward_tier_batches(features, features[..., :3], F.one_hot(q,4).to(features),
                                                    snr, kind, paired_noise=True)[0]
                    torch.manual_seed(seed)
                    b = model.forward_tier_batches(features, features[..., :3], F.one_hot(q,4).to(features),
                                                   snr, kind, paired_noise=True)[0]
                    torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-5)
                    rows.append(dict(layout=layout, snr_db=snr, path=kind, max_absolute_difference=float((a-b).abs().max())))
    return dict(scene=scene.name, scope='first two original blocks', passed=True, rows=rows)


def summarize(out):
    rows, checks = [], []
    for f in sorted((out/'evaluation/final').glob('*/summary.json')):
        before = read(out/'evaluation/initial'/f.parent.name/'summary.json')['rows']
        after = read(f)['rows']
        key = lambda r: (r['path'], r['snr_db'], r['layout'], r['trial'])
        lookup = {key(r): r for r in before}
        if set(lookup) != {key(r) for r in after}: raise RuntimeError('endpoint matrix changed')
        for r in after:
            a = lookup[key(r)]
            if r['seed'] != a['seed'] or [v['view'] for v in r['views']] != [v['view'] for v in a['views']]:
                raise RuntimeError('paired endpoint views/noise changed')
            if r['total_channel_uses'] != a['total_channel_uses']: raise RuntimeError('fixed-map cost changed')
            metrics = [k for k in base.METRICS if k in r]
            delta = {k: r[k]-a[k] for k in metrics}
            rows.append(dict(scene=r['scene'], path=r['path'], snr_db=r['snr_db'], layout=r['layout'],
                             trial=r['trial'], initial={k:a[k] for k in metrics}, final={k:r[k] for k in metrics},
                             change=delta, total_channel_uses=r['total_channel_uses']))
            if r['snr_db'] == 10. or r['path'] == 'clean':
                passed = all(math.isclose(a[k],r[k],abs_tol=1e-6,rel_tol=1e-6) for k in metrics)
                checks.append(dict(scene=r['scene'], path=r['path'], layout=r['layout'], trial=r['trial'], passed=passed))
    report = dict(status='complete', rows=rows, anchor_checks=checks,
        ten_db_and_clean_preserved=all(c['passed'] for c in checks), cost_protocol='digital_side_file_v1',
        quality_conditioning='reliable side assumption; actual digital delivery reported separately',
        actual_delivery={p.parent.name:read(p)['results'] for p in (out/'actual_delivery').glob('*/summary.json')})
    json_write(out/'review_summary.json', report)
    if not report['ten_db_and_clean_preserved']: raise RuntimeError('10 dB preservation check failed')
    return report


def main(args, max_updates=None):
    saved, metadata, source_cfg, cfg, specs, heldouts, plan = inputs(args)
    if args.dry_run: print(json.dumps(plan,indent=2)); return plan
    out = Path(args.out).resolve()
    if out.exists() and not args.resume: raise FileExistsError('use a new OUT or --resume')
    if args.resume and read(out/'plan.json') != plan: raise ValueError('resume source/settings changed')
    if args.resume and (out/'complete.json').exists(): write_review(out); return read(out/'complete.json')
    device = device_for(args.device)
    torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    original_state = saved['state_dict']
    source = None
    if args.resume:
        saved = torch.load(out/'checkpoints/latest/codec.pt', map_location='cpu', weights_only=True)
        if saved['config'] != cfg.to_dict(): raise ValueError('resume codec config changed')
    model = build_codec(cfg).to(device)
    if args.resume:
        model.load_state_dict(saved['state_dict'])
        progress = saved['training']['progress']
        if progress['training_family'] != FAMILY: raise ValueError('resume family changed')
    else:
        missing, unexpected = model.load_state_dict(original_state, strict=False)
        expected = {k for k in model.state_dict() if adapter_parameter(k)}
        if set(missing) != expected or unexpected: raise ValueError('source/adapter state mismatch')
        with preserved_rng(device): source = build_codec(source_cfg).to(device)
        source.load_state_dict(original_state)
        source.requires_grad_(False)
        progress = dict(training_family=FAMILY, step=0, phase='channel_render', phase_step=0,
            train_snrs=[5.,15.], channel_trained=True, phase_complete=False, snr_adapter_only=True,
            module_lrs={'snr_adapter':args.lr}, next_scene=0, scene_completed={s['name']:0 for s in specs},
            view_schedulers={}, evaluated=[], monitors=[], preflight=False, initialization_verified=False,
            noiseless_counts={s['name']:{str(q):0 for q in (1,2,3)} for s in specs}, optimization_seconds=0.)
        json_write(out/'plan.json', plan)
    model.set_training_snrs([5.,15.])
    optimizer = configure_optimizer(model, args.lr)
    if args.resume:
        optimizer.load_state_dict(saved['training']['optimizer'])
        rng = saved['training']['rng']; old = saved['training']['scene_record']
        repair_logs(out,progress['step'])
    else: rng = capture_rng(); old = None
    del saved, original_state
    scenes = []
    for spec in specs:
        raw, degree = read_ply(spec['ply'])
        scenes.append(make_refinement_scene(spec,raw,degree,model,args,out,
            metadata['scene_records'][spec['name']],plan['camera_splits'][spec['name']]))
    records = {s.name:s.record for s in scenes}
    if old is not None: validate_scene(old,records)
    else: json_write(out/'training.json',dict(plan,scene_records=records))
    for s in scenes: json_write(s.out/'scene.json',dict(s.record,**s.spec))
    restore_rng(rng)
    cycles = {s.name:RefinementViewCycle(records[s.name]['train_view_names'],args.seed,s.name,1,
                progress['view_schedulers'].get(s.name)) for s in scenes}
    for s in scenes:
        if cycles[s.name].updates != progress['scene_completed'][s.name]: raise ValueError('camera/step counter mismatch')
    def save(label='latest'):
        if frozen_digest(model.state_dict()) != plan['frozen_state_sha256']: raise RuntimeError('frozen backbone changed')
        save_training_checkpoint(out/'checkpoints'/label/'codec.pt',model,optimizer,progress,capture_rng(),records)
        json_write(out/'progress.json',dict(progress,status='running'))
    if not args.resume: save('initial'); save()
    if not progress['initialization_verified']:
        if source is None:
            original = torch.load(plan['source_checkpoint'],map_location='cpu',weights_only=True)
            with preserved_rng(device): source = build_codec(source_cfg).to(device)
            source.load_state_dict(original['state_dict']); source.requires_grad_(False)
        json_write(out/'metrics/initialization.json',dict(rows=[initialization_probe(model,source,s,args.seed) for s in scenes]))
        progress['initialization_verified']=True; save()
    del source
    if not progress['preflight']:
        from comparison_baselines.run import _execute
        from comparison_baselines.side_link_policy import side_link_config
        from dataclasses import asdict
        for snr in (5.,10.):
            config=side_link_config(snr,args.side_link_policy,args.digital_device)
            _execute(args.digital_python,'comparison_baselines.digital_link',
                ['self-test','--snr-db',30]+[v for k,x in asdict(config).items() for v in ('--'+k.replace('_','-'),x)],
                out/f'metrics/digital_preflight_{snr:g}.log')
        progress['preflight']=True; save()
    perceptual = None
    if args.lpips:
        with preserved_rng(device): perceptual=LPIPSEvaluator(device)
    def evaluate(label, validation=False):
        done = progress['monitors'] if validation else progress['evaluated']
        if label in done: return
        save()
        options=copy.copy(args)
        if validation: options.trials=args.monitor_trials
        with preserved_rng(device):
            for s in scenes:
                evaluate_matrix(model,s,options,out,label,scope='validation' if validation else 'test',perceptual=perceptual)
            if not validation:
                for spec in heldouts:
                    raw,degree=read_ply(spec['ply']); s=make_scene(spec,raw,degree,model,args,out)
                    if s.fingerprint in {v.fingerprint for v in scenes}: raise ValueError('heldout PLY alias')
                    evaluate_matrix(model,s,options,out,label,perceptual=perceptual)
        done.append(label); save()
        if validation:
            json_write(out/'evaluation'/label/'progress.json',dict(step=progress['step'],phase='adapter',
                optimization_seconds=progress['optimization_seconds'],scope='validation',selection=False))
    def monitor():
        if progress['next_scene']==0:
            visit=progress['step']//len(scenes)
            if visit%args.validate_every==0 or visit==args.steps_per_scene:
                evaluate(f'monitor_{progress["step"]:06d}',True)
    evaluate('initial'); monitor()
    cache={}; updated=0
    while progress['step'] < plan['total_updates']:
        s=scenes[progress['next_scene']]; visit=progress['scene_completed'][s.name]+1
        if s.name not in cache:
            with preserved_rng(device): cache[s.name]=[b['y'] for b in cache_teacher(model,s)]
        indices=cycles[s.name].next(args.views_per_step)
        optimizer.zero_grad(set_to_none=True)
        if device.type=='cuda': torch.cuda.synchronize(device)
        start=time.perf_counter()
        stats=train_step(model,s,cache[s.name],[s.cameras[i] for i in indices],args,visit)
        norm, gradstats=clip_codec_gradients(model,mode='none')
        optimizer.step()
        if device.type=='cuda': torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        progress.update(step=progress['step']+1,phase_step=progress['step']+1,
            next_scene=(progress['next_scene']+1)%len(scenes),optimization_seconds=progress['optimization_seconds']+elapsed)
        progress['scene_completed'][s.name]=visit
        progress['view_schedulers'][s.name]=cycles[s.name].state_dict()
        if 'noiseless_tier' in stats: progress['noiseless_counts'][s.name][str(stats['noiseless_tier'])]+=1
        append_json(out/'metrics/loss.jsonl',dict(stats,**gradstats,grad_norm=float(norm),step=progress['step'],
            scene=s.name,scene_step=visit,step_seconds=elapsed,
            training_view_names=[str(s.cameras[i].image_name) for i in indices],view_coverage=cycles[s.name].coverage()))
        if visit==1 or visit%10==0 or visit==args.steps_per_scene:
            print(f'adapter {s.name} {visit}/{args.steps_per_scene}: q={stats["layout"]}, '
                  f'SNR={[round(v,2) for v in stats["snrs"]]}, render={stats["render_mse"]:.7g}, '
                  f'latent={stats["latent_mse"]:.7g}, grad={float(norm):.5g}, sec={elapsed:.2f}',flush=True)
        if progress['next_scene']==0 and visit%args.save_every==0: save()
        monitor(); updated+=1
        if max_updates is not None and updated>=max_updates: save(); return dict(progress,status='interrupted')
    progress['phase_complete']=True; save('final'); save()
    evaluate('final')
    for name in args.delivery_scenes:
        s=next((s for s in scenes if s.name==name),None)
        if s is None:
            spec=next(s for s in heldouts if s['name']==name)
            raw,degree=read_ply(spec['ply']); s=make_scene(spec,raw,degree,model,args,out)
        actual_delivery(model,s,args,out,perceptual=perceptual)
    report=summarize(out)
    from .multisnr_plots import write_figures
    write_figures(out)
    result=dict(status='complete',version=FAMILY,step=progress['step'],codec_id=model_id(model),
        frozen_backbone_verified=True,ten_db_and_clean_preserved=report['ten_db_and_clean_preserved'],
        scene_completed=progress['scene_completed'],noiseless_counts=progress['noiseless_counts'],
        heldout_optimizer_updates=0,actual_side_delivery_completed=True,selection='fixed_final')
    json_write(out/'progress.json',dict(progress,status='complete'))
    json_write(out/'complete.json',result); write_review(out)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__': main(parser().parse_args())
