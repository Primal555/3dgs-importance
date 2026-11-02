"""Full-scene MaskGaussian-style learning with fixed normalized RD and ending."""
import argparse
import json
import math
import signal
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from .allocation import GaussianTierMask, deployment_tiers
from .allocation_distortion import ImageDistortion, calibrate
from .allocation_mirror import ViewWindowScheduler
from .allocation_objective import FixedRateDistortion
from .budgeted_allocation_evaluation import ExactDeploymentCost, evaluate_map, tier_hash
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_state import read_log, rewrite, upsert_jsonl
from .cli import device_for, seed_all
from .data import read_ply
from .dense_rd_control import DeploymentPlateau
from .dense_rd_training import DenseRateMeter, NormalizedMaskedTask
from .mask_gradient import local_mask_step
from .model_factory import build_codec
from .modular_checkpoint import capture_rng, restore_rng
from .modular_experiment import json_write
from .render_objective import spaced_indices
from .render_validation import append_json
from .successive_multiscene_inputs import make_scene
from .transport import model_id

FAMILY = 'dense_mask_fixed_rd_v1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'scene', 'out'): p.add_argument('--'+key, required=True)
    p.add_argument('--manifest'); p.add_argument('--device', default='cuda')
    for name, default in (('allocation-steps',2000), ('window-views',4), ('calibration-views',4),
                          ('validate-every',100), ('save-every',100), ('profile-every',10),
                          ('test-views',32), ('trials',4), ('image-views',2), ('cpu-threads',4),
                          ('rate-chunk-size',65536), ('min-steps',500), ('patience',3), ('seed',42)):
        p.add_argument('--'+name, type=int, default=default)
    for name, default in (('beta',.1), ('keep-lr',.01), ('tier-lr',.001), ('adam-eps',1e-15),
                          ('snr',10.), ('structure-share',.2), ('lr-factor',.5), ('min-lr-scale',.125),
                          ('rd-tolerance',.001), ('map-tolerance',.005), ('cost-tolerance',.002)):
        p.add_argument('--'+name, type=float, default=default)
    p.add_argument('--lambda-s', type=float)
    p.add_argument('--lpips', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def controller_for(args):
    return DeploymentPlateau(args.min_steps, args.patience, args.lr_factor, args.min_lr_scale,
                             args.rd_tolerance, args.map_tolerance, args.cost_tolerance)


def validate(args):
    positive = ('allocation_steps','window_views','calibration_views','validate_every','save_every',
                'trials','image_views','cpu_threads','rate_chunk_size')
    if any(getattr(args, k) < 1 for k in positive) or min(args.test_views,args.profile_every) < 0:
        raise ValueError('invalid training/evaluation budgets')
    if args.snr != 10.: raise ValueError('this restoration retains fixed 10dB')
    if not math.isfinite(args.beta) or args.beta < 0: raise ValueError('invalid beta')
    for key in ('keep_lr','tier_lr','adam_eps'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) <= 0: raise ValueError('invalid '+key)
    if not 0 < args.structure_share < 1: raise ValueError('invalid structure share')
    if args.lambda_s is not None and (not math.isfinite(args.lambda_s) or args.lambda_s < 0):
        raise ValueError('invalid lambda_s')
    controller_for(args)


@torch.no_grad()
def deploy(mask, snr, seed):
    device = mask.logits.device
    probabilities = torch.cat([mask.probabilities(ids, snr).cpu() for ids in
        torch.arange(len(mask.logits), device=device).split(65536)])
    return deployment_tiers(probabilities, seed=seed, draws=10)


def cpu_mask(mask):
    return {k: v.detach().cpu().clone() for k,v in mask.state_dict().items()}


def atomic_save(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary); temporary.replace(path)


def repair_logs(out, step):
    for path in (Path(out)/'metrics').glob('*.jsonl'):
        rows, _, _ = read_log(path)
        rewrite(path, [r for r in rows if r['step'] <= step])


def finish_report(out, rows, progress, reference):
    q2 = rows['uniform_q2']
    comparisons = {}
    for name in ('latest', 'best_rd'):
        r = rows[name]
        comparisons[name] = dict(source_psnr_difference=r['source_psnr']-q2['source_psnr'],
            photo_psnr_difference=r['photo_psnr']-q2['photo_psnr'],
            cost_saving=1-r['total_channel_uses']/q2['total_channel_uses'],
            rd_difference=r['rd_score']-q2['rd_score'],
            meets_q2_goal=r['source_psnr'] >= q2['source_psnr'] and r['total_channel_uses'] < q2['total_channel_uses'])
    result = dict(version=FAMILY, rows=rows, against_q2=comparisons, progress=progress,
                  objective_reference=reference)
    json_write(Path(out)/'review_summary.json', result)
    archive_path = Path(out)/'review.tar.gz'
    temporary = Path(out)/'review.tar.gz.tmp'
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in sorted(Path(out).rglob('*')):
            relative = path.relative_to(out)
            if path.is_file() and relative.parts[0] in (
                    'evaluation','metrics','deployment','review_summary.json','training.json','plan.json',
                    'progress.json','complete.json'):
                archive.add(path, arcname=str(relative), recursive=False)
    temporary.replace(archive_path)
    return result


def main(args, max_updates=None):
    """max_updates is a bounded interruption hook for exact-resume tests."""
    validate(args)
    saved, cfg, spec, old, options, source_identity = load_inputs(args)
    if cfg.channel_variant != 'successive' or tuple(cfg.rates) != (0,8,24,48):
        raise ValueError('completed successive codec with 0/8/24/48 required')
    out = Path(args.out).resolve(); checkpoint = out/'checkpoints/latest/state.pt'
    execution = {'out','run','manifest','resume','dry_run','device','cpu_threads','save_every','profile_every',
                 'allocation_image_objective'}
    plan = dict(version=FAMILY, source=source_identity,
                settings={k:v for k,v in vars(args).items() if k not in execution},
                historical_commits=['3dc7df8','0e4e411','c495a21'])
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return plan
    if out.exists() and not args.resume: raise FileExistsError('use a new output directory or --resume')
    if args.resume and not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads); seed_all(args.seed)
    model = build_codec(cfg).to(device); model.load_state_dict(saved['state_dict']); del saved
    model.eval().requires_grad_(False); model.zero_grad(set_to_none=True); model.set_training_snrs([args.snr])
    raw, degree = read_ply(spec['ply'])
    scene = make_scene(spec, raw, degree, model, options, out, old=old); del raw
    model.set_scene_geometry(scene.geometry); codec_id = model_id(model)
    selected = set(spaced_indices(len(scene.cameras), args.calibration_views))
    calibration = [c for i,c in enumerate(scene.cameras) if i in selected]
    training = [c for i,c in enumerate(scene.cameras) if i not in selected]
    record = dict(scene.record, calibration_view_names=[str(c.image_name) for c in calibration],
                  allocation_train_view_names=[str(c.image_name) for c in training])
    views = ViewWindowScheduler(record['allocation_train_view_names'], args.seed+9001)
    exact = ExactDeploymentCost(model, scene, codec_id, args.snr, options.net_bits_per_use)
    mask = GaussianTierMask(len(scene.raw)).to(device)
    optimizer = torch.optim.Adam([dict(params=[mask.keep_logits], lr=args.keep_lr),
                                  dict(params=[mask.logits], lr=args.tier_lr)], eps=args.adam_eps)
    controller = controller_for(args)
    for folder in ('metrics','deployment','checkpoints'): (out/folder).mkdir(parents=True, exist_ok=True)
    if args.resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (state['plan'] != plan or state['scene'] != record or state['codec_id'] != codec_id
                or state['config'] != cfg.to_dict()): raise ValueError('exact resume identity/settings mismatch')
        mask.load_state_dict(state['mask']); optimizer.load_state_dict(state['optimizer'])
        views.load_state_dict(state['views']); controller.load_state_dict(state['controller'])
        reference, progress, best = state['reference'], state['progress'], state['best']
        initial_q = state['initial_q']; repair_logs(out, progress['step']); restore_rng(state['rng'])
        if (out/'complete.json').is_file() and (out/'review_summary.json').is_file() and (out/'review.tar.gz').is_file():
            return json.loads((out/'review_summary.json').read_text(encoding='utf-8'))
    else:
        full = torch.full((len(scene.raw),), 3, dtype=torch.long)
        ref_args = argparse.Namespace(**vars(args)); ref_args.lpips = False
        ref_args.allocation_image_objective = ImageDistortion('mse_ssim', 0.)
        measured = evaluate_map(model, scene, full, calibration, ref_args,
            folder=out/'evaluation/q3_reference', label='q3_reference')
        cost = exact.details(full)
        reference = dict(calibrate(measured['source_objective_mse'], measured['source_structure_error'],
            share=args.structure_share, lambda_s=args.lambda_s), c_ref=cost['total_channel_uses'], q3_cost=cost)
        initial_q = deploy(mask, args.snr, args.seed)
        progress = dict(step=0, stop_reason=None, frozen_codec_verified=False, best_step=0)
        best = None
        json_write(out/'plan.json', plan)
        json_write(out/'training.json', dict(plan, scene=record, objective_reference=reference,
            optimizer='Adam; dense hierarchical hard-Gumbel ST; epsilon='+str(args.adam_eps),
            trainable_modules=['per-point keep table', 'per-point conditional tier table'],
            allocation_sampling='single', deployment_draws=10, deployment_seed=args.seed,
            keep_prior=.99, conditional_tier_prior=[1/3]*3,
            rate_proxy='expected single-draw payload + measured XYZ per retained point; detached metadata/block scales',
            gradient_proxy='zero-presence q0 shadow; whole-block positive candidates; same-step replay cache',
            frozen_codec_id=codec_id, checkpoint_scope='table/Adam/controller/views/RNG; source codec referenced'))
    args.allocation_image_objective = ImageDistortion('mse_ssim', reference['lambda_s'])
    objective = FixedRateDistortion(args.beta, reference['d_ref'], reference['c_ref'], 'source_distortion')
    rate_meter = DenseRateMeter(exact, scene, reference)

    def measure(label, q, cameras, *, images=False, perceptual=True):
        ev_args = argparse.Namespace(**vars(args)); ev_args.lpips = args.lpips and perceptual
        result = evaluate_map(model, scene, q, cameras, ev_args, folder=out/'evaluation'/label,
                              label=label, save_images=images)
        cost = exact.details(q)
        measured_step = 0 if label == 'initial' else best['step'] if label == 'best_rd' else progress['step']
        row = dict(result, **cost, step=measured_step, event=label,
                   rd_score=objective.score(result['source_distortion'], cost['total_channel_uses']))
        upsert_jsonl(out/'metrics/validation.jsonl', row)
        return row

    def check_deployment(q):
        nonlocal best
        row = measure(f'calibration_{progress["step"]:06d}', q, calibration, perceptual=False)
        event = controller.observe(progress['step'], row['rd_score'], row['total_channel_uses'], q)
        if best is None or row['rd_score'] < best['rd_score']:
            best = dict(mask=cpu_mask(mask), q=q.clone(), rd_score=row['rd_score'], step=progress['step'])
            progress['best_step'] = progress['step']
        for group, lr in zip(optimizer.param_groups, (args.keep_lr,args.tier_lr)):
            group['lr'] = lr*controller.scale
        upsert_jsonl(out/'metrics/control.jsonl', dict(step=progress['step'], event='deployment',
            rd_score=row['rd_score'], total_channel_uses=row['total_channel_uses'], **event))
        print(f'deployment {progress["step"]}: RD={row["rd_score"]:.7g}, '
              f'tiers={row["tier_counts"]}, action={event["action"]}, lr_scale={controller.scale:g}', flush=True)
        if event['action'] == 'stop': progress['stop_reason'] = 'stable_deployment_plateau'

    def save():
        if model_id(model) != codec_id: raise RuntimeError('frozen codec changed')
        atomic_save(checkpoint, dict(version=1, plan=plan, scene=record, codec_id=codec_id,
            config=cfg.to_dict(), mask=cpu_mask(mask), optimizer=optimizer.state_dict(),
            views=views.state_dict(), controller=controller.state_dict(), reference=reference,
            progress=dict(progress), best=best, initial_q=initial_q, rng=capture_rng()))
        json_write(out/'progress.json', dict(progress, beta=args.beta, best_calibration_rd=best['rd_score'],
            keep_lr=optimizer.param_groups[0]['lr'], tier_lr=optimizer.param_groups[1]['lr']))

    if best is None:
        check_deployment(initial_q); save()
    measure('initial', initial_q, scene.test_cameras, images=True)
    interrupted = []
    previous_handlers = {}
    for name in ('SIGINT','SIGTERM'):
        sig = getattr(signal, name)
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    updates = 0
    try:
        while progress['step'] < args.allocation_steps and not progress['stop_reason']:
            if interrupted: break
            started = time.perf_counter(); step = progress['step']+1
            camera_indices = views.next(args.window_views)
            task = NormalizedMaskedTask([training[i] for i in camera_indices], scene.reference,
                scene.degree, scene.white_background, distortion=args.allocation_image_objective,
                d_ref=reference['d_ref'])
            optimizer.zero_grad(set_to_none=True)
            profile = bool(args.profile_every and (step == 1 or step % args.profile_every == 0))
            loss, stats = local_mask_step(model, mask, scene.groups, scene.group_ids, scene.geometry,
                args.snr, 'awgn', task, args.beta, rate_meter, train_codec=False,
                sampling='single', reuse_codec=True, rate_chunk_size=args.rate_chunk_size, profile=profile)
            norm = torch.nn.utils.clip_grad_norm_(mask.parameters(), math.inf, error_if_nonfinite=True)
            optimizer.step(); progress['step'] = step; updates += 1
            if not all(torch.isfinite(p).all() for p in mask.parameters()): raise RuntimeError('nonfinite table')
            if device.type == 'cuda': torch.cuda.synchronize(device)
            stats.update(task.stats)
            stats.update(step=step, phase='allocation', loss=float(loss), beta=args.beta,
                grad_norm=float(norm), keep_lr=optimizer.param_groups[0]['lr'], tier_lr=optimizer.param_groups[1]['lr'],
                d_ref=reference['d_ref'], c_ref=reference['c_ref'],
                allocation_rate_normalizer_definition='full q3 exact total channel uses / source count',
                training_view_names=[str(training[i].image_name) for i in camera_indices],
                step_seconds=time.perf_counter()-started)
            append_json(out/'metrics/training.jsonl', stats)
            if step == 1 or step % 10 == 0:
                print(f'dense allocation {step}/{args.allocation_steps}: MSE={stats["source_mse"]:.7g}, '
                    f'L={float(loss):.7g}, sampled={stats["sampled_tier_counts"]}, '
                    f'keep_mean={stats["mean_keep_probability"]:.5f}, '
                    f'image_rows={stats["mask_image_grad_nonzero_rows"]}/{len(scene.raw)}, '
                    f'sec={stats["step_seconds"]:.2f}', flush=True)
            if step % args.validate_every == 0 or step == args.allocation_steps:
                check_deployment(deploy(mask, args.snr, args.seed)); save()
            elif step % args.save_every == 0: save()
            if max_updates is not None and updates >= max_updates: interrupted.append(0)
        if interrupted and not progress['stop_reason'] and progress['step'] < args.allocation_steps:
            save(); print('interrupted: latest table/Adam/RNG saved', flush=True); return dict(progress)
    finally:
        for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
    # A resumed run can end between the last calibration and a new early-stop request.
    if controller.history[-1]['step'] != progress['step']:
        check_deployment(deploy(mask, args.snr, args.seed))
    progress['stop_reason'] = progress['stop_reason'] or 'maximum_steps'
    progress['frozen_codec_verified'] = model_id(model) == codec_id
    if not progress['frozen_codec_verified']: raise RuntimeError('frozen codec changed')
    save()
    latest_q = deploy(mask, args.snr, args.seed)
    rows = dict(initial=measure('initial', initial_q, scene.test_cameras, images=True),
        latest=measure('latest', latest_q, scene.test_cameras),
        best_rd=measure('best_rd', best['q'], scene.test_cameras, images=True),
        uniform_q2=measure('uniform_q2', torch.full_like(latest_q, 2), scene.test_cameras),
        uniform_q3=measure('uniform_q3', torch.full_like(latest_q, 3), scene.test_cameras))
    for name, q in (('initial',initial_q), ('latest',latest_q), ('best_rd',best['q'])):
        np.save(out/'deployment'/f'{name}_tiers.npy', q.numpy().astype(np.uint8))
    completion = dict(status='complete', version=FAMILY, **progress, best_calibration_rd=best['rd_score'],
        tier_sha256=tier_hash(best['q']))
    json_write(out/'complete.json', completion)
    result = finish_report(out, rows, progress, reference)
    print(json.dumps(dict(completion=completion, against_q2=result['against_q2']), ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__': main(parser().parse_args())
