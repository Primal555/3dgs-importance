"""Paired Truck q1 -> q2 -> q3 training: joint updates versus frozen prefixes.

Independent experiment; production training and checkpoint schemas unchanged.
Both arms start from identical weights and fresh Adam at EACH phase boundary.
Only the allowed parameter elements receive gradients; no weight decay, clipping,
rate penalty or accept/reject update rule. Reliable 16bit XYZ stays fixed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .optimization import preserved_rng
from .residual_cooperation import evaluate_scene, probe_batch, render_gradient, _append
from .transport import load_checkpoint, model_id, save_checkpoint


OPTIMIZER_NOTE = 'fresh Adam at EACH q1/q2/q3 phase, BOTH arms; no historical moments; weight_decay=0'


def validate_model(model):
    if (model.cfg.decoder_mode != 'residual' or model.cfg.prefix_mode != 'progressive'
            or tuple(model.cfg.rates) != (0, 4, 12, 24)
            or model.cfg.position_delivery != 'quantized' or model.cfg.position_bits != 16):
        raise ValueError('checkpoint must use residual attributes, 0/4/12/24 rates and reliable 16bit XYZ')


class StageUpdater:
    """Element-level Adam masks, including encoder rows and decoder input columns.

    A tensor with both frozen/trainable slices has requires_grad=True. We mask
    its gradient BEFORE Adam. Phase-local zero moments and zero weight decay
    keep forbidden slices exactly unchanged. Audits fail rather than silently
    repairing a broken freeze. Baselines stay on the model device.
    """
    def __init__(self, model, tier, policy, lr):
        validate_model(model)
        if tier not in (1, 2, 3) or policy not in ('joint', 'frozen'):
            raise ValueError('invalid stage/policy')
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('lr must be finite and positive')
        self.model, self.tier, self.policy = model, tier, policy
        self.baseline = {n:v.detach().clone() for n,v in model.state_dict().items()}
        self.masks = {}
        first, last = ((0, 8), (8, 24), (24, 48))[tier-1]
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            mask = torch.zeros_like(parameter, dtype=torch.bool)
            if name.startswith('learned.') and not name.startswith('learned.heads.xyz.'):
                if policy == 'joint':
                    mask.fill_(True)
                elif name.startswith('learned.symbol_head.'):
                    mask[first:last] = True
                elif name == 'learned.dec_in.weight':
                    mask[:, first:last] = True
                    mask[:, 48+first:48+last] = True
                elif name == 'learned.tier.weight':
                    mask[tier] = True
                elif name.startswith('learned.refinements.'):
                    if tier >= 2 and name.startswith(f'learned.refinements.{tier-2}.'):
                        mask.fill_(True)
                elif tier == 1:
                    mask.fill_(True)  # shared encoder/SNR/decoder, base attributes
            self.masks[name] = mask
            parameter.requires_grad_(bool(mask.any()))
        self.optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                          lr=lr, weight_decay=0.)

    @torch.no_grad()
    def audit(self):
        largest, checked = 0., 0
        for name, value in self.model.state_dict().items():
            mask = self.masks.get(name)
            forbidden = torch.ones_like(value,dtype=torch.bool) if mask is None else ~mask
            checked += int(forbidden.sum())
            if forbidden.any():
                current, original = value[forbidden], self.baseline[name][forbidden]
                if not torch.equal(current.contiguous().view(torch.uint8),
                                   original.contiguous().view(torch.uint8)):
                    difference = float((current-original).abs().max())
                    raise RuntimeError(f'frozen state changed: {name}, max_abs={difference}')
        return {'forbidden_max_abs':largest, 'frozen_elements_checked':checked}

    def step(self):
        before = {n:p.detach().clone() for n,p in self.model.named_parameters() if p.requires_grad}
        raw_norm, masked_norm = 0., 0.
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f'nonfinite gradient: {name}')
            raw_norm += float(parameter.grad.double().square().sum())
            parameter.grad.mul_(self.masks[name])
            masked_norm += float(parameter.grad.double().square().sum())
        if masked_norm == 0:
            raise RuntimeError('zero allowed gradient; refusing to count a no-op as a training update')
        self.optimizer.step()
        update_norm = sum(float((p.detach()-before[n]).double().square().sum())
                          for n,p in self.model.named_parameters() if n in before)**.5
        if not math.isfinite(update_norm) or update_norm == 0:
            raise RuntimeError('nonfinite/zero parameter update')
        self.audit()
        return {'grad_norm_before_mask':raw_norm**.5,'grad_norm':masked_norm**.5,
                'update_norm':update_norm}


@torch.no_grad()
def prefix_snapshot(model, scene, tier, snr, kind, seed, blocks):
    """Fixed blocks/full-slot noise; snapshots cover raw outputs AND sent symbols."""
    from .cli import seed_all
    device = next(model.parameters()).device
    predictions, symbols = [], []
    points, remaining = 0, blocks
    with preserved_rng(device):
        seed_all(seed)
        for features, ids in zip(scene.groups, scene.group_ids):
            take = min(len(features), remaining)
            features, ids = features[:take].to(device), ids[:take].to(device)
            valid = ids >= 0
            q = torch.where(valid,tier,0)
            signal = model.learned.encode_full(features,features[...,:3],valid,snr)
            # Apply the [blocks, points] mask first, yielding [valid points,
            # real slots]. Mixing it with a slice in one index expression
            # slices the point axis on older PyTorch (e.g. 2.4), not slots.
            symbols.append(signal[valid][:, :2*model.cfg.rates[tier]].cpu())
            predictions.append(probe_batch(model,features,q,scene.geometry,snr,kind).cpu())
            points += int(valid.sum())
            remaining -= take
            if remaining == 0:
                break
    return {'prediction':torch.cat(predictions),'symbols':torch.cat(symbols),
            'points':points,'blocks':blocks-remaining}


def run_controls(model, scene, out, steps=(1000,1000,1000), lr=1e-5, validate_every=100,
                 views_per_step=2, heldout_views=4, validation_trials=2, snr=10.,
                 kind='awgn', seed=42, white_background=False, make_plots=True,
                 image_views=1, audit_blocks=4, checkpoint_training=None):
    """Same update schedule in both arms, original-PLY render MSE, no allocator."""
    from .render_objective import spaced_indices
    validate_model(model)
    if (len(steps)!=3 or any(not isinstance(s,int) or s<1 for s in steps) or
            validate_every<1 or not 1<=views_per_step<=len(scene.cameras) or
            heldout_views<1 or validation_trials<1 or image_views<0 or audit_blocks<1):
        raise ValueError('invalid update/view/audit counts')
    if not math.isfinite(lr) or lr<=0 or not math.isfinite(snr) or not 0<=seed<2**32:
        raise ValueError('invalid lr/SNR/seed')
    if kind not in ('awgn','none'):
        raise ValueError('only AWGN/noiseless channels supported')
    if not scene.val_cameras or set(map(id,scene.cameras)) & set(map(id,scene.val_cameras)):
        raise ValueError('need disjoint training and heldout cameras')
    out = Path(out)
    # CLI creates plan/scene metadata first; never overwrite existing experiment.
    if any((out/f).exists() for f in ('joint','frozen','summary.json','sampling_plan.jsonl')):
        raise FileExistsError('experiment exists; use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    device = next(model.parameters()).device
    initial = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    flags = {n:p.requires_grad for n,p in model.named_parameters()}
    grads = {n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
    original_mode, identity = model.training, model_id(model)
    heldout = [scene.val_cameras[i] for i in spaced_indices(len(scene.val_cameras),heldout_views)]
    validation_seeds = [(seed+2000003+t*104729)%2**32 for t in range(validation_trials)]
    schedule = []
    for tier,count in enumerate(steps,1):
        for phase_step in range(1,count+1):
            step = len(schedule)+1
            generator = torch.Generator().manual_seed(seed+step*7919)
            indices = torch.randperm(len(scene.cameras),generator=generator)[:views_per_step].tolist()
            row = {'step':step,'phase_step':phase_step,'tier':tier,
                   'noise_seed':(seed+step*104729)%2**32,'camera_indices':indices,
                   'camera_names':[str(scene.cameras[i].image_name) for i in indices]}
            schedule.append(row)
            _append(out/'sampling_plan.jsonl',row)
    losses, validations, arms = [], [], {}

    def validate(arm,step,phase,image_directory=None):
        model.eval()
        for tier in (1,2,3):
            samples = []
            for trial,noise_seed in enumerate(validation_seeds):
                images = image_directory/f'q{tier}' if image_directory is not None and trial==0 else None
                metrics = evaluate_scene(model,scene,heldout,tier,snr,kind,noise_seed,
                    white_background,image_directory=images,image_views=image_views)
                samples.append({'trial':trial,'noise_seed':noise_seed,**metrics})
            row = {'arm':arm,'step':step,'training_tier':phase,'tier':tier,
                   'symbols':model.cfg.rates[tier],'heldout_views':len(heldout),
                   'validation_trials':validation_trials,
                   **{key:sum(s[key] for s in samples)/len(samples) for key in ('mse','psnr','ssim','l1')},
                   'samples':samples}
            validations.append(row)
            _append(out/arm/'validation.jsonl',row)
            print(f'{arm} validation {step}: q{tier} PSNR={row["psnr"]:.3f}, SSIM={row["ssim"]:.4f}',flush=True)

    try:
        with preserved_rng(device):
            for arm in ('joint','frozen'):
                model.load_state_dict(initial,strict=True)
                model.zero_grad(set_to_none=True)
                model.eval()  # deterministic modules; autograd remains enabled
                (out/arm).mkdir()
                arms[arm] = {'initial_model_id':model_id(model),'actual_updates':0}
                if arms[arm]['initial_model_id'] != identity:
                    raise RuntimeError('paired initializer mismatch')
                validate(arm,0,0,out/'images/initial' if arm=='joint' else None)
                offset = 0
                for tier,count in enumerate(steps,1):
                    updater = StageUpdater(model,tier,arm,lr)
                    audit_seed = (seed+3000017)%2**32
                    baselines = {q:prefix_snapshot(model,scene,q,snr,kind,audit_seed,audit_blocks)
                                 for q in range(1,tier)}
                    allowed = {n:int(mask.sum()) for n,mask in updater.masks.items() if mask.any()}
                    (out/arm/f'q{tier}_trainable.json').write_text(json.dumps(allowed,indent=2),encoding='utf-8')
                    print(f'{arm} q{tier}: {count} actual updates, lr={lr:g}, allowed elements={sum(allowed.values())}',flush=True)
                    for plan in schedule[offset:offset+count]:
                        started = time.perf_counter()
                        cameras = [scene.cameras[i] for i in plan['camera_indices']]
                        loss,_,_ = render_gradient(model,scene,cameras,tier,snr,kind,plan['noise_seed'],
                                                   white_background,collect_snapshot=False)
                        if not math.isfinite(loss):
                            raise RuntimeError('nonfinite render loss')
                        update = updater.step()
                        if device.type=='cuda':
                            torch.cuda.synchronize(device)
                        row = {**plan,'arm':arm,'loss':loss,'lr':lr,**update,
                               'step_seconds':time.perf_counter()-started}
                        losses.append(row)
                        _append(out/arm/'loss.jsonl',row)
                        arms[arm]['actual_updates'] += 1
                        if plan['phase_step']==1 or plan['phase_step']%10==0 or plan['phase_step']==count:
                            print(f'{arm} q{tier} {plan["phase_step"]}/{count}: loss={loss:.7g}, '
                                  f'grad={update["grad_norm"]:.4g}, update={update["update_norm"]:.4g}, '
                                  f'sec={row["step_seconds"]:.2f}',flush=True)
                        if plan['step']%validate_every==0 or plan['phase_step']==count:
                            audit = {**updater.audit(),'arm':arm,'tier':tier,'step':plan['step'],
                                     'audit_noise_seed':audit_seed,'lower_prefixes':[]}
                            for q,baseline in baselines.items():
                                current = prefix_snapshot(model,scene,q,snr,kind,audit_seed,audit_blocks)
                                drift = {'tier':q,'points':current['points'],'blocks':current['blocks'],
                                    'prediction_max_abs':float((current['prediction']-baseline['prediction']).abs().max()),
                                    'symbol_max_abs':float((current['symbols']-baseline['symbols']).abs().max())}
                                if arm=='frozen' and max(drift['prediction_max_abs'],drift['symbol_max_abs'])>1e-6:
                                    raise RuntimeError(f'frozen prefix functional audit failed: {drift}')
                                audit['lower_prefixes'].append(drift)
                            _append(out/arm/'freeze_audits.jsonl',audit)
                            final_images = out/'images'/arm/'final' if plan['step']==len(schedule) else None
                            validate(arm,plan['step'],tier,final_images)
                    folder = out/arm/'checkpoints'/f'q{tier}'
                    folder.mkdir(parents=True)
                    training = {**(checkpoint_training or {}),'experiment':'residual_freeze_controls',
                                'policy':arm,'completed_tier':tier,'lr':lr,'optimizer':OPTIMIZER_NOTE,
                                'updates_from_initializer':offset+count,'initializer_model_id':identity}
                    save_checkpoint(folder/'codec.pt',model,offset+count,training)
                    torch.save({'optimizer':updater.optimizer.state_dict(),'training':training},folder/'optimizer.pt')
                    offset += count
                arms[arm]['final_model_id'] = model_id(model)
    finally:
        model.load_state_dict(initial,strict=True)
        model.train(original_mode)
        for name,parameter in model.named_parameters():
            parameter.requires_grad_(flags[name])
            parameter.grad = grads[name]
    summary = {'updates_per_arm':len(schedule),'stage_updates':list(steps),'lr':lr,
               'optimizer':OPTIMIZER_NOTE,'arms':arms,'source_model_unchanged':model_id(model)==identity,
               'loss_target':'original source PLY render MSE; no rate/attribute auxiliary loss',
               'validation_views':[str(c.image_name) for c in heldout],
               'validation_noise_seeds':validation_seeds,'validation_trials':validation_trials,
               'audit_note':'all frozen parameter/buffer elements checked every update; functional prefix checks on fixed blocks',
               'final_metrics':[r for r in validations if r['step']==len(schedule)],
               'initial_metrics':[r for r in validations if r['step']==0]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    if make_plots:
        plot_controls(out,losses,validations,steps,summary)
    return summary


def plot_controls(out,losses,validations,steps,summary):
    """Static experiment charts: paired same-axis MSE, PSNR/SSIM small multiples.

    Chart contract: compare learning/retention, not assume a winning arm; ordered
    updates (normally 3000 loss points and >=31 validation points), blue solid vs
    orange dashed, raw MSE plus within-phase trailing average, same metrics/axes
    within each row. Synthetic smoke runs have too few points for conclusions.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(out)/'charts'
    folder.mkdir(parents=True,exist_ok=True)
    styles = {'joint':('#316D9C','-'),'frozen':('#D56B2A','--')}
    boundaries = (steps[0],steps[0]+steps[1])
    fig,ax = plt.subplots(figsize=(10,4.5),layout='constrained')
    for arm,(color,line) in styles.items():
        rows = [r for r in losses if r['arm']==arm]
        ax.plot([r['step'] for r in rows],[r['loss'] for r in rows],color=color,alpha=.22,linewidth=.8)
        smooth = []
        for index,row in enumerate(rows):
            window = [x['loss'] for x in rows[max(0,index-19):index+1] if x['tier']==row['tier']]
            smooth.append(sum(window)/len(window))
        ax.plot([r['step'] for r in rows],smooth,color=color,linestyle=line,label=arm,linewidth=1.5)
    for boundary in boundaries:
        ax.axvline(boundary,color='#555555',linestyle=':',linewidth=.8)
    ax.set(xlabel='Actual optimizer updates per arm (q1 -> q2 -> q3)',ylabel='Source PLY render MSE',
           title='Sequential prefix training: image MSE\nRaw traces + trailing 20-update mean, reset at each phase')
    ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax.grid(alpha=.16); ax.legend()
    fig.savefig(folder/'loss_comparison.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(12,7),sharex=True,sharey='row',layout='constrained')
    for column,tier in enumerate((1,2,3)):
        for row,key in enumerate(('psnr','ssim')):
            ax = axes[row,column]
            for arm,(color,line) in styles.items():
                values = [r for r in validations if r['arm']==arm and r['tier']==tier]
                ax.plot([v['step'] for v in values],[v[key] for v in values],color=color,linestyle=line,
                        marker='o',markersize=3,markerfacecolor='white' if arm=='frozen' else color,label=arm)
            for boundary in boundaries:
                ax.axvline(boundary,color='#555555',linestyle=':',linewidth=.8)
            ax.set_title(f'q{tier}: {((4,12,24)[column])} complex symbols / Gaussian')
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.grid(alpha=.16)
            if column==0:
                ax.set_ylabel('PSNR (dB)' if key=='psnr' else 'SSIM')
            if row==1:
                ax.set_xlabel('Actual updates per arm')
    axes[0,0].legend()
    fig.suptitle(f'Fixed heldout views: prefix quality during sequential training\n'
                 f'{len(summary["validation_views"])} cameras, {summary["validation_trials"]} fixed noise trials; target = source PLY')
    fig.savefig(folder/'validation_comparison.png',dpi=150); plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='residual experiment root OR its residual subdirectory')
    p.add_argument('--out',required=True,help='NEW directory outside the source run')
    p.add_argument('--checkpoint',help='optional checkpoint inside source checkpoints')
    p.add_argument('--manifest',help='migrated paths allowed, exact original Truck/split required')
    p.add_argument('--stage-steps',type=int,nargs=3,default=[1000,1000,1000],metavar=('Q1','Q2','Q3'))
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--validate-every',type=int,default=100)
    p.add_argument('--views-per-step',type=int,default=2)
    p.add_argument('--heldout-views',type=int,default=4)
    p.add_argument('--validation-trials',type=int,default=2)
    p.add_argument('--image-views',type=int,default=1)
    p.add_argument('--audit-blocks',type=int,default=4)
    p.add_argument('--blocks-per-batch',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def main(args):
    source = Path(args.run).resolve()
    run = source/'residual' if (source/'residual/training.json').is_file() else source
    out = Path(args.out).resolve()
    if out==source or source in out.parents or out in source.parents:
        raise ValueError('output must be outside the source run, not its ancestor or descendant')
    if out.exists():
        raise FileExistsError(f'use a new output directory: {out}')
    if (any(s<1 for s in args.stage_steps) or args.validate_every<1 or args.views_per_step<1 or
            args.heldout_views<1 or args.validation_trials<1 or args.image_views<0 or args.audit_blocks<1 or
            args.cpu_threads<1 or (args.blocks_per_batch is not None and args.blocks_per_batch<1) or
            not math.isfinite(args.lr) or args.lr<=0 or not 0<=args.seed<2**32):
        raise ValueError('invalid experimental configuration')
    record = json.loads((run/'training.json').read_text(encoding='utf-8'))
    checkpoint = (Path(args.checkpoint) if args.checkpoint else run/'checkpoints/final/codec.pt').resolve()
    if (run/'checkpoints').resolve() not in checkpoint.parents:
        raise ValueError('checkpoint must be inside source checkpoints')
    if record.get('decoder_mode')!='residual' or record.get('channel','awgn') not in ('awgn','none'):
        raise ValueError('need a residual AWGN/noiseless source experiment')
    scene_identity = json.loads((run/'scenes/truck/scene.json').read_text(encoding='utf-8'))
    plan = {**vars(args),'source_run':str(run),'checkpoint':str(checkpoint),
            'manifest':args.manifest or record['manifest'],'optimizer':OPTIMIZER_NOTE,
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'scope':'Truck only; same q1->q2->q3 order, cameras, noise, phase resets in BOTH arms',
            'loss':'original PLY render MSE; XYZ fixed; no allocator/rate penalty',
            'snr':record['snr'],'channel':record['channel']}
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:
        return
    from .cli import device_for
    from .data import read_ply
    from .multiscene import Scene, manifest, parser as training_parser
    device = device_for(args.device)
    if device.type=='cpu':
        torch.set_num_threads(args.cpu_threads)
    model = load_checkpoint(checkpoint,device)
    validate_model(model)
    opts = training_parser().parse_args(['--out',str(out)])
    for name in ('blocks_per_batch','resolution','white_background','images','validation_views','net_bits_per_use'):
        setattr(opts,name,record[name])
    opts.blocks_per_batch = args.blocks_per_batch or opts.blocks_per_batch
    opts.views_per_step = args.views_per_step
    spec = next(s for s in manifest(plan['manifest']) if s['name']=='truck')
    raw,degree = read_ply(spec['ply'])
    if degree!=model.cfg.sh_degree:
        raise ValueError('Truck SH degree differs from checkpoint')
    out.mkdir(parents=True,exist_ok=False)
    scene = Scene(spec,raw,degree,model,opts,out,with_mask=False)
    current = json.loads((scene.out/'scene.json').read_text(encoding='utf-8'))
    for name in ('fingerprint','source_gaussians','train_view_names','validation_view_names'):
        if current[name]!=scene_identity[name]:
            raise ValueError(f'Truck identity/split differs from source: {name}')
    plan.update(source_model_id=model_id(model),source_gaussians=len(raw),scene_identity_verified=True,
                blocks_per_batch=opts.blocks_per_batch,
                source_step=int(torch.load(checkpoint,map_location='cpu',weights_only=True)['step']))
    del raw
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    summary = run_controls(model,scene,out,steps=tuple(args.stage_steps),lr=args.lr,
        validate_every=args.validate_every,views_per_step=args.views_per_step,heldout_views=args.heldout_views,
        validation_trials=args.validation_trials,snr=record['snr'],kind=record['channel'],seed=args.seed,
        white_background=opts.white_background,make_plots=not args.no_plots,image_views=args.image_views,
        audit_blocks=args.audit_blocks,checkpoint_training=record)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=plan['checkpoint_sha256']:
        raise RuntimeError('source checkpoint file changed during experiment')
    (out/'complete.json').write_text(json.dumps({'status':'complete','updates_per_arm':summary['updates_per_arm'],
        'source_checkpoint_unchanged':True,'source_model_unchanged':summary['source_model_unchanged'],
        'output':str(out)},indent=2),encoding='utf-8')
    print(f'Paired sequential training complete: {out}',flush=True)


if __name__ == '__main__':
    main(parser().parse_args())
