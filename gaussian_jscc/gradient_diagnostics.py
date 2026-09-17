"""Bounded paired experiment, not a replacement for the normal training CLI.

Same checkpoint, fresh Adam, per-step RNG, q/block/camera/SNR schedules. The
only treatment is detaching XYZ at the attribute decoder's grid input. Direct
position and rasterizer gradients remain intact. Observer uses autograd.grad
and never clips/steps or modifies RNG; normal backward still runs exactly once.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn.utils.rnn import pad_sequence

from .cli import device_for, sample_tiers, seed_all
from .data import prepare, read_ply, to_features, to_raw
from .losses import reconstruction_loss, objective_stats, position_training_inputs
from .training import full_scene_step
from .transport import load_checkpoint, save_checkpoint
from .optimization import parameter_group


class GradientProbe:
    """Accumulate exact component parameter gradients BEFORE clipping."""
    def __init__(self, model):
        self.named = list(model.named_parameters())
        self.params = [p for _, p in self.named]
        self.groups = {}
        offset = 0
        for name, p in self.named:
            self.groups.setdefault(parameter_group(name), []).append(slice(offset, offset+p.numel()))
            offset += p.numel()
        self.vectors = {}

    def __call__(self, objectives):
        for key, objective in objectives.items():
            grads = torch.autograd.grad(objective, self.params, retain_graph=True, allow_unused=True)
            flat = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                              for p, g in zip(self.params, grads)])
            if not torch.isfinite(flat).all():
                raise RuntimeError(f'Nonfinite {key} component gradient')
            if key not in self.vectors:
                self.vectors[key] = flat
            else:
                self.vectors[key].add_(flat)

    def report(self):
        result = {}
        for key, vec in self.vectors.items():
            result[key] = {'total_norm': float(vec.norm()), 'groups': {
                group: float(torch.cat([vec[s] for s in slices]).norm())
                for group, slices in self.groups.items()}}
        cosines = {}
        keys = list(self.vectors)
        for i, a in enumerate(keys):
            for b in keys[i+1:]:
                for group in ('all', 'geometry_decoder', 'geometry_encoder'):
                    x, y = self.vectors[a], self.vectors[b]
                    if group != 'all':
                        x = torch.cat([x[s] for s in self.groups[group]])
                        y = torch.cat([y[s] for s in self.groups[group]])
                    denom = x.norm() * y.norm()
                    cosines[f'{a}__{b}__{group}'] = float(x.dot(y)/denom) if denom > 1e-20 else None
        result['cosines'] = cosines
        total = sum(self.vectors.values())
        actual = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1)
                            for p in self.params])
        result['component_sum_relative_error'] = float((total-actual).norm()/actual.norm().clamp_min(1e-12))
        return result


def gradient_stats(model):
    groups = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            groups.setdefault(parameter_group(name), []).append(p.grad.detach().reshape(-1))
    return {k: {'norm': float(torch.cat(v).norm()), 'max_abs': float(torch.cat(v).abs().max())}
            for k, v in groups.items()}


def update_stats(model, before):
    groups = {}
    for name, p in model.named_parameters():
        if name not in before:
            continue
        groups.setdefault(parameter_group(name), []).append((before[name], p.detach()-before[name]))
    return {k: {'update_norm': float(torch.cat([d.reshape(-1) for _, d in v]).norm()),
                'relative_update': float(torch.cat([d.reshape(-1) for _, d in v]).norm()/
                                         torch.cat([p.reshape(-1) for p, _ in v]).norm().clamp_min(1e-12))}
            for k, v in groups.items()}


def largest_gradients(model, limit=8):
    rows = [{'parameter':name, 'norm':float(p.grad.norm()), 'max_abs':float(p.grad.abs().max())}
            for name,p in model.named_parameters() if p.grad is not None]
    return sorted(rows,key=lambda r:r['norm'],reverse=True)[:limit]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@torch.no_grad()
def evaluate_blocks(model, blocks, geometry, args, step):
    """Fixed source blocks, all positive tiers; no held-out-scene claim."""
    device = next(model.parameters()).device
    indices = torch.linspace(0, len(blocks)-1, min(args.eval_blocks, len(blocks))).round().long().tolist()
    rows = []
    conditions = [('none', 10.)] + [('awgn', s) for s in args.eval_snrs]
    for kind, snr in conditions:
        for tier in (1, 2, 3):
            squared, distances, objectives, counts = [], [], [], []
            raw_squared, outside = [], []
            for index in indices:
                seed_all(args.seed + 100000 + index)
                f = blocks[index].to(device)
                q = torch.full((len(f),), tier, device=device, dtype=torch.long)
                pred = model(f, f[:, :3], q, snr, kind)
                # Match the actual receiver/render path, which clips bbox units.
                delta = (pred[:, :3].clamp(0,1)-f[:, :3]) * geometry.span.to(device)
                raw_squared.append(((pred[:, :3]-f[:, :3])*geometry.span.to(device)).square().sum())
                outside.append(((pred[:, :3]<0)|(pred[:, :3]>1)).sum())
                squared.append(delta.square().sum())
                distances.append(delta.norm(dim=-1))
                objectives.append(reconstruction_loss(pred, f, geometry, model)*len(f))
                counts.append(len(f))
            n = sum(counts)
            rmse = torch.sqrt(sum(squared)/(3*n))
            distances = torch.cat(distances)
            rows.append({'step':step, 'channel':kind, 'snr':snr, 'tier':tier, 'gaussians':n,
                         'position_rmse':float(rmse), 'position_nrmse':float(rmse/geometry.span.norm().to(device)),
                         'distance_p50':float(distances.median()),
                         'distance_p95':float(torch.quantile(distances, .95)),
                         'unclipped_position_rmse':float(torch.sqrt(sum(raw_squared)/(3*n))),
                         'out_of_bounds_fraction':float(sum(outside)/(3*n)),
                         'reconstruction_loss':float(sum(objectives)/n)})
    return rows


@torch.no_grad()
def evaluate_render(model, blocks, raw, geometry, cameras, degree, args, directory):
    from .rendering import evaluate_views
    device = next(model.parameters()).device
    decoded = []
    for begin in range(0, len(blocks), args.blocks_per_batch):
        seed_all(args.seed + 200000 + begin)
        f = pad_sequence(blocks[begin:begin+args.blocks_per_batch], batch_first=True).to(device)
        q = torch.zeros(f.shape[:2], device=device, dtype=torch.long)
        for i,b in enumerate(blocks[begin:begin+args.blocks_per_batch]):
            q[i,:len(b)] = args.render_eval_tier
        choices = torch.nn.functional.one_hot(q,4).to(f.dtype)
        pred,_,_ = model.forward_tier_batches(f,f[...,:3],choices,args.render_eval_snr,'awgn')
        decoded.append(to_raw(pred[q>0],geometry,model).cpu())
    result = evaluate_views(torch.cat(decoded).to(device), raw.to(device), cameras,
                            degree,args.white_background,directory,hybrid_ablation=True)
    result.update(snr=args.render_eval_snr,tier=args.render_eval_tier,channel='awgn')
    write_json(directory/'metrics.json',result)
    return result['mean']


def make_plots(out, all_logs, evaluations):
    # Optional presentation only; numerical outputs remain authoritative.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; JSON/CSV results are complete, charts skipped.', flush=True)
        return
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for variant, rows in all_logs.items():
        x=[r['step'] for r in rows]
        axes[0,0].plot(x,[r['geometry_loss'] for r in rows],alpha=.65,label=variant)
        axes[0,1].semilogy(x,[r['grad_norm'] for r in rows],alpha=.65,label=variant)
        selected=[r for r in evaluations[variant] if r['tier']==2 and r['channel']=='none']
        axes[1,0].plot([r['step'] for r in selected],[r['position_rmse'] for r in selected],marker='o',label=variant)
        selected=[r for r in rows if 'updates' in r]
        axes[1,1].semilogy([r['step'] for r in selected],
                           [max(r['updates']['geometry_decoder']['relative_update'],1e-20) for r in selected],
                           marker='.',label=variant)
    boundary=next((r['step']-.5 for r in next(iter(all_logs.values())) if r['phase']=='render'),None)
    if boundary is not None:
        for ax in axes.flat:
            ax.axvline(boundary,color='gray',linestyle=':',alpha=.6)
    for ax,title in zip(axes.flat,('Training geometry loss (random conditions)',
                                  'Global gradient norm BEFORE clipping',
                                  'Fixed-block XYZ RMSE: tier2, noiseless',
                                  'Geometry decoder relative Adam update (probed steps)')):
        ax.set_title(title); ax.set_xlabel('Step'); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(out/'diagnostics.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),squeeze=False)
    for column,(variant,rows) in enumerate(all_logs.items()):
        probes=[r for r in rows if 'component_gradients' in r]
        for term in ('geometry','attributes','render'):
            selected=[r for r in probes if term in r['component_gradients']]
            axes[0,column].semilogy([r['step'] for r in selected],
                [max(r['component_gradients'][term]['groups']['geometry_decoder'],1e-20) for r in selected],
                marker='.',label=term)
        for pair in ('geometry__attributes','geometry__render','attributes__render'):
            key=pair+'__geometry_decoder'
            selected=[r for r in probes if r['component_gradients']['cosines'].get(key) is not None]
            axes[1,column].plot([r['step'] for r in selected],
                [r['component_gradients']['cosines'][key] for r in selected],marker='.',label=pair)
        axes[0,column].set_title(variant+': geometry decoder component norms')
        axes[1,column].set_title(variant+': geometry decoder gradient cosines')
        axes[1,column].set_ylim(-1.05,1.05); axes[1,column].axhline(0,color='gray',linewidth=.7)
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.set_xlabel('Step'); ax.legend(fontsize=8)
        if boundary is not None:
            ax.axvline(boundary,color='gray',linestyle=':',alpha=.6)
    fig.tight_layout(); fig.savefig(out/'gradient_components.png',dpi=160); plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('ply','checkpoint','out'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--source')
    p.add_argument('--device',default='cuda')
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--render-steps',type=int,default=100)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--render-lr',type=float,default=1e-5)
    p.add_argument('--attr-weight',type=float,default=1.)
    p.add_argument('--clip-norm',type=float,default=1.)
    p.add_argument('--probe-every',type=int,default=50)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--eval-blocks',type=int,default=8)
    p.add_argument('--eval-snrs',type=float,nargs='+',default=[0.,10.,20.])
    p.add_argument('--snr-range',type=float,nargs=2,default=[0.,20.])
    p.add_argument('--blocks-per-batch',type=int,default=32)
    p.add_argument('--training-data-device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--test-views',type=int,default=2)
    p.add_argument('--render-eval-tier',type=int,choices=[1,2,3],default=2)
    p.add_argument('--render-eval-snr',type=float,default=10.)
    p.add_argument('--resolution',type=int,default=2)
    p.add_argument('--images',default='images')
    p.add_argument('--white-background',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-plots',action='store_true')
    args=p.parse_args(argv)
    if min(args.steps,args.render_steps,args.test_views)<0 or args.steps+args.render_steps<1:
        p.error('nonnegative phase lengths and at least one step required')
    if min(args.probe_every,args.eval_every,args.eval_blocks,args.blocks_per_batch)<1:
        p.error('probe/eval intervals, block counts must be positive')
    if not all(math.isfinite(x) and x>0 for x in (args.lr,args.render_lr,args.clip_norm)):
        p.error('positive finite learning rates and clip norm required')
    if not math.isfinite(args.attr_weight) or args.attr_weight<0:
        p.error('nonnegative finite attr weight required')
    if not all(math.isfinite(x) for x in args.snr_range+args.eval_snrs+[args.render_eval_snr]) or args.snr_range[0]>args.snr_range[1]:
        p.error('invalid SNR range')
    device=device_for(args.device)
    if args.render_steps and (not args.source or device.type!='cuda'):
        p.error('render steps require --source and CUDA; CPU checks use --render-steps 0 --test-views 0')
    if args.source and args.test_views and device.type!='cuda':
        p.error('test renders require CUDA')
    if device.type=='cpu' and args.training_data_device=='cuda':
        p.error('CPU checks require --training-data-device cpu')
    initial=load_checkpoint(args.checkpoint,device)
    if initial.cfg.architecture!='geometry_first':
        p.error('geometry_first checkpoint required')
    raw,degree=read_ply(args.ply)
    if initial.cfg.sh_degree!=degree:
        p.error('PLY/checkpoint SH degree mismatch')
    raw,geometry,_=prepare(raw,initial.cfg.morton_bits)
    storage=device if args.training_data_device=='cuda' else torch.device('cpu')
    with torch.no_grad():
        blocks=[to_features(rb.to(device),geometry,initial)[0].to(storage) for rb in raw.split(initial.cfg.block_size)]
    config=initial.cfg.to_dict()
    del initial
    groups=[pad_sequence(blocks[i:i+args.blocks_per_batch],batch_first=True)
            for i in range(0,len(blocks),args.blocks_per_batch)] if args.render_steps else []
    cameras=test_cameras=reference=None
    if args.render_steps or (args.source and args.test_views):
        from .rendering import load_cameras, render, RenderReference
        from utils.loss_utils import ssim
        if args.render_steps:
            cameras=load_cameras(args.source,args.resolution,args.white_background,args.images,'train')
            reference=RenderReference(raw,degree,args.white_background,'source')
        if args.test_views:
            test_cameras=load_cameras(args.source,args.resolution,args.white_background,args.images,'test')[:args.test_views]
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',{**vars(args),'codec_config':config,'torch_version':str(torch.__version__),
                                 'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                                 'note':'Fresh Adam for both arms; only context XYZ backward path differs. No cross-scene validation.'})
    all_logs,evaluations,summary={},{},{}
    baseline_render=None
    total=args.steps+args.render_steps
    for variant in ('attached','detached'):
        folder=out/variant; folder.mkdir()
        model=load_checkpoint(args.checkpoint,device).train()
        model.detach_attribute_context_xyz=variant=='detached'
        optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        logs=[]; evals=evaluate_blocks(model,blocks,geometry,args,0)
        write_json(folder/'evaluation.json',evals)
        if test_cameras and baseline_render is None:
            baseline_render=evaluate_render(model,blocks,raw,geometry,test_cameras,degree,args,out/'initial_test')
        with (folder/'loss.jsonl').open('w',encoding='utf-8') as log:
            for step in range(1,total+1):
                seed_all(args.seed+step)
                started=time.perf_counter()
                is_render=step>args.steps
                optimizer.param_groups[0]['lr']=args.render_lr if is_render else args.lr
                snr=random.uniform(*args.snr_range)
                probe=GradientProbe(model) if step==1 or step%args.probe_every==0 or step in (args.steps,total,args.steps+1) else None
                optimizer.zero_grad(set_to_none=True)
                if not is_render:
                    index=random.randrange(len(blocks)); f=blocks[index].to(device)
                    q=sample_tiers(len(f),device)
                    pred=model(f,f[:,:3],q,snr,'awgn')
                    loss,terms=reconstruction_loss(pred,f,geometry,model,return_terms=True,
                                                    **position_training_inputs(model,f,q,snr))
                    stats={k+'_loss':float(v.detach().mean()) for k,v in terms.items()}
                    if probe:
                        geo=terms['geometry'].mean()*model.cfg.geometry_weight
                        probe({'geometry':geo,'attributes':loss-geo})
                    loss.backward()
                    identity={'block':index,'tier_counts':torch.bincount(q,minlength=4).tolist(),
                              'tier_sha256':hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest()}
                else:
                    camera=random.choice(cameras); gt=reference.get(camera,device)
                    qs=[sample_tiers(len(f),torch.device('cpu')).to(storage) for f in blocks]
                    batches=[(f,pad_sequence(qs[i*args.blocks_per_batch:(i+1)*args.blocks_per_batch],batch_first=True))
                             for i,f in enumerate(groups)]
                    def distortion(scene):
                        image=render(scene,camera,degree,args.white_background)
                        return .8*(image-gt).abs().mean()+.2*(1-ssim(image,gt))
                    loss,stats=full_scene_step(model,batches,geometry,snr,'awgn',distortion,
                                              attr_weight=args.attr_weight,mode='replay',gradient_observer=probe)
                    joined=torch.cat(qs)
                    identity={'camera':camera.image_name,'tier_counts':torch.bincount(joined,minlength=4).tolist(),
                              'tier_sha256':hashlib.sha256(joined.cpu().numpy().tobytes()).hexdigest()}
                if not torch.isfinite(loss):
                    raise RuntimeError(f'{variant} step {step}: nonfinite loss')
                row={'step':step,'phase':'render' if is_render else 'attribute','loss':float(loss.detach()),
                     'snr':snr,'seed':args.seed+step,'learning_rate':optimizer.param_groups[0]['lr'],**identity,**stats}
                row.update(objective_stats(row,model,args.attr_weight if is_render else 1.))
                row['gradients_before']=gradient_stats(model)
                if probe:
                    row['component_gradients']=probe.report()
                    row['largest_parameter_gradients']=largest_gradients(model)
                    before={n:p.detach().clone() for n,p in model.named_parameters()}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_norm,error_if_nonfinite=True)
                row.update(grad_norm=float(norm),clip_factor=min(1.,args.clip_norm/(float(norm)+1e-6)),
                           gradients_after=gradient_stats(model))
                optimizer.step()
                if probe:
                    row['updates']=update_stats(model,before)
                    del before,probe
                if device.type=='cuda':
                    torch.cuda.synchronize(device)
                row['step_seconds']=time.perf_counter()-started
                log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush(); logs.append(row)
                if step%10==0 or step==1:
                    print(f'{variant} {step}/{total} {row["phase"]}: loss={row["loss"]:.4f}, '
                          f'grad={row["grad_norm"]:.2f}, clip={row["clip_factor"]:.5f}',flush=True)
                if step%args.eval_every==0 or step in (args.steps,total):
                    evals.extend(evaluate_blocks(model,blocks,geometry,args,step))
                    write_json(folder/'evaluation.json',evals)
                if step in (args.steps,total):
                    save_checkpoint(folder/f'codec_{step}.pt',model,step,
                                    {'diagnostic_variant':variant,'detach_attribute_context_xyz':variant=='detached',
                                     'initializer':args.checkpoint,'fresh_adam':True})
        save_checkpoint(folder/'codec.pt',model,total,{'diagnostic_variant':variant,
                         'detach_attribute_context_xyz':variant=='detached','initializer':args.checkpoint})
        test=evaluate_render(model,blocks,raw,geometry,test_cameras,degree,args,folder/'test') if test_cameras else None
        all_logs[variant]=logs; evaluations[variant]=evals
        phase_summary={}
        for phase in ('attribute','render'):
            selected=[r for r in logs if r['phase']==phase]
            if selected:
                norms=torch.tensor([r['grad_norm'] for r in selected],dtype=torch.float64)
                phase_summary[phase]={'steps':len(selected),'clip_fraction':sum(r['clip_factor']<1 for r in selected)/len(selected),
                                      'grad_norm_median':float(norms.median()),'grad_norm_p95':float(torch.quantile(norms,.95)),
                                      'grad_norm_max':float(norms.max())}
        summary[variant]={'clipped_steps':sum(r['clip_factor']<1 for r in logs),'steps':total,
                          'phase_gradient_summary':phase_summary,
                          'final_fixed_evaluation':[r for r in evals if r['step']==total],
                          'initial_test':baseline_render,'final_test':test}
        write_json(out/'summary.json',summary)
        del model,optimizer
    # Verify experimental schedule, including complete per-row q identities.
    for a,b in zip(all_logs['attached'],all_logs['detached']):
        for key in ('seed','snr','phase','tier_sha256','camera','block'):
            if a.get(key)!=b.get(key):
                raise RuntimeError(f'Paired schedule mismatch at {a["step"]}: {key}')
    for a,b in zip(evaluations['attached'],evaluations['detached']):
        if a['step']==0 and a!=b:
            # CUDA atomic reductions can differ slightly despite identical inputs.
            for key in ('position_rmse','reconstruction_loss'):
                if not math.isclose(a[key],b[key],rel_tol=1e-5,abs_tol=1e-6):
                    raise RuntimeError('Initial forward equality check failed')
    summary['paired_schedule_verified']=True
    write_json(out/'summary.json',summary)
    with (out/'fixed_evaluation.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['variant',*evaluations['attached'][0]])
        writer.writeheader()
        for variant,ev in evaluations.items():
            writer.writerows({'variant':variant,**r} for r in ev)
    if not args.no_plots:
        make_plots(out,all_logs,evaluations)
    print(f'Finished paired diagnostic: {out}. Original checkpoint unchanged.',flush=True)
