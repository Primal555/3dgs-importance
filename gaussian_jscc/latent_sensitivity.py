"""Frozen decoder sensitivity; no training, channel invocation or tier changes."""
import argparse
import csv
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from .data import read_ply, to_features, to_raw, Geometry, morton_order
from .local_response import footprint, response_terms, stencil, view_frames
from .position_delivery import delivered_positions
from utils.sh_utils import eval_sh


def relative_noise(y, level, seed, point=None):
    """Exact per-point RMS(delta)/RMS(y)=level; zero-signal rows stay zero.

    A private CPU RNG gives paired normalized noise directions across variants,
    devices and levels, without changing the caller's RNG. Learned latent bases
    need not align, so this is an isotropic stress test, not matched semantics.
    """
    if y.ndim!=2 or not len(y) or not y.shape[-1] or not torch.isfinite(y).all():
        raise ValueError('finite nonempty [points,latent] required')
    if not math.isfinite(level) or level<0 or (point is not None and not 0<=point<len(y)):
        raise ValueError('invalid perturbation level or point index')
    noise=torch.randn(y.shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float32).to(y)
    noise=noise/noise.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-20)
    delta=noise*y.detach().square().mean(-1,keepdim=True).sqrt()*level
    if point is not None:
        keep=torch.zeros(len(y),dtype=torch.bool,device=y.device); keep[point]=True
        delta=delta.masked_fill(~keep[:,None],0)
    return delta


def color_values(features, model, directions):
    """Physical unclipped RGB, before opacity or footprint multiplication."""
    coefficients=features[:,11:]*model.attr_std[8:]+model.attr_mean[8:]
    count=(model.cfg.sh_degree+1)**2
    sh=torch.cat((coefficients[:,:3,None],coefficients[:,3:].reshape(len(features),3,count-1)),-1)
    rgb=eval_sh(model.cfg.sh_degree,sh[:,None],directions[None])+.5
    return rgb.expand(len(features),len(directions),3)


def source_grid_response(features, source, geometry, model, directions):
    """Isolated black/white response on a common SOURCE footprint stencil.

    This is not a full perspective/occlusion-aware scene rendering. Both arms
    use the same teacher's sample grid, not their own predicted footprint grids.
    """
    reference=to_raw(source,geometry,model); predicted=to_raw(features,geometry,model)
    d,frames=view_frames(directions)
    ref_chol,ref_radius=footprint(reference,frames)
    probes=torch.einsum('nvij,pj->nvpi',ref_chol,stencil(source))
    chol,radius=footprint(predicted,frames)
    black,white=response_terms(predicted,d,chol,radius,ref_radius,probes,model.cfg.sh_degree)
    return torch.stack((black,white),-2)


def point_errors(observed, reference):
    return (observed-reference).square().flatten(1).mean(1)


@torch.no_grad()
def audit_model(model, raw, geometry, *, block_index, levels, trials, directions, seed, sparse_level):
    if not len(raw) or trials<1 or directions<1: raise ValueError('nonempty block and positive trials/directions required')
    device=next(model.parameters()).device; mode=model.training; rows=[]; spread=[]
    model.eval(); model.set_scene_geometry(geometry)
    try:
        f,_=to_features(raw.to(device),geometry,model)
        active=torch.ones((1,len(f)),dtype=torch.bool,device=device)
        # Any positive value denotes presence here; no symbol prefix is selected.
        q=active.long(); side=delivered_positions(f[:,:3],q[0],model.cfg)[None]
        y=model.encode_representation(f[None],f[None,:,:3],active)[0]
        def decode(value): return model.decode_representation(value[None],q,side)[0]
        clean=decode(y)
        dirs=F.normalize(torch.randn(directions,3,generator=torch.Generator().manual_seed(seed+7919)),dim=-1).to(f)
        clean_color=color_values(clean,model,dirs); target_color=color_values(f,model,dirs)
        clean_response=source_grid_response(clean,f,geometry,model,dirs)
        target_response=source_grid_response(f,f,geometry,model,dirs)
        clean_color_error=point_errors(clean_color,target_color)
        clean_response_error=point_errors(clean_response,target_response)
        power=float(y.double().square().sum())
        for trial in range(trials):
            noise_seed=seed+100003*block_index+1009*trial
            for level in levels:
                delta=relative_noise(y,level,noise_seed); predicted=decode(y+delta)
                color=color_values(predicted,model,dirs)
                response=source_grid_response(predicted,f,geometry,model,dirs)
                errors={'color':point_errors(color,clean_color),'response':point_errors(response,clean_response),
                        'source_color':point_errors(color,target_color),'source_response':point_errors(response,target_response)}
                row={'block':block_index,'trial':trial,'level':level,'points':len(f),
                     'input_nmse':float(delta.double().square().sum())/max(power,1e-30),
                     'latent_power':power/y.numel(),
                     'clean_source_color_mse':float(clean_color_error.mean()),
                     'clean_source_response_mse':float(clean_response_error.mean())}
                for name,value in errors.items():
                    if not torch.isfinite(value).all(): raise RuntimeError('nonfinite decoder output errors')
                    row[name+'_mse']=float(value.mean()); row['_'+name+'_errors']=value.cpu()
                rows.append(row)
            point=int(torch.randint(len(f),(1,),generator=torch.Generator().manual_seed(noise_seed+31)))
            delta=relative_noise(y,sparse_level,noise_seed,point=point); predicted=decode(y+delta)
            color_errors=point_errors(color_values(predicted,model,dirs),clean_color)
            response_errors=point_errors(source_grid_response(predicted,f,geometry,model,dirs),clean_response)
            others=torch.arange(len(f),device=device)!=point
            color_energy=float(color_errors.sum()); response_energy=float(response_errors.sum())
            spread.append({'block':block_index,'trial':trial,'target_slot':point,'points':len(f),'level':sparse_level,
                'target_color_mse':float(color_errors[point]),
                'other_color_mse':float(color_errors[others].mean()) if others.any() else 0.,
                'color_spill_fraction':float(color_errors[others].sum())/max(color_energy,1e-30),
                'response_spill_fraction':float(response_errors[others].sum())/max(response_energy,1e-30),
                'other_points_above_rgb_rmse_1e4':int((color_errors[others].sqrt()>1e-4).sum()),
                'other_points':int(others.sum())})
    finally:
        model.train(mode)
    return rows,spread


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='completed four-arm PointNet context suite directory')
    p.add_argument('--out',required=True,help='NEW output directory; input weights are never overwritten')
    p.add_argument('--ply',help='override source path after migration; exact fingerprint must match')
    p.add_argument('--device',default='cpu')
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--stages',nargs='+',choices=['semantic_render','channel_render'],default=['semantic_render'])
    p.add_argument('--blocks',type=int,default=32)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--directions',type=int,default=8)
    p.add_argument('--levels',nargs='+',type=float,default=[0.,.05,.1,.2,.35,.5])
    p.add_argument('--sparse-level',type=float,default=.2)
    p.add_argument('--seed',type=int,default=42)
    return p


def write_csv(path,rows):
    if not rows: return
    fields=[key for key in rows[0] if not key.startswith('_')]
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        writer.writerows({k:row[k] for k in fields} for row in rows)


def main(args):
    from .allocation import scene_fingerprint
    from .cli import device_for
    from .transport import load_checkpoint
    from .modular_experiment import json_write
    if (min(args.blocks,args.trials,args.directions,args.cpu_threads)<1 or not args.levels
            or len(set(args.levels))!=len(args.levels) or len(set(args.stages))!=len(args.stages)
            or any(not math.isfinite(v) or v<0 for v in args.levels)
            or not math.isfinite(args.sparse_level) or args.sparse_level<0):
        raise ValueError('invalid audit sampling or perturbation settings')
    if 0. not in args.levels: raise ValueError('levels must include zero-perturbation control')
    out=Path(args.out).resolve(); run=Path(args.run).resolve()
    if out.exists(): raise FileExistsError('use a NEW diagnostic output directory')
    record=json.loads((run/'baseline/training.json').read_text(encoding='utf-8'))
    expected=record['scene']
    specs=[s for s in record['selected_scenes'] if s['name']=='truck']
    if len(specs)!=1: raise ValueError('requires one Truck source scene')
    ply=Path(args.ply or specs[0]['ply'])
    raw,degree=read_ply(ply)
    if len(raw)!=expected['source_gaussians'] or scene_fingerprint(raw)!=expected['fingerprint'] or degree!=expected['sh_degree']:
        raise ValueError('PLY differs from trained scene; refusing substitute input')
    geometry=Geometry(**expected['geometry'])
    device=device_for(args.device); torch.set_num_threads(args.cpu_threads)
    jobs=[]; signature=None; first_mean=first_std=None
    for stage in args.stages:
        for group in ('baseline','both'):
            path=run/group/'checkpoints'/('end_'+stage)/'codec.pt'
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved.get('training',{}).get('scene_record')!=expected:
                raise ValueError('checkpoint scene identity/split mismatch')
            model=load_checkpoint(path,device)
            if model.cfg.context_variant!=group or model.cfg.sh_degree!=degree:
                raise ValueError('checkpoint architecture/group mismatch')
            cfg={k:v for k,v in model.cfg.to_dict().items() if not k.startswith('context_')}
            if signature is not None and cfg!=signature: raise ValueError('non-context codec configuration differs')
            signature=cfg
            if first_mean is not None and (not torch.equal(model.attr_mean.cpu(),first_mean) or not torch.equal(model.attr_std.cpu(),first_std)):
                raise ValueError('attribute normalization differs across controls')
            first_mean=model.attr_mean.cpu().clone(); first_std=model.attr_std.cpu().clone()
            jobs.append((stage,group,path,model))
    block_size=signature['block_size']
    order=torch.from_numpy(morton_order(geometry.quantize(raw[:,:3]).numpy()).astype('int64'))
    raw=raw[order]
    count=math.ceil(len(raw)/block_size)
    selected=torch.linspace(0,count-1,min(args.blocks,count)).round().long().unique().tolist()
    out.mkdir(parents=True)
    json_write(out/'plan.json',{'source_run':str(run),'ply':str(ply),'fingerprint':expected['fingerprint'],
        'stages':args.stages,'blocks':selected,'source_gaussians':len(raw),'block_size':block_size,
        'levels':args.levels,'trials':args.trials,'directions':args.directions,'seed':args.seed,
        'relative_noise':'exact per-point RMS(delta)/RMS(y); isotropic, private paired CPU RNG; zero latent stays zero',
        'response_grid':'same original-source Gaussian footprint; 25 probes; black/white background',
        'sparse_level':args.sparse_level,'sparse_threshold_rgb_rmse':1e-4,
        'scope':'sampled blocks; no full scene rendering, no tier allocation/training, no channel, no optimizer',
        'checkpoints':[str(path) for _,_,path,_ in jobs]})
    rows=[]; spread=[]
    for stage,group,path,model in jobs:
        before={n:p.detach().clone() for n,p in model.state_dict().items()}
        for position,index in enumerate(selected):
            block=raw[index*block_size:(index+1)*block_size]
            part,local=audit_model(model,block,geometry,block_index=index,levels=args.levels,
                trials=args.trials,directions=args.directions,seed=args.seed,sparse_level=args.sparse_level)
            rows.extend(dict(r,stage=stage,group=group) for r in part)
            spread.extend(dict(r,stage=stage,group=group,target_source_id=int(order[index*block_size+r['target_slot']])) for r in local)
            print(f'{stage}/{group}: block {position+1}/{len(selected)}',flush=True)
        if any(not torch.equal(p,before[n]) for n,p in model.state_dict().items()):
            raise RuntimeError('diagnostic changed model weights/statistics')
    curves=[]
    for stage,group,_,_ in jobs:
        for level in sorted(args.levels):
            chosen=[r for r in rows if r['stage']==stage and r['group']==group and r['level']==level]
            total=sum(r['points'] for r in chosen)
            summary={'stage':stage,'group':group,'level':level,'point_trials':total,
                'input_nmse':sum(r['input_nmse']*r['latent_power']*r['points'] for r in chosen)/max(sum(r['latent_power']*r['points'] for r in chosen),1e-30),
                'clean_source_color_mse':sum(r['clean_source_color_mse']*r['points'] for r in chosen)/total,
                'clean_source_response_mse':sum(r['clean_source_response_mse']*r['points'] for r in chosen)/total}
            for name in ('color','response','source_color','source_response'):
                values=torch.cat([r['_'+name+'_errors'] for r in chosen])
                summary[name+'_mse']=float(values.mean())
                summary[name+'_point_p95']=float(values.quantile(.95))
            summary['color_rmse_per_relative_rms']=math.sqrt(summary['color_mse'])/level if level else None
            summary['response_rmse_per_relative_rms']=math.sqrt(summary['response_mse'])/level if level else None
            curves.append(summary)
    json_write(out/'summary.json',{'curves':curves,'weights_unchanged':True,'channel_invoked':False,'optimizer_updates':0,
        'caveats':['Isotropic latent noise is not the learned channel residual distribution.',
                   'Local response lacks scene occlusion/perspective/pixel low-pass; not scene PSNR.',
                   'Different latent bases; compare stress response, not semantics of coordinate-wise perturbations.',
                   'Block sampling is diagnostic, not a full-scene importance-weighted quality measure.']})
    write_csv(out/'curves.csv',curves); write_csv(out/'per_block.csv',rows); write_csv(out/'spread.csv',spread)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for stage,group,_,_ in jobs:
        selected_rows=[r for r in curves if r['stage']==stage and r['group']==group]
        for ax,key in zip(axes,('color_mse','response_mse')):
            ax.plot([r['level'] for r in selected_rows],[r[key] for r in selected_rows],marker='o',
                    color='#277da8' if group=='baseline' else '#cf7d19',
                    linestyle='-' if stage=='semantic_render' else '--',label=stage+'/'+group)
            ax.set(xlabel='Per-point relative latent RMS perturbation',ylabel=key,
                   yscale='symlog',title='Output change vs own clean decoder'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/'sensitivity.png',dpi=150); plt.close(fig)
    json_write(out/'complete.json',{'status':'complete','groups':['baseline','both'],'stages':args.stages,
        'sampled_blocks':len(selected),'sampled_gaussians':sum(len(raw[i*block_size:(i+1)*block_size]) for i in selected),
        'weights_unchanged':True,'channel_invoked':False})
    print(f'Completed frozen decoder audit: {out}',flush=True)


if __name__=='__main__': main(parser().parse_args())
