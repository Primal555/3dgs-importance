"""Local-only v4 error attribution and bounded real-scene optimization.

Never changes production modules/checkpoints. Oracle cases are attribution,
not deployable codecs. Run from the repo; outputs go to a NEW directory.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_jscc.block_geometry import BlockGeometry, position_objective
from gaussian_jscc.codec import channel
from gaussian_jscc.data import read_ply, prepare
from gaussian_jscc.transport import load_checkpoint


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def parts(model, received, q):
    _, mask, gain = model.layout(q, received.dtype)
    data = received * mask / gain.clamp_min(1e-8)
    pooled = model.average(data, (q > 0).float())
    center = (pooled[..., :3] + 1) / 2
    radius = ((pooled[..., 3:4].clamp(-1, 1) - 1) * -math.log(model.radius_floor) / 2).exp()
    offset = []
    for axis in range(3):
        select = (model.offset_axis == axis).float() * mask
        offset.append((data * select).sum(-1) / select.sum(-1).clamp_min(1))
    return center, radius, torch.stack(offset, -1)


@torch.no_grad()
def decomposition(model, blocks, span, indices, tier, seed):
    errors = {k: [] for k in ('full', 'center', 'scale', 'offset', 'interaction',
                              'oracle_center', 'oracle_scale', 'oracle_reference', 'oracle_offset')}
    per_block = []
    max_reconciliation = 0.
    for index in indices:
        xyz = blocks[index]
        q = torch.full((len(xyz),), tier)
        center, radius = model.reference(xyz, torch.ones(len(xyz)))
        offset = (xyz - center) / radius
        torch.manual_seed(seed + index)
        sent = model.encode(xyz, q, 10.)
        received = channel(sent.reshape(-1, 2), 10., 'awgn').reshape_as(sent)
        c, s, r = parts(model, received, q)
        actual = model.decode(received, q, 10.)
        assert torch.allclose(actual, c+s*r, atol=2e-6, rtol=2e-6), 'Decomposition requires zero learned decoder correction'
        ec, es, er = c-center, s-radius, r-offset
        terms = dict(full=actual-xyz, center=ec.expand_as(xyz), scale=es*offset,
                     offset=radius*er, interaction=es*er,
                     oracle_center=center+s*r-xyz, oracle_scale=c+radius*r-xyz,
                     oracle_reference=center+radius*r-xyz, oracle_offset=c+s*offset-xyz)
        max_reconciliation = max(max_reconciliation, float((terms['full']-sum(terms[k] for k in ('center','scale','offset','interaction'))).abs().max()))
        for name, delta in terms.items():
            errors[name].append(delta * span)
        per_block.append(dict(block=index, n=len(xyz), radius=float(radius),
                              mse=float(((actual-xyz)*span).square().mean()),
                              ref_energy=float(sent[:,:4].square().sum(-1).mean()),
                              filler_energy=float(sent[:,7].square().mean()),
                              total_geometry_energy=int(model.rates[tier])))
    merged = {k: torch.cat(v) for k,v in errors.items()}
    mse = {k: float(v.square().mean()) for k,v in merged.items()}
    cross = {}
    names = ('center','scale','offset','interaction')
    for i,a in enumerate(names):
        for b in names[i+1:]:
            cross[a+'__'+b] = float((2*merged[a]*merged[b]).mean())
    return dict(tier=tier, seed=seed, n=len(merged['full']), mse=mse,
                rmse={k: math.sqrt(v) for k,v in mse.items()}, cross_mse=cross,
                additive_mse_residual=mse['full']-sum(mse[k] for k in names)-sum(cross.values()),
                max_coordinate_reconciliation=max_reconciliation,
                full_distance_p95=float(torch.quantile(merged['full'].norm(dim=-1),.95)), blocks=per_block)


@torch.no_grad()
def radius_survey(xyz, span, model, sizes):
    """Source-only expected error terms; isolated scale-noise interaction omitted.

    Offset variance follows exactly from AWGN sigma^2/gain^2 and repetition.
    Center variance follows averaging N repeated centers. No oracle deployment.
    """
    rows, sizes_summary = [], []
    for size in sizes:
        records = []
        for start in range(0,len(xyz),size):
            x = xyz[start:start+size]
            c,s = model.reference(x,torch.ones(len(x)))
            dist = (x-c).norm(dim=-1)
            r50 = float(dist.median())
            records.append(dict(block=start//size,n=len(x),radius=float(s),
                                median_distance=r50, radius_to_median=float(s)/max(r50,1e-12)))
        for tier in (1,2,3):
            _, mask, gain = model.layout(torch.tensor([tier]),torch.float32)
            repeats = torch.stack([(mask * (model.offset_axis==j)).sum() for j in range(3)])
            noise_var = .05 / float(gain.square())
            local_factor = float((span.square()/repeats).mean())*noise_var
            center_factor = float(span.square().mean())*noise_var/4
            total_offset = sum(r['n']*r['radius']**2*local_factor for r in records)/len(xyz)
            total_center = sum(center_factor for r in records)/len(xyz)
            ordered = sorted(records,key=lambda r:r['n']*r['radius']**2,reverse=True)
            denom = sum(r['n']*r['radius']**2 for r in records)
            sizes_summary.append(dict(block_size=size,tier=tier,n=len(xyz),blocks=len(records),
                expected_offset_rmse=math.sqrt(total_offset), expected_center_rmse=math.sqrt(total_center),
                expected_center_offset_rmse=math.sqrt(total_center+total_offset),
                top5_blocks_offset_mse_share=sum(r['n']*r['radius']**2 for r in ordered[:5])/denom,
                top5_percent_blocks_offset_mse_share=sum(r['n']*r['radius']**2 for r in ordered[:max(1,math.ceil(.05*len(ordered)))])/denom))
        rows.extend(dict(block_size=size,**r) for r in records)
    return rows, sizes_summary


@torch.no_grad()
def pilot_control(model, blocks, span, indices, tier, seed):
    """Diagnostic candidate: replace discarded filler with a known pilot.

    SAME number of complex symbols, SAME mean geometry power, SAME AWGN.
    Receiver estimates normalization from received pilot, NOT source energy.
    Not integrated in transport or trained; small/fading packets unvalidated.
    """
    deltas=[];power=[];gain_errors=[];signal_energy=[]
    for index in indices:
        x=blocks[index];q=torch.full((len(x),),tier)
        sent=model.encode(x,q,10.)
        _,mask,gain=model.layout(q,sent.dtype)
        base=sent*mask/gain
        base[:,7]=.25
        sender_gain=(float(model.rates[tier])/base.square().sum(-1).mean()).sqrt()
        z=base*sender_gain
        torch.manual_seed(seed+index)
        received=channel(z.reshape(-1,2),10.,'awgn').reshape_as(z)
        receiver_gain=(received[:,7].mean()/.25).clamp_min(1e-3)
        decoded_data=received/receiver_gain
        # Reuse algebra via a re-scaled symbol vector; it cannot access clean c/s.
        c,s,r=parts(model,decoded_data*gain,q)
        y=c+s*r
        deltas.append((y-x)*span)
        power.append(float(z.square().sum(-1).mean())/float(model.rates[tier]))
        gain_errors.append(float(receiver_gain/sender_gain-1))
        signal_energy.append(dict(n=len(x),pilot=float(z[:,7].square().mean()),
                                  reference=float(z[:,:4].square().sum(-1).mean()),
                                  offset=float((z.square()*mask*(model.offset_axis>=0)).sum(-1).mean())))
    delta=torch.cat(deltas)
    return dict(tier=tier,seed=seed,n=len(delta),rmse=float(delta.square().mean().sqrt()),
                p95=float(torch.quantile(delta.norm(dim=-1),.95)),
                maximum_power_deviation=max(abs(v-1) for v in power),
                max_abs_relative_gain_error=max(abs(v) for v in gain_errors),energy=signal_energy)


@torch.no_grad()
def evaluate(model, blocks, span, indices, seeds):
    rows=[]
    for tier in (1,2,3):
        sums=[]; losses=[]; distances=[]
        for seed in seeds:
            for index in indices:
                xyz=blocks[index];q=torch.full((len(xyz),),tier)
                torch.manual_seed(seed+index)
                z=model.encode(xyz,q,10.)
                y=model.decode(channel(z.reshape(-1,2),10.,'awgn').reshape_as(z),q,10.)
                delta=(y-xyz)*span
                sums.append(delta.square().sum());distances.append(delta.norm(dim=-1))
                losses.append(float(position_objective(y,xyz)[0]))
        n=sum(len(d) for d in distances)
        rows.append(dict(tier=tier,rmse=math.sqrt(float(sum(sums)/(3*n))),
                         p95=float(torch.quantile(torch.cat(distances),.95)),local_objective=sum(losses)/len(losses)))
    return rows


def train_short(initial, blocks, span, indices, steps, profile, out):
    model=copy.deepcopy(initial)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    logs=[]; evaluations=[]
    def validate(step):
        rows=evaluate(model,blocks,span,indices,[100042,200042])
        evaluations.append(dict(step=step,metrics=rows))
        print(profile,step,[(r['tier'],round(r['rmse'],4)) for r in rows],flush=True)
    validate(0)
    started=time.perf_counter()
    for step in range(steps):
        torch.manual_seed(420000+step)
        index=int(torch.randint(len(blocks),(1,)))
        x=blocks[index]; optimizer.zero_grad(set_to_none=True)
        total=0.
        for tier in (1,2,3):
            q=torch.full((len(x),),tier)
            z=model.encode(x,q,10.)
            y=model.decode(channel(z.reshape(-1,2),10.,'awgn').reshape_as(z),q,10.)
            if profile=='current_local':
                loss,_=position_objective(y,x)
            else:
                loss=100*((y-x)*span/span.norm()).square().mean()
            (loss/3).backward();total+=float(loss.detach())/3
        groups={}
        for name,sub in [('encoder',model.encoder),('decoder',model.decoder)]:
            norm=torch.nn.utils.clip_grad_norm_(sub.parameters(),1.,error_if_nonfinite=True)
            groups[name]=float(norm)
        optimizer.step()
        logs.append(dict(step=step+1,block=index,loss=total,gradient_norms=groups))
        if (step+1)%100==0 or step+1==steps:
            validate(step+1)
            write(out/f'{profile}.json',dict(logs=logs,evaluations=evaluations,seconds=time.perf_counter()-started))
    torch.save(model.state_dict(),out/f'{profile}_geometry_only.pt')
    return dict(logs=logs,evaluations=evaluations,seconds=time.perf_counter()-started)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ply',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=300)
    p.add_argument('--pilot-only',action='store_true')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    codec=load_checkpoint(args.checkpoint,'cpu');codec.enable_block_geometry();model=codec.block_geometry
    raw,degree=read_ply(args.ply);raw,geometry,_=prepare(raw,codec.cfg.morton_bits)
    xyz=geometry.normalize(raw[:,:3]);span=geometry.span
    blocks=list(xyz.split(codec.cfg.block_size));indices=torch.linspace(0,len(blocks)-1,8).round().long().tolist()
    meta=dict(ply=args.ply,checkpoint=args.checkpoint,source_sha256=digest(args.ply),
              checkpoint_sha256=digest(args.checkpoint),code_sha256=digest('gaussian_jscc/block_geometry.py'),
              source_count=len(raw),span=span.tolist(),block_size=codec.cfg.block_size,indices=indices,
              snr=10,noise_variance_per_real=.05,steps=args.steps,torch_version=torch.__version__,
              note='diagnostic only; source-scene blocks, no held-out scene or renderer quality claim')
    write(out/'provenance.json',meta)
    if args.pilot_only:
        rows=[pilot_control(model,blocks,span,indices,tier,seed)
              for seed in (100042,200042,300042) for tier in (1,2,3)]
        whole=pilot_control(model,blocks,span,list(range(len(blocks))),1,100042)
        write(out/'pilot_control.json',dict(fixed_blocks=rows,whole_scene_q1=whole))
        print('pilot fixed',[(r['tier'],r['seed'],r['rmse']) for r in rows],flush=True)
        print('pilot whole q1', {k:v for k,v in whole.items() if k!='energy'},flush=True)
        return
    rows=[]
    for seed in (100042,200042,300042):
        for tier in (1,2,3):
            row=decomposition(model,blocks,span,indices,tier,seed);rows.append(row)
            print('decompose',tier,seed,{k:round(row['rmse'][k],4) for k in ('full','center','scale','offset','oracle_reference')},flush=True)
    # Whole-scene q1 confirms whether the eight-block diagnostic overstates error.
    whole=decomposition(model,blocks,span,list(range(len(blocks))),1,100042)
    write(out/'decomposition.json',dict(fixed_blocks=rows,whole_scene_q1=whole))
    survey,sensitivity=radius_survey(xyz,span,model,(4096,1024,256,64))
    write(out/'radius_survey.json',survey);write(out/'block_size_sensitivity.json',sensitivity)
    for r in sensitivity:
        if r['tier']==1: print('block sensitivity',r,flush=True)
    for profile in ('current_local','scene_mse'):
        train_short(model,blocks,span,indices,args.steps,profile,out)


if __name__=='__main__':
    main()
