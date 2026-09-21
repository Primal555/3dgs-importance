"""Read-only q3/noiseless position error decomposition; no renderer required.

Oracle centering/affine alignment uses ground truth for diagnosis ONLY. It is
not a deployable receiver and must never be reported as communication quality.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from gaussian_jscc.data import read_ply, prepare, to_features
from gaussian_jscc.transport import load_checkpoint
from gaussian_jscc.checkpoint_history import file_hash


def decompose(predicted, target, radii):
    """Exact Euclidean SSE split into block-mean offset and centered error."""
    p,t=predicted.double(),target.double()
    delta=p-t
    mean=delta.mean(0)
    centered=delta-mean
    total=delta.square().sum()
    common=len(delta)*mean.square().sum()
    relative=centered.square().sum()
    pc=p-p.mean(0);tc=t-t.mean(0)
    # Diagnostic best affine map, not part of inference. Rank deficiency is OK.
    aligned=pc @ (torch.linalg.pinv(pc,rtol=1e-6) @ tc)
    affine_error=(aligned-tc).square().sum()
    return dict(points=len(p),sse=float(total),common_sse=float(common),relative_sse=float(relative),
                affine_oracle_sse=float(affine_error),
                xyz_rmse=float((total/(3*len(p))).sqrt()),
                common_fraction=float(common/total) if total>0 else 0.,
                centered_rmse=float((relative/(3*len(p))).sqrt()),
                affine_oracle_rmse=float((affine_error/(3*len(p))).sqrt()),
                distance_over_radius_p50=float((delta.norm(dim=-1)/radii.double().clamp_min(1e-12)).median()),
                centered_distance_over_radius_p50=float((centered.norm(dim=-1)/radii.double().clamp_min(1e-12)).median()),
                centered_spread_ratio=float((pc.square().sum()/tc.square().sum().clamp_min(1e-20)).sqrt()),
                mean_error=mean.tolist())


def aggregate(rows):
    n=sum(r['points'] for r in rows)
    totals={key:sum(r[key] for r in rows) for key in ('sse','common_sse','relative_sse','affine_oracle_sse')}
    result={'blocks':len(rows),'points':n,
            **{key.replace('sse','rmse'):math.sqrt(v/(3*n)) for key,v in totals.items()},
            'common_sse_fraction':totals['common_sse']/max(totals['sse'],1e-30)}
    mean=torch.tensor([r['mean_error'] for r in rows],dtype=torch.float64)
    counts=torch.tensor([r['points'] for r in rows],dtype=torch.float64)
    global_error=(mean*counts[:,None]).sum(0)/n
    result['global_mean_error_world']=global_error.tolist()
    result['global_translation_sse_fraction']=float(n*global_error.square().sum())/max(totals['sse'],1e-30)
    for key in ('distance_over_radius_p50','centered_distance_over_radius_p50','centered_spread_ratio'):
        result['mean_block_'+key]=sum(r[key] for r in rows)/len(rows)
    return result


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.cpu_threads)
    if args.max_blocks<0 or args.blocks_per_batch<1 or args.cpu_threads<1:
        raise ValueError('invalid block/thread counts')
    out=Path(args.out)
    if out.exists():
        raise FileExistsError('choose a new diagnosis output directory')
    before=file_hash(args.checkpoint)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    record=saved.get('training',{})
    model=load_checkpoint(args.checkpoint,args.device).eval()
    if model.cfg.context_mode!='multiscale_self' or model.cfg.position_delivery!='learned' or model.cfg.rates[-1]!=32:
        raise ValueError('expected multiscale_self learned-XYZ 32-symbol checkpoint')
    raw,degree=read_ply(args.ply)
    if degree!=model.cfg.sh_degree or record.get('source_gaussians',len(raw))!=len(raw):
        raise ValueError('checkpoint and PLY do not match degree/count')
    raw,geometry,_=prepare(raw,model.cfg.morton_bits)
    size=model.cfg.block_size;blocks=(len(raw)+size-1)//size
    held=set(record.get('bootstrap_validation_blocks',[]))
    selected=(list(range(blocks)) if args.max_blocks==0 else
              sorted(set(torch.linspace(0,blocks-1,min(blocks,args.max_blocks)).round().long().tolist())|held))
    if any(i<0 or i>=blocks for i in selected):
        raise ValueError('invalid held-out block indices')
    captures={}
    hooks=[model.learned.heads['xyz'].register_forward_hook(lambda m,i,o:captures.update(own=o)),
           model.learned.context_heads['xyz'].register_forward_hook(lambda m,i,o:captures.update(context=o))]
    rows=[]
    try:
        for offset in range(0,len(selected),args.blocks_per_batch):
            indices=selected[offset:offset+args.blocks_per_batch]
            sources=[raw[i*size:(i+1)*size] for i in indices]
            features=[to_features(r.to(args.device),geometry,model)[0] for r in sources]
            f=pad_sequence(features,batch_first=True)
            q=torch.zeros(f.shape[:2],dtype=torch.long,device=args.device)
            for j,r in enumerate(sources):
                q[j,:len(r)]=3
            z=model.learned.encode(f,f[...,:3],q,args.snr)
            decoded=model.learned.decode(z,q,args.snr) # identity channel
            own=captures['own'];correction=model.learned.dec_geometry_gate.tanh()*captures['context']
            torch.testing.assert_close(decoded[...,:3][q>0],(own+correction)[q>0],rtol=2e-5,atol=2e-6)
            for j,(index,source) in enumerate(zip(indices,sources)):
                n=len(source);radii=source[:,4:7].amax(-1).exp()
                for label,pred in (('full',decoded[j,:n,:3]),('self_only',own[j,:n])):
                    world=geometry.denormalize(pred).cpu()
                    row=decompose(world,source[:,:3],radii)
                    row.update(block=index,split='heldout' if index in held else 'training_pool',path=label)
                    corr=correction[j,:n].cpu()*geometry.span
                    row['context_correction_rmse_world']=float(corr.square().mean().sqrt())
                    rows.append(row)
            print(f'Diagnosed {min(offset+len(indices),len(selected))}/{len(selected)} blocks',flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    if file_hash(args.checkpoint)!=before:
        raise RuntimeError('checkpoint changed during diagnosis')
    summary={}
    for path in ('full','self_only'):
        for split in ('all','heldout','training_pool'):
            part=[r for r in rows if r['path']==path and (split=='all' or r['split']==split)]
            if part:
                summary[path+'/'+split]=aggregate(part)
    result=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=before,checkpoint_step=saved.get('step'),
                ply=str(args.ply),ply_sha256=file_hash(args.ply),
                channel='none',tier=3,symbols=32,snr_conditioning=args.snr,
                total_scene_blocks=blocks,selected_blocks=selected,
                selection='all blocks' if args.max_blocks==0 else 'spaced blocks plus all recorded heldout blocks',
                oracle_warning='centering and affine maps use source XYZ, diagnosis only; NOT a receiver or render PSNR',
                self_only_warning='remove only final XYZ Context correction; latent still contains encoder Context',
                aggregation='point-weighted SSE/RMSE; radius statistics are equal-block mean of medians',
                summary=summary)
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'blocks.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('ply','checkpoint','out'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--max-blocks',type=int,default=128,help='0 = full scene; otherwise spaced sample plus heldout blocks')
    parser.add_argument('--blocks-per-batch',type=int,default=4)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--snr',type=float,default=10.,help='conditioning only, no channel noise')
    run(parser.parse_args())


if __name__=='__main__':
    main()
