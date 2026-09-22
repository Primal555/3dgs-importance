"""Matched real-PLY CPU/GPU check: b256 baseline, b512 baseline, b512 refinement.

Same 512-point fit/heldout regions, sampled points, q3/none, loss and LR.
No rendering: these metrics are NOT PSNR or proof of perceptual improvement.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import read_ply, prepare, to_features, fit_feature_statistics
from gaussian_jscc.spatial_response import spatial_response_loss
from gaussian_jscc.optimization import clip_codec_gradients
from gaussian_jscc.transport import save_checkpoint
from gaussian_jscc.checkpoint_history import file_hash
from scripts.compare_received_memory import append, position_metrics


def run(args):
    if args.regions < 8 or args.regions % 4 or min(args.steps,args.every,args.batch_regions,args.threads)<1:
        raise ValueError('regions >=8 divisible by 4 and positive counts required')
    if args.batch_regions > args.regions*3//4:
        raise ValueError('batch exceeds fit regions')
    torch.set_num_threads(args.threads)
    root=Path(args.out)
    if root.exists():raise FileExistsError('choose a new output directory')
    raw,degree=read_ply(args.ply);raw,geometry,_=prepare(raw,16)
    total=len(raw)//512
    if total<args.regions:raise ValueError('not enough complete 512-point regions')
    indices=[int((i+.5)*total/args.regions) for i in range(args.regions)]
    splits={'fit':[i for i in range(args.regions) if i%4!=3],
            'heldout':[i for i in range(args.regions) if i%4==3]}
    cfg=CodecConfig(architecture='learned_split_logcov',context_mode='multiscale_self',
                    encoder_attention='geometric_point',decoder_attention='transformer_trunk',sh_degree=degree)
    template=GaussianCodec(cfg);fit_feature_statistics(raw,template)
    features=torch.stack([to_features(raw[i*512:(i+1)*512],geometry,template)[0] for i in indices]).to(args.device)
    root.mkdir(parents=True)
    protocol=dict(vars(args),region_size=512,region_indices=indices,splits=splits,
                  ply_sha256=file_hash(args.ply),source_gaussians=len(raw),base_config=cfg.to_dict(),
                  scope='same-scene matched fit/heldout points; random weights; q3/none; constant LR 2e-4; no clipping; no rendering')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    del raw
    summaries=[]
    for case,size,mode in [('baseline256',256,'none'),('baseline512',512,'none'),('progressive512',512,'progressive')]:
        torch.manual_seed(args.seed)
        config=dict(cfg.to_dict(),block_size=size,decoder_refinement=mode)
        model=GaussianCodec(CodecConfig.from_dict(config)).to(args.device)
        model.attr_mean.copy_(template.attr_mean);model.attr_std.copy_(template.attr_std)
        optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
        sampler=torch.Generator().manual_seed(args.seed+10000)
        folder=root/case;folder.mkdir()
        captured=[];hooks=[]
        if mode=='progressive':
            trunk=model.learned.dec_trunk
            hooks.append(trunk.initial_xyz.register_forward_hook(lambda m,a,y:captured.append(y)))
            hooks.extend(r.register_forward_hook(lambda m,a,y:captured.append(y[1])) for r in trunk.refiners)

        @torch.no_grad()
        def evaluate(step):
            model.eval();result={'step':step}
            for split,selection in splits.items():
                rows=[]
                for index in selection:
                    f=features[index].reshape(-1,size,features.shape[-1]);q=torch.full(f.shape[:2],3,device=f.device,dtype=torch.long)
                    captured.clear()
                    pred=model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).float(),10,'none')[0]
                    p,t=pred.flatten(0,1),f.flatten(0,1)
                    loss,stats=spatial_response_loss(p,t,geometry,model,directions=torch.eye(3,device=f.device))
                    row=dict(position_metrics(p[:,:3],t[:,:3],geometry.span.to(f)),loss=float(loss),
                             position=stats['spatial_position_response'],shape=stats['spatial_logcov_shape_mse'])
                    for depth,xyz in enumerate(captured):
                        row[f'stage{depth}_xyz_mse']=float(((xyz.flatten(0,1)-t[:,:3]).double()*geometry.span.to(f).double()).square().mean())
                    rows.append(row)
                    append(folder/'blocks.jsonl',dict(step=step,split=split,region=indices[index],**row))
                result[split]={k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
            append(folder/'validation.jsonl',result)
            print(case,step,{s:round(result[s]['mse']**.5,5) for s in splits},flush=True)
            model.train()
            return result

        evaluate(0);start=time.perf_counter()
        for step in range(1,args.steps+1):
            chosen=torch.randperm(len(splits['fit']),generator=sampler)[:args.batch_regions]
            f=features[[splits['fit'][int(i)] for i in chosen]].reshape(-1,size,features.shape[-1])
            directions=torch.randn(4,3,generator=sampler).to(args.device)
            q=torch.full(f.shape[:2],3,device=f.device,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True);captured.clear()
            pred=model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).float(),10,'none')[0]
            loss,_=spatial_response_loss(pred.flatten(0,1),f.flatten(0,1),geometry,model,directions=directions)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();norm,groups=clip_codec_gradients(model,mode='none')
            if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
            optimizer.step();captured.clear()
            append(folder/'loss.jsonl',dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),**groups))
            if step%args.every==0 or step==args.steps:last=evaluate(step)
        for hook in hooks:hook.remove()
        save_checkpoint(folder/'codec.pt',model,args.steps,{'experiment_protocol':str(root/'protocol.json')})
        summaries.append(dict(case=case,seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),**last))
        (root/'comparison.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    return summaries


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ply',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--every',type=int,default=100)
    p.add_argument('--regions',type=int,default=64);p.add_argument('--batch-regions',type=int,default=2)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',default='cpu')
    return p


if __name__=='__main__':run(build_parser().parse_args())
