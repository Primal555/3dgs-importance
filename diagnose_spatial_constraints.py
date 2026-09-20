"""CPU/CUDA small regression on real checkpoint: no renders or checkpoint writes.

Evaluate source-scale counterfactuals, optionally run a short in-memory codec
optimization. This is NOT the random-start training launcher and is not an
image-quality benchmark. Sampled complete blocks preserve decoder context.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from gaussian_jscc.data import read_ply, prepare, to_features
from gaussian_jscc.transport import load_checkpoint
from gaussian_jscc.spatial_response import spatial_response_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--ply',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=0)
    p.add_argument('--sample-blocks',type=int,default=16)
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--fine-weight',type=float,default=1.,help='0: v2 ablation; 1: v3 fine precision')
    p.add_argument('--lr',type=float,default=2e-4)
    args=p.parse_args()
    if args.steps<0 or args.sample_blocks<4:
        p.error('steps >= 0 and sample-blocks >= 4 required')
    if not math.isfinite(args.lr) or args.lr<=0 or not math.isfinite(args.fine_weight) or args.fine_weight<0:
        p.error('positive finite lr and nonnegative finite fine-weight required')
    out=Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    device=torch.device(args.device)
    path=Path(args.checkpoint)
    digest=lambda:hashlib.sha256(path.read_bytes()).hexdigest()
    original_hash=digest()
    model=load_checkpoint(path,device)
    if model.cfg.position_delivery!='learned':
        p.error('learned-XYZ checkpoint required')
    raw,degree=read_ply(args.ply)
    if degree!=model.cfg.sh_degree:
        p.error('SH degree mismatch')
    raw,g,_=prepare(raw,model.cfg.morton_bits)
    count=len(raw)//model.cfg.block_size
    if count<args.sample_blocks:
        p.error('not enough complete source blocks')
    indices=torch.linspace(0,count-1,args.sample_blocks).long()
    source=torch.stack([raw[i*model.cfg.block_size:(i+1)*model.cfg.block_size] for i in indices]).to(device)
    features,_=to_features(source.flatten(0,1),g,model)
    features=features.reshape(args.sample_blocks,model.cfg.block_size,-1)
    train,heldout=features[::2],features[1::2]
    directions=torch.eye(3,device=device)

    def decode(f,q):
        return model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).to(f),10.,'awgn')[0]

    @torch.no_grad()
    def evaluate():
        model.eval()
        results={}
        for tier in (1,2,3):
            torch.manual_seed(args.seed+100+tier)
            if device.type=='cuda':torch.cuda.manual_seed_all(args.seed+100+tier)
            pred=decode(heldout,torch.full(heldout.shape[:2],tier,device=device)).flatten(0,1)
            target=heldout.flatten(0,1)
            variants={}
            for name,xyz,shape in [('decoded',False,False),('source_scale',False,True),
                                   ('source_xyz',True,False),('source_xyz_scale',True,True)]:
                v=pred.clone()
                if xyz:v[:,:3]=target[:,:3]
                if shape:v[:,4:7]=target[:,4:7]
                loss,stats=spatial_response_loss(v,target,g,model,directions=directions,fine_weight=args.fine_weight)
                variants[name]={'loss':float(loss),**stats}
            results[str(tier)]=variants
        return results

    before=evaluate()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    trace=[]
    for step in range(args.steps):
        model.train()
        torch.manual_seed(args.seed+1000+step)
        if device.type=='cuda':torch.cuda.manual_seed_all(args.seed+1000+step)
        chosen=torch.randint(len(train),(min(4,len(train)),),device=device)
        f=train[chosen]
        tier=step%4+1
        q=(torch.randint(1,4,f.shape[:2],device=device) if tier==4 else
           torch.full(f.shape[:2],tier,device=device))
        pred=decode(f,q)
        loss,stats=spatial_response_loss(pred.flatten(0,1),f.flatten(0,1),g,model,fine_weight=args.fine_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads=[v.grad for v in model.parameters() if v.grad is not None]
        if not torch.isfinite(loss) or not all(torch.isfinite(v).all() for v in grads):
            raise RuntimeError('nonfinite loss/gradient; optimizer not stepped')
        norm=float(torch.stack([v.norm() for v in grads]).norm())
        optimizer.step()
        trace.append({'step':step+1,'loss':float(loss.detach()),'grad_norm':norm,**stats})
        if step%10==0:print(f'{step+1}/{args.steps}: loss={float(loss.detach()):.5f}, grad={norm:.4g}',flush=True)
    after=evaluate()
    if digest()!=original_hash:raise RuntimeError('source checkpoint changed')
    out.mkdir(parents=True)
    report={'scope':'sampled blocks; diagnostic optimization only; no render or checkpoint writes',
            'checkpoint_sha256':original_hash,'args':vars(args),'sample_block_indices':indices.tolist(),
            'train_blocks':'even entries','heldout_blocks':'odd entries',
            'validation':'fixed q1/q2/q3 AWGN at 10 dB, one trial, XYZ/scale replacements are teacher-only diagnostics',
            'before':before,'after':after,'training':trace}
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for tier in ('1','2','3'):
        a,b=before[tier]['decoded'],after[tier]['decoded']
        print(f'q{tier}: radius ratio p50 {a["max_axis_ratio_p50"]:.3f} -> {b["max_axis_ratio_p50"]:.3f}; '
              f'XYZ RMSE {a["xyz_rmse_world"]:.3f} -> {b["xyz_rmse_world"]:.3f}')
    print(f'Saved diagnostics to {out}; input checkpoint unchanged.')


if __name__=='__main__':
    main()
