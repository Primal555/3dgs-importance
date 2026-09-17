"""Paired real-PLY gradient probes + mixed-tier CPU training (not render quality)."""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussian_jscc.transport import load_checkpoint,save_checkpoint
from gaussian_jscc.data import read_ply,prepare,to_features
from gaussian_jscc.losses import PROFILES,reconstruction_loss,position_training_inputs
from gaussian_jscc.optimization import all_tier_attribute_step,clip_codec_gradients,parameter_group
from gaussian_jscc.training import joint_scene_step
from gaussian_jscc.allocation import GaussianTierMask
from diagnose_block_geometry import digest


def upgrade(model):
    model.cfg.individual_tiers=True
    model.block_geometry.individual=True
    model.cfg.loss_profile='robust_v4'
    for key,value in PROFILES['robust_v4'].items(): setattr(model.cfg,key,value)
    model.cfg.__post_init__()
    return model


def norms(model):
    groups={}
    for name,p in model.named_parameters():
        if p.grad is not None:
            key=parameter_group(name)
            groups[key]=groups.get(key,0.)+float(p.grad.square().sum())
    return {k:v**.5 for k,v in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('ply','checkpoint','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=20);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    base=load_checkpoint(args.checkpoint,'cpu').train()
    if base.cfg.position_head!='reference_v6' or base.cfg.individual_tiers:
        raise ValueError('initializer must be the previous reference_v6 codec without individual_tiers')
    raw,_=read_ply(args.ply);raw,geo,_=prepare(raw,base.cfg.morton_bits)
    size=base.cfg.block_size;indices=[0,len(raw)//size//2]
    blocks=[to_features(raw[i*size:(i+1)*size],geo,base)[0] for i in indices]
    result=dict(checkpoint_sha256=digest(args.checkpoint),source_blocks=indices,
        note='CPU gradients/attribute proxy only; NOT held-out rendering or communications success',probes=[],logs=[])
    def write(): (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for label in ('old','wire_only','wire_and_robust_loss'):
        m=copy.deepcopy(base)
        if label!='old':m.cfg.individual_tiers=True;m.block_geometry.individual=True
        if label=='wire_and_robust_loss':upgrade(m)
        for bi,f in enumerate(blocks):
            for layout in ('low','mixed'):
                torch.manual_seed(1000+bi)
                q=torch.ones(len(f),dtype=torch.long) if layout=='low' else torch.randint(0,4,(len(f),))
                m.zero_grad(set_to_none=True)
                c=F.one_hot(q,4).float()
                pred,_,active=m.forward_tiers(f,f[:,:3],c,10,'awgn')
                loss,terms=reconstruction_loss(pred,f,geo,m,active=active,return_terms=True,
                    **position_training_inputs(m,f,c,10))
                loss.backward()
                row=dict(variant=label,block=indices[bi],layout=layout,loss=float(loss.detach()),
                    gradients=norms(m),terms={k:float(v.detach().mean()) for k,v in terms.items()},
                    position_rmse=float((((pred[q>0,:3]-f[q>0,:3])*geo.span).square().mean().sqrt()).detach()))
                result['probes'].append(row);write()
                print('probe',label,indices[bi],layout,row['gradients'],flush=True)
    model=upgrade(copy.deepcopy(base));optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    for step in range(args.steps):
        torch.manual_seed(2000+step);optimizer.zero_grad(set_to_none=True)
        loss,stats=all_tier_attribute_step(model,blocks[step%len(blocks)],geo,10,'awgn')
        norm,clip=clip_codec_gradients(model,1.,'branch');optimizer.step()
        result['logs'].append(dict(step=step+1,loss=float(loss),grad_norm=float(norm),**stats,**clip));write()
        print('mixed training',step+1,'loss',float(loss),'grad',float(norm),flush=True)
    target=torch.cat(blocks).detach();mask=GaussianTierMask(len(target));batches=[];offset=0
    for b in blocks:
        batches.append((b[None],torch.arange(offset,offset+len(b))[None]));offset+=len(b)
    def distortion(scene,active):
        f,_=to_features(scene,geo,model)
        return F.smooth_l1_loss(f*active[:,None],target)
    model.zero_grad(set_to_none=True);torch.manual_seed(3000)
    loss,stats=joint_scene_step(model,mask,batches,geo,10,'awgn',distortion,beta=0.,attr_weight=.1,mode='replay')
    result['joint_proxy']=dict(loss=float(loss),mask_grad_norm=float(mask.logits.grad.norm()),
                              gradients=norms(model))
    write();save_checkpoint(out/'codec.pt',model,args.steps,dict(diagnostic_only=True))


if __name__=='__main__':main()
