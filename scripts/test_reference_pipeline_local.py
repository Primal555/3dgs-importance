"""Real PLY attribute/context + mask backward smoke test. NOT render quality.

Uses two source blocks; a differentiable attribute proxy replaces CUDA rendering.
Writes diagnostics only, not deployment codec/allocation checkpoints.
"""
import argparse
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussian_jscc.data import read_ply,prepare,to_features
from gaussian_jscc.transport import load_checkpoint
from gaussian_jscc.optimization import all_tier_attribute_step,clip_codec_gradients
from gaussian_jscc.training import joint_scene_step
from gaussian_jscc.allocation import GaussianTierMask
from diagnose_block_geometry import write,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--ply',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=10)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    model=load_checkpoint(args.checkpoint,'cpu').train()
    if model.cfg.position_head!='reference_v6':raise ValueError('requires v6 checkpoint')
    raw,_=read_ply(args.ply);raw,geo,_=prepare(raw,model.cfg.morton_bits)
    size=model.cfg.block_size;indices=[0,max(0,len(raw)//size//2)]
    blocks=[to_features(raw[i*size:(i+1)*size],geo,model)[0] for i in indices]
    target=torch.cat(blocks).detach();mask=GaussianTierMask(len(target))
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    mask_optimizer=torch.optim.Adam(mask.parameters(),lr=1e-4)
    batches=[];offset=0
    for b in blocks:
        batches.append((b[None],torch.arange(offset,offset+len(b))[None]));offset+=len(b)
    logs=[]
    for phase in ('attribute_context','joint_proxy'):
        for step in range(args.steps):
            torch.manual_seed(1000+step);optimizer.zero_grad(set_to_none=True);mask_optimizer.zero_grad(set_to_none=True)
            if phase=='attribute_context':
                loss,stats=all_tier_attribute_step(model,blocks[step%len(blocks)],geo,10.,'awgn')
            else:
                def distortion(scene,active):
                    f,_=to_features(scene,geo,model)
                    return F.smooth_l1_loss(f*active[:,None],target)
                loss,stats=joint_scene_step(model,mask,batches,geo,10.,'awgn',distortion,
                                            beta=0.,attr_weight=.1,mode='replay')
            norm,clip=clip_codec_gradients(model,1.,'branch')
            mask_norm=float(torch.nn.utils.clip_grad_norm_(mask.parameters(),1.,error_if_nonfinite=True))
            optimizer.step();mask_optimizer.step()
            logs.append(dict(phase=phase,step=step+1,loss=float(loss),codec_grad_norm=float(norm),
                             mask_grad_norm=mask_norm,**clip))
            print(phase,step+1,'loss',float(loss),'mask_grad',mask_norm,flush=True)
            write(out/'results.json',dict(checkpoint_sha256=digest(args.checkpoint),source_blocks=indices,
                  source_count=len(target),note='CPU attribute proxy; NOT rendered views or held-out quality',logs=logs))


if __name__=='__main__':main()
