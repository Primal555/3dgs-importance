"""CPU whole-scene comparison plus 300-step mixed-layout geometry training.

No renderer-quality claim. Original PLY and checkpoints are never overwritten.
"""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gaussian_jscc.transport import load_checkpoint,save_checkpoint
from gaussian_jscc.data import read_ply,prepare,to_features
from gaussian_jscc.optimization import all_tier_geometry_step,clip_codec_gradients
from test_pilot_geometry_local import measure
from diagnose_block_geometry import digest,write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ply',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--steps',type=int,default=300)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    model=load_checkpoint(args.checkpoint,'cpu')
    raw,_=read_ply(args.ply);raw,geometry,_=prepare(raw,model.cfg.morton_bits)
    features,_=to_features(raw,geometry,model)
    features=list(features.split(model.cfg.block_size));blocks=[f[:,:3] for f in features]
    indices=torch.linspace(0,len(blocks)-1,8).round().long().tolist();fixed=[blocks[i] for i in indices]
    meta=dict(checkpoint=args.checkpoint,checkpoint_sha256=digest(args.checkpoint),ply_sha256=digest(args.ply),
              code_sha256=digest('gaussian_jscc/reference_geometry.py'),steps=args.steps,snr=10.,source_count=len(raw),
              fixed_indices=indices,scope='same-scene geometry only; independent validation noise; no CUDA rendering')
    write(out/'provenance.json',meta)
    before=[measure(model.block_geometry,blocks,geometry.span,t,100042) for t in (1,2,3)]
    write(out/'v5_input_whole_scene.json',before);print('input v5',before,flush=True)
    torch.manual_seed(42);model.enable_block_geometry('reference_v6');model.cfg.geometry_clean_weight=1.
    model.cfg.loss_profile='position_v3'
    for n,p in model.named_parameters():p.requires_grad_(n.startswith('block_geometry.'))
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    initial=[measure(model.block_geometry,blocks,geometry.span,t,100042) for t in (1,2,3)]
    write(out/'v6_initial_whole_scene.json',initial);print('initial v6',initial,flush=True)
    logs=[];evaluations=[]
    def validate(step):
        rows=[measure(model.block_geometry,fixed,geometry.span,t,100042) for t in (1,2,3)]
        clean=[measure(model.block_geometry,fixed,geometry.span,t,100042,'none') for t in (1,2,3)]
        evaluations.append(dict(step=step,awgn=rows,clean=clean))
        write(out/'training.json',dict(logs=logs,evaluations=evaluations))
        print('step',step,'fixed RMSE',[(r['tier'],r['rmse']) for r in rows],flush=True)
    validate(0)
    for step in range(args.steps):
        torch.manual_seed(420000+step)
        index=int(torch.randint(len(features),(1,)))
        optimizer.zero_grad(set_to_none=True)
        loss,stats=all_tier_geometry_step(model,features[index],geometry,10.,'awgn')
        norm,clipping=clip_codec_gradients(model,1.,'branch')
        optimizer.step()
        logs.append(dict(step=step+1,loss=float(loss),block=index,grad_norm=float(norm),**clipping,**stats))
        if (step+1)%100==0 or step+1==args.steps:validate(step+1)
    rows=[measure(model.block_geometry,blocks,geometry.span,t,s) for s in (100042,200042) for t in (1,2,3)]
    write(out/'v6_trained_whole_scene.json',rows);save_checkpoint(out/'codec.pt',model,args.steps,meta)
    print('final v6',rows,flush=True)


if __name__=='__main__':main()
