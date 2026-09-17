"""CPU real-scene regression: unchanged symbol budgets, fresh receiver/no CSI leak.

Measures XYZ, not rendered image quality. Whole-scene results and fixed-block
short-training results are explicitly separated. Never modifies the initializer.
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_jscc.block_geometry import BlockGeometry, PilotBlockGeometry, position_objective
from gaussian_jscc.codec import channel
from gaussian_jscc.data import prepare, read_ply
from gaussian_jscc.transport import load_checkpoint, save_checkpoint
from diagnose_block_geometry import digest, write


@torch.no_grad()
def measure(model, blocks, span, tier, seed, kind='awgn'):
    errors, power_errors = [], []
    for i, xyz in enumerate(blocks):
        q = torch.full((len(xyz),), tier, dtype=torch.long)
        torch.manual_seed(seed+i)
        z = model.encode(xyz, q, 10.)
        power_errors.append(abs(float(z.square().sum()/model.rates[q].sum())-1))
        y = model.decode(channel(z.reshape(-1,2),10.,kind).reshape_as(z),q,10.)
        errors.append((y-xyz)*span)
    delta = torch.cat(errors)
    return dict(tier=tier, seed=seed, channel=kind, count=len(delta),
                rmse=float(delta.square().mean().sqrt()),
                p95=float(torch.quantile(delta.norm(dim=-1),.95)),
                maximum_power_deviation=max(power_errors))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ply', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True); p.add_argument('--steps', type=int, default=300)
    p.add_argument('--skip-survey', action='store_true')
    p.add_argument('--clean-weight', type=float, default=1.)
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    if args.steps < 0 or not 0 <= args.clean_weight < float('inf'):
        raise ValueError('steps and clean weight must be finite/nonnegative')
    torch.set_num_threads(4); torch.manual_seed(42)
    codec=load_checkpoint(args.checkpoint,'cpu')
    raw,_=read_ply(args.ply); raw,geometry,_=prepare(raw,codec.cfg.morton_bits)
    blocks=list(geometry.normalize(raw[:,:3]).split(codec.cfg.block_size))
    span=geometry.span
    indices=torch.linspace(0,len(blocks)-1,8).round().long().tolist()
    fixed=[blocks[i] for i in indices]
    write(out/'provenance.json',dict(ply_sha256=digest(args.ply), checkpoint_sha256=digest(args.checkpoint),
          code_sha256=digest('gaussian_jscc/block_geometry.py'), source_count=len(raw),
          indices=indices, snr=10, geometry_rates=codec.cfg.geometry_rates, steps=args.steps,clean_weight=args.clean_weight,
          torch=str(torch.__version__), device='cpu', all_units='source scene coordinate units, not metres'))
    # Same model weights (zero residual outputs), noise seed, and Gaussian population.
    initialization_rng=torch.get_rng_state()
    survey=[]
    for size in (() if args.skip_survey else (0,4096,1024,256,64)):
        torch.manual_seed(42)
        model=(BlockGeometry(codec.cfg.geometry_rates,codec.cfg.hidden) if size==0 else
               PilotBlockGeometry(codec.cfg.geometry_rates,codec.cfg.hidden,size))
        for tier in (1,2,3):
            row=measure(model,blocks,span,tier,100042)
            row.update(version='v4' if size==0 else 'v5',group_size=size)
            survey.append(row); write(out/'whole_scene_initial.json',survey)
            print('whole scene',row,flush=True)
    # Parameter-sweep noise must not change the short-training initialization.
    torch.set_rng_state(initialization_rng)
    codec.enable_block_geometry('block_pilot_v5')
    model=codec.block_geometry
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    logs=[]; evaluations=[]
    def validate(step):
        rows=[measure(model,fixed,span,t,s) for s in (100042,200042) for t in (1,2,3)]
        clean=[measure(model,fixed,span,t,100042,'none') for t in (1,2,3)]
        evaluations.append(dict(step=step,fixed_blocks=rows,noiseless=clean))
        write(out/'short_training.json',dict(logs=logs,evaluations=evaluations))
        print('short validation',step,[(r['tier'],round(r['rmse'],4)) for r in rows],flush=True)
    validate(0)
    for step in range(args.steps):
        torch.manual_seed(420000+step)
        xyz=blocks[int(torch.randint(len(blocks),(1,)))]
        optimizer.zero_grad(set_to_none=True); total=0.
        for tier in (1,2,3):
            q=torch.full((len(xyz),),tier,dtype=torch.long)
            z=model.encode(xyz,q,10.)
            y=model.decode(channel(z.reshape(-1,2),10.,'awgn').reshape_as(z),q,10.)
            loss,_=position_objective(y,xyz)
            if args.clean_weight:
                clean=model.decode(z,q,10.)
                clean_loss,_=position_objective(clean,xyz)
                loss=loss+args.clean_weight*clean_loss
            (loss/3).backward(); total+=float(loss.detach())/3
        grads={name:float(torch.nn.utils.clip_grad_norm_(module.parameters(),1.,error_if_nonfinite=True))
               for name,module in [('encoder',model.encoder),('decoder',model.decoder)]}
        optimizer.step()
        logs.append(dict(step=step+1,loss=total,gradient_norms=grads))
        if (step+1)%100==0 or step+1==args.steps:
            validate(step+1)
    final=[measure(model,blocks,span,t,s) for s in (100042,200042) for t in (1,2,3)]
    write(out/'whole_scene_trained.json',final)
    save_checkpoint(out/'codec.pt',codec,args.steps)
    print('final whole scene',final,flush=True)


if __name__=='__main__':
    main()
