"""Real-PLY CPU short training. This is NOT a rendered-quality experiment."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import read_ply, prepare, to_features, to_raw
from gaussian_jscc.render_objective import bootstrap_loss
from gaussian_jscc.optimization import clip_codec_gradients, preserved_rng
from gaussian_jscc.transport import save_checkpoint, load_checkpoint, model_id
from gaussian_jscc.benchmark import parameter_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ply',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=400)
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    raw,degree=read_ply(args.ply)
    raw,geometry,_=prepare(raw,16)
    cfg=CodecConfig(architecture='learned_joint',loss_profile='learned_v1',sh_degree=degree,
                    hidden=48,depth=2,grid_dim=8,levels=(4,8),planes=False,
                    block_size=256,decoder_window=32)
    model=GaussianCodec(cfg)
    model.attr_mean.copy_(raw[:,3:].mean(0))
    model.attr_std.copy_(raw[:,3:].std(0,unbiased=False).clamp_min(.01))
    indices=np.linspace(0,len(raw)//256-1,40,dtype=int)
    source=torch.stack([raw[int(i)*256:(int(i)+1)*256] for i in indices])
    features=to_features(source.flatten(0,1),geometry,model)[0].reshape(40,256,-1)
    train_ids=torch.tensor([i for i in range(40) if i%5])
    val_ids=torch.tensor([i for i in range(40) if not i%5])
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)

    @torch.no_grad()
    def evaluate():
        with preserved_rng(torch.device('cpu')):
            torch.manual_seed(54321)
            model.eval()
            result={}
            for name,ids in [('train',train_ids[:8]),('held_out_blocks',val_ids)]:
                f=features[ids]
                for tier in (1,2,3):
                    q=torch.full(f.shape[:2],tier,dtype=torch.long)
                    pred,_,_=model.forward_tier_batches(f,f[...,:3],torch.nn.functional.one_hot(q,4).float(),10,'awgn')
                    decoded=to_raw(pred.flatten(0,1),geometry,model)
                    metrics=parameter_metrics(source[ids].flatten(0,1),decoded)
                    metrics['bootstrap_loss']=float(bootstrap_loss(pred.flatten(0,1),f.flatten(0,1)))
                    metrics['complex_symbols_per_gaussian']=cfg.rates[tier]
                    metrics['mean_transmitted_complex_energy']=float(model.encode(f[0],f[0,:,:3],q[0],10).square().sum(-1).mean())
                    result[f'{name}_q{tier}']=metrics
            model.train()
            return result

    initial=evaluate()
    started=time.perf_counter()
    for step in range(1,args.steps+1):
        step_started=time.perf_counter()
        selected=train_ids[torch.randint(len(train_ids),(4,))]
        f=features[selected]
        q=torch.randint(0,4,f.shape[:2]) if step%4==0 else torch.full(f.shape[:2],1+(step%3),dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        pred,_,_=model.forward_tier_batches(f,f[...,:3],torch.nn.functional.one_hot(q,4).float(),10,'awgn')
        loss=bootstrap_loss(pred[q>0],f[q>0])
        loss.backward()
        norm,stats=clip_codec_gradients(model,10.,'none')
        before=[p.detach().clone() for p in model.parameters()]
        optimizer.step()
        update=float(torch.stack([(p.detach()-old).norm() for p,old in zip(model.parameters(),before)]).norm())
        row={'step':step,'phase':'bootstrap','objective':'render_mse_v1','snr':10,
             'loss':float(loss.detach()),'grad_norm':float(norm),'update_norm':update,
             'step_seconds':time.perf_counter()-step_started,
             **stats}
        with (out/'loss.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(row)+'\n')
        if step==1 or step%50==0:
            print(f'{step}/{args.steps}: loss={float(loss.detach()):.5f}, grad={float(norm):.3f}, update={update:.5f}',flush=True)
    final=evaluate()
    save_checkpoint(out/'codec.pt',model,args.steps,{'scope':'40 real source blocks; CPU; no renderer; fixed AWGN10; no clipping'})
    fresh=load_checkpoint(out/'codec.pt','cpu')
    assert model_id(fresh)==model_id(model)
    q=torch.arange(256)%4
    f=features[val_ids[0]]
    with torch.no_grad():
        symbols=model.encode(f,f[:,:3],q,10)
        torch.testing.assert_close(model.decode(symbols,q,10),fresh.decode(symbols,q,10))
    result={'scope':'real PLY; 32 train blocks and 8 held-out blocks; fixed AWGN10; freshly initialized learned codec; no GPU rendering',
            'steps':args.steps,'seconds':time.perf_counter()-started,'config':cfg.to_dict(),
            'block_indices':indices.tolist(),'initial':initial,'final':final,'fresh_receiver_matches':True,
            'note':'Different architecture/objective from old runs. Loss magnitude is not a fair cross-architecture quality metric.'}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    from gaussian_jscc.plots import safe_plot
    safe_plot('training',out)
    print(json.dumps({k:{'xyz_rmse_initial':initial[k]['position_rmse'],
                         'xyz_rmse_final':v['position_rmse']} for k,v in final.items()},indent=2),flush=True)


if __name__=='__main__':
    main()
