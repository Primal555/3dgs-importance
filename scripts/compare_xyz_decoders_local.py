"""Matched real-PLY CPU decoder comparison, not a render-quality benchmark."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import read_ply, prepare, to_features, fit_feature_statistics
from gaussian_jscc.spatial_response import spatial_response_loss
from gaussian_jscc.transport import save_checkpoint


def summarize(roots, out, tail_checks=3):
    """Conservative local screening, not a proof of full-scene render gains."""
    if tail_checks<1:
        raise ValueError('positive tail_checks required')
    roots=[Path(p) for p in roots];out=Path(out)
    protocols=[json.loads((p/'protocol.json').read_text(encoding='utf-8')) for p in roots]
    for key in ('ply','steps','every','blocks','seeds','config'):
        if any(p[key]!=protocols[0][key] for p in protocols):
            raise ValueError(f'comparison protocol mismatch: {key}')
    rows=[];curves=[]
    for root,p in zip(roots,protocols):
        for seed in p['seeds']:
            for mode in [m if a=='window' else a+'__'+m
                         for a in p.get('decoder_attentions',['window']) for m in p['modes']]:
                folder=root/f'{mode}_seed{seed}'
                checks=[json.loads(s) for s in (folder/'validation.jsonl').read_text().splitlines()]
                curves.append((mode,seed,checks))
                if checks[-1]['step']!=p['steps']:
                    raise ValueError('incomplete run')
                tail=[v for v in checks if v['step']>0][-tail_checks:]
                row={'mode':mode,'seed':seed,'steps':[v['step'] for v in tail]}
                for split in ('fit','heldout'):
                    for key in ('mse','common_mse','relative_mse'):
                        row[f'{split}_{key}']=sum(v[split][key] for v in tail)/len(tail)
                rows.append(row)
    if len({(r['mode'],r['seed']) for r in rows})!=len(rows):
        raise ValueError('duplicate mode/seed')
    refs={r['seed']:r for r in rows if r['mode']=='additive'}
    decisions={}
    for mode in sorted({r['mode'] for r in rows}-{'additive'}):
        candidates=[r for r in rows if r['mode']==mode]
        checks=[]
        for r in candidates:
            ref=refs.get(r['seed'])
            if ref is None or ref['steps']!=r['steps']:
                raise ValueError('missing matched baseline')
            checks.append(all(r[f'{split}_mse']<=ref[f'{split}_mse'] for split in ('fit','heldout')))
        decisions[mode]={'local_screen_pass':len(checks)>=2 and all(checks),
                         'seeds_passing':sum(checks),'seeds_tested':len(checks)}
    result={'criterion':'tail mean XYZ MSE no worse than matched additive in BOTH fit and heldout for every tested seed; >=2 seeds; engineering screen, NOT statistical significance or render quality',
            'tail_checks':tail_checks,'rows':rows,'decisions':decisions}
    paired=[]
    index={(r['mode'],r['seed']):r for r in rows}
    for r in rows:
        if r['mode'].startswith('feature_point__'):
            ref=index.get((r['mode'].split('__',1)[1],r['seed']))
            if ref is not None:
                if ref['steps']!=r['steps']:
                    raise ValueError('unmatched decoder-attention comparison steps')
                paired.append(dict(mode=r['mode'],seed=r['seed'],reference=ref['mode'],
                                   fit_mse_ratio=r['fit_mse']/max(ref['fit_mse'],1e-30),
                                   heldout_mse_ratio=r['heldout_mse']/max(ref['heldout_mse'],1e-30)))
    result['paired_decoder_attention']=paired
    out.mkdir(parents=True,exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    seeds=protocols[0]['seeds']
    fig,axes=plt.subplots(2,len(seeds),figsize=(6*len(seeds),8),squeeze=False)
    for col,seed in enumerate(seeds):
        for row,split in enumerate(('fit','heldout')):
            axis=axes[row,col]
            for mode,s,checks in curves:
                if s==seed:
                    axis.plot([v['step'] for v in checks],[v[split]['mse']**.5 for v in checks],label=mode)
            axis.set(title=f'{split}, seed={seed}',xlabel='Step',ylabel='World XYZ RMSE',yscale='log')
            axis.legend(fontsize=8);axis.grid(alpha=.2)
    fig.suptitle('CPU small-block comparison; fixed q3, no noise; NOT render quality')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    return result


def run(args):
    if min(args.steps,args.threads,args.every)<1:
        raise ValueError('positive steps/threads/every required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads)
    raw,degree=read_ply(args.ply)
    raw,geometry,_=prepare(raw,16)
    config=CodecConfig(architecture='learned_split_logcov',context_mode='multiscale_self',
                       encoder_attention='geometric_point',sh_degree=degree)
    template=GaussianCodec(config);fit_feature_statistics(raw,template)
    indices=args.blocks
    if len(indices)<4 or len(indices)!=len(set(indices)) or min(indices)<0 or (max(indices)+1)*256>len(raw):
        raise ValueError('need >=4 distinct complete block indices')
    features=torch.stack([to_features(raw[i*256:(i+1)*256],geometry,template)[0] for i in indices])
    del raw
    splits={'fit':list(range(0,len(indices),2)),'heldout':list(range(1,len(indices),2))}
    protocol=dict(vars(args),config=config.to_dict(),splits=splits,
                  scope='Random weights; full-PLY statistics; fixed q3/none; no clipping; CPU; no renderer',
                  comparison='Same seed for shared initialization; separate identical sampling/direction RNG per run; block_center initial XYZ differs')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results={}
    for seed in args.seeds:
        for attention,mode in [(a,m) for a in getattr(args,'decoder_attentions',['window']) for m in args.modes]:
            torch.manual_seed(seed)
            cfg=config.to_dict();cfg['xyz_decoder']=mode
            cfg['decoder_attention']=attention
            model=GaussianCodec(CodecConfig.from_dict(cfg))
            model.attr_mean.copy_(template.attr_mean);model.attr_std.copy_(template.attr_std)
            optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
            sampler=torch.Generator().manual_seed(seed+10000)
            case=mode if attention=='window' else attention+'__'+mode
            key=f'{case}_seed{seed}';folder=out/key;folder.mkdir()
            history=[]

            @torch.no_grad()
            def evaluate(step):
                result={'step':step}
                model.eval()
                for split,ids in splits.items():
                    metrics=[]
                    for i in ids:
                        f=features[i];q=torch.full(f.shape[:1],3,dtype=torch.long)
                        pred=model(f,f[:,:3],q,10,'none')
                        loss,stats=spatial_response_loss(pred,f,geometry,model,directions=torch.eye(3))
                        delta=(pred[:,:3]-f[:,:3]).double()*geometry.span.double()
                        center=delta.mean(0,keepdim=True)
                        metrics.append(dict(loss=float(loss),mse=float(delta.square().mean()),
                                            common_mse=float(center.square().mean()),
                                            relative_mse=float((delta-center).square().mean()),
                                            position=stats['spatial_position_response'],
                                            shape=stats['spatial_logcov_shape_mse']))
                    result[split]={k:sum(m[k] for m in metrics)/len(metrics) for k in metrics[0]}
                model.train();history.append(result)
                with (folder/'validation.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(result)+'\n')
                print(key,step,{s:round(result[s]['mse']**.5,4) for s in splits},flush=True)

            evaluate(0)
            started=time.perf_counter()
            for step in range(1,args.steps+1):
                ids=torch.randperm(len(splits['fit']),generator=sampler)[:2]
                f=features[[splits['fit'][int(i)] for i in ids]]
                directions=torch.randn(4,3,generator=sampler)
                q=torch.full(f.shape[:2],3,dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                pred=model.forward_tier_batches(f,f[...,:3],torch.nn.functional.one_hot(q,4).float(),10,'none')[0]
                loss,_=spatial_response_loss(pred.flatten(0,1),f.flatten(0,1),geometry,model,directions=directions)
                loss.backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not all(torch.isfinite(g).all() for g in grads):
                    raise RuntimeError(f'nonfinite gradients: {key} step {step}')
                norm=float(torch.stack([g.detach().norm() for g in grads]).norm())
                optimizer.step()
                with (folder/'loss.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(dict(step=step,loss=float(loss.detach()),grad_norm=norm))+'\n')
                if step%args.every==0 or step==args.steps:
                    evaluate(step)
            save_checkpoint(folder/'codec.pt',model,args.steps,{'scope':protocol['scope']})
            results[key]={'initial':history[0],'final':history[-1],'seconds':time.perf_counter()-started}
            (out/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ply',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--every',type=int,default=50)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    p.add_argument('--decoder-attentions',nargs='+',choices=['window','feature_point'],default=['window'])
    p.add_argument('--modes',nargs='+',choices=['additive','block_center','context_center','residual_center','symbol_skip'],default=['additive','block_center','context_center'])
    p.add_argument('--blocks',type=int,nargs='+',default=[40,440,840,1240,1640,2040,2440,2840])
    run(p.parse_args())
