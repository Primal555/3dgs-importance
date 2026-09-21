"""Larger matched CPU experiment: trunk vs zero-start block localization.

Random weights, q3/none, unchanged spatial_logcov_v1. No CUDA renderer.
Disjoint fit/heldout blocks from ONE scene, not cross-scene generalization.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
from gaussian_jscc.codec import CodecConfig, GaussianCodec
from gaussian_jscc.data import read_ply, prepare, to_features, fit_feature_statistics
from gaussian_jscc.spatial_response import spatial_response_loss
from gaussian_jscc.transport import save_checkpoint
from gaussian_jscc.optimization import clip_codec_gradients
from gaussian_jscc.checkpoint_history import file_hash


def append(path, row):
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, allow_nan=False)+'\n')


def position_metrics(pred, target, span):
    p, t = pred.double()*span.double(), target.double()*span.double()
    delta = p-t
    mean = delta.mean(0, keepdim=True)
    pc, tc = p-p.mean(0), t-t.mean(0)
    return dict(mse=float(delta.square().mean()), common_mse=float(mean.square().mean()),
                relative_mse=float((delta-mean).square().mean()),
                spread_ratio=float((pc.square().sum()/tc.square().sum().clamp_min(1e-20)).sqrt()))


def summarize(root):
    root = Path(root)
    protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    rows, histories = [], {}
    for seed in protocol['seeds']:
        for mode in protocol['modes']:
            key = f'{mode}_seed{seed}'
            folder = root/key if (root/key).is_dir() else root/f'seed{seed}'/key
            values = [json.loads(line) for line in (folder/'validation.jsonl').read_text().splitlines()]
            if values[-1]['step'] != protocol['steps']:
                raise ValueError('cannot summarize incomplete training')
            histories[key] = values
            tail = [v for v in values if v['step'] > 0][-3:]
            row = dict(mode=mode, seed=seed, steps=[v['step'] for v in tail])
            for split in ('fit','heldout'):
                row[split] = {k:sum(v[split][k] for v in tail)/len(tail) for k in tail[0][split]}
            rows.append(row)
    result = dict(scope='last three fixed checks; XYZ is world-space MSE; no rendering; do not select favorable checkpoints', rows=rows)
    (root/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    for metric in ('mse','common_mse','relative_mse','loss','shape','spread_ratio'):
        fig, axes = plt.subplots(2, len(protocol['seeds']), figsize=(6*len(protocol['seeds']), 7), squeeze=False)
        for col, seed in enumerate(protocol['seeds']):
            for row, split in enumerate(('fit','heldout')):
                ax = axes[row,col]
                for mode in protocol['modes']:
                    history = histories[f'{mode}_seed{seed}']
                    exponent = .5 if metric.endswith('mse') else 1
                    ax.plot([v['step'] for v in history], [v[split][metric]**exponent for v in history], label=mode)
                ax.set(title=f'{split}, seed={seed}', xlabel='Step', ylabel=metric.replace('mse','RMSE'), yscale='log')
                ax.legend(); ax.grid(alpha=.2)
        fig.suptitle('Same-scene CPU reconstruction; NOT render PSNR')
        fig.tight_layout(); fig.savefig(root/f'{metric}.png', dpi=140); plt.close(fig)
    return result


def run(args):
    if min(args.steps,args.every,args.threads,args.blocks_per_batch,args.save_every)<1 or args.blocks<8 or args.blocks%4:
        raise ValueError('positive counts required; blocks must be >=8 and divisible by 4')
    if len(args.seeds)!=len(set(args.seeds)) or len(args.modes)!=len(set(args.modes)):
        raise ValueError('duplicate seeds/modes')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('choose a fresh experiment output directory')
    torch.set_num_threads(args.threads)
    raw, degree = read_ply(args.ply)
    raw, geometry, _ = prepare(raw, 16)
    cfg = CodecConfig(architecture='learned_split_logcov',context_mode='multiscale_self',
                      encoder_attention='geometric_point',decoder_attention='transformer_trunk',sh_degree=degree)
    template = GaussianCodec(cfg); fit_feature_statistics(raw, template)
    total = len(raw)//cfg.block_size
    if total < args.blocks:
        raise ValueError('not enough complete PLY blocks')
    # Stratified midpoints over ALL complete blocks, not selected by outcomes.
    indices = [int((i+.5)*total/args.blocks) for i in range(args.blocks)]
    splits = {'fit':[i for i in range(args.blocks) if i%4!=3],
              'heldout':[i for i in range(args.blocks) if i%4==3]}
    if args.blocks_per_batch > len(splits['fit']):
        raise ValueError('batch exceeds fit blocks')
    features = torch.stack([to_features(raw[i*256:(i+1)*256],geometry,template)[0] for i in indices])
    out.mkdir(parents=True)
    protocol = dict(vars(args), indices=indices, splits=splits, codec_config=cfg.to_dict(),
                    source_gaussians=len(raw), ply_sha256=file_hash(args.ply),
                    scope='CPU; random weights; q3/none; fixed LR2e-4; no clipping; SAME loss; full-PLY normalization statistics; no camera/render test')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2),encoding='utf-8')
    del raw
    print(f'Protocol: {len(splits["fit"])} fit + {len(splits["heldout"])} heldout blocks; {args.steps} steps/case; seeds={args.seeds}', flush=True)
    for seed in args.seeds:
        for mode in args.modes:
            torch.manual_seed(seed)
            config = cfg.to_dict(); config['decoder_localization'] = mode
            model = GaussianCodec(CodecConfig.from_dict(config))
            model.attr_mean.copy_(template.attr_mean); model.attr_std.copy_(template.attr_std)
            optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
            sampler = torch.Generator().manual_seed(seed+10000)
            folder = out/f'{mode}_seed{seed}'; folder.mkdir()
            captures = {}
            hook = None
            if mode=='token_translation':
                hook = model.learned.dec_trunk.localization.register_forward_hook(lambda module, inputs, output: captures.update(shift=output.detach()))

            @torch.no_grad()
            def evaluate(step):
                model.eval(); result = {'step':step}
                for split, ids in splits.items():
                    metrics = []
                    for index in ids:
                        f = features[index]; q = torch.full(f.shape[:1],3,dtype=torch.long)
                        pred = model(f,f[:,:3],q,10,'none')
                        loss, stats = spatial_response_loss(pred,f,geometry,model,directions=torch.eye(3))
                        xyz = position_metrics(pred[:,:3],f[:,:3],geometry.span)
                        row = dict(xyz,loss=float(loss),position=stats['spatial_position_response'],shape=stats['spatial_logcov_shape_mse'])
                        if mode=='token_translation':
                            shift = captures['shift'][0]
                            base = position_metrics(pred[:,:3]-shift,f[:,:3],geometry.span)
                            row.update({f'without_translation_{k}':v for k,v in base.items()})
                            row['translation_mse'] = float((shift*geometry.span).square().mean())
                        metrics.append(row)
                        append(folder/'blocks.jsonl',dict(step=step,split=split,block=indices[index],**row))
                    result[split] = {k:sum(r[k] for r in metrics)/len(metrics) for k in metrics[0]}
                append(folder/'validation.jsonl',result)
                print(folder.name,step,{s:round(result[s]['mse']**.5,4) for s in splits},flush=True)
                model.train()

            evaluate(0)
            start = time.perf_counter()
            for step in range(1,args.steps+1):
                tick = time.perf_counter()
                ids = torch.randperm(len(splits['fit']),generator=sampler)[:args.blocks_per_batch]
                f = features[[splits['fit'][int(i)] for i in ids]]
                directions = torch.randn(4,3,generator=sampler)
                q = torch.full(f.shape[:2],3,dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                pred = model.forward_tier_batches(f,f[...,:3],F.one_hot(q,4).float(),10,'none')[0]
                loss,_ = spatial_response_loss(pred.flatten(0,1),f.flatten(0,1),geometry,model,directions=directions)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite objective')
                loss.backward()
                norm, gradient_stats = clip_codec_gradients(model,mode='none')
                optimizer.step()
                append(folder/'loss.jsonl',dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),seconds=time.perf_counter()-tick,**gradient_stats))
                if step%args.every==0 or step==args.steps:
                    evaluate(step)
                    print(f'Elapsed {time.perf_counter()-start:.1f}s',flush=True)
                if step%args.save_every==0 or step==args.steps:
                    save_checkpoint(folder/f'codec_{step}.pt',model,step,
                                    {'source_gaussians':protocol['source_gaussians'],
                                     'bootstrap_validation_blocks':[indices[i] for i in splits['heldout']],
                                     'scope':protocol['scope'],'experiment_protocol':str(out/'protocol.json')})
            if hook is not None:
                hook.remove()
    return summarize(out)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ply',required=True); parser.add_argument('--out',required=True)
    parser.add_argument('--steps',type=int,default=2000); parser.add_argument('--every',type=int,default=200)
    parser.add_argument('--save-every',type=int,default=500)
    parser.add_argument('--blocks',type=int,default=64); parser.add_argument('--blocks-per-batch',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4); parser.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    parser.add_argument('--workers',type=int,choices=[1,2],default=1,help='independent seed processes; each still pairs identical old/new protocols')
    parser.add_argument('--modes',nargs='+',choices=['none','token_translation'],default=['none','token_translation'])
    return parser


def run_parallel(args):
    """Independent seeds may run concurrently; preserve one experiment folder."""
    if len(args.seeds)!=len(set(args.seeds)):
        raise ValueError('duplicate seeds')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=False)
    def worker(seed):
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--ply',args.ply,
             '--out',str(root/f'seed{seed}'),'--steps',str(args.steps),'--every',str(args.every),
             '--save-every',str(args.save_every),'--blocks',str(args.blocks),
             '--blocks-per-batch',str(args.blocks_per_batch),'--threads',str(args.threads),
             '--seeds',str(seed),'--modes',*args.modes]
        with (root/f'seed{seed}.log').open('w',encoding='utf-8') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'Seed {seed} complete',flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker,args.seeds))
    protocol=json.loads((root/f'seed{args.seeds[0]}'/'protocol.json').read_text(encoding='utf-8'))
    protocol.update(vars(args))
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    return summarize(root)


if __name__=='__main__':
    args=build_parser().parse_args()
    (run_parallel if args.workers>1 else run)(args)
