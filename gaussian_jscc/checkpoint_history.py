"""Read-only, fixed-condition scene rendering of saved bootstrap checkpoints."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import prepare, read_ply, to_features
from .transport import load_checkpoint
from .render_validation import validate_render


def checkpoint_paths(training, start, stop, every):
    if min(start, stop, every) < 1 or stop < start or (stop-start) % every:
        raise ValueError('positive start/stop/every required, with stop on the requested interval')
    selected = [(s, Path(training)/f'codec_{s}.pt') for s in range(start,stop+1,every)]
    missing = [str(p) for _, p in selected if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing requested checkpoints (not skipped): '+', '.join(missing))
    return selected


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def plot_history(rows, out):
    """PNG/SVG six-metric history, ten checkpoints by four rate layouts.

    Separate source/photo references; never smooth. Fewer than eight observed
    checkpoints use markers only. Distinct tier colors plus marker shapes.
    CSV/JSON retain view/trial scope and communication costs for audit.
    """
    from .plots import _plt, _finish
    plt=_plt()
    fig,axes=plt.subplots(2,3,figsize=(16,8))
    first=rows[0]
    fig.suptitle(f'Checkpoint render history | {first["channel"]}, {first["snr"]:g} dB | '
                 f'{first["views"]} fixed views x {first["trials"]} trials\nEvaluation only: no weight updates')
    metrics=(('source_psnr','PSNR vs source PLY (dB)'),('source_ssim','SSIM vs source PLY'),
             ('source_mse','MSE vs source PLY'),('photo_psnr','PSNR vs photo (dB)'),
             ('photo_ssim','SSIM vs photo'),('xyz_rmse_retained','XYZ RMSE (scene units)'))
    for tier,color,marker in zip(('1','2','3','mixed'),('#2F6B9A','#D8A72E','#D96C2F','#737A36'),('o','s','^','D')):
        group=[r for r in rows if r['layout']==tier]
        for ax,(key,title) in zip(axes.flat,metrics):
            ax.plot([r['step'] for r in group],[r[key] for r in group],color=color,marker=marker,
                    linestyle='-' if len(group)>=8 else 'none',markersize=4,label='q'+tier)
            ax.set(title=title,xlabel='Bootstrap training step')
    for ax,key in ((axes[1,0],'reference_photo_psnr'),(axes[1,1],'reference_photo_ssim')):
        ax.axhline(rows[0][key],color='#59636E',linestyle='--',label='Source PLY vs photo')
    for ax in axes.flat:
        ax.legend(fontsize=8)
    _finish(fig,Path(out)/'quality_vs_step')


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','ply','source'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--out', default=None,
                   help='default: TRAINING/render_history; existing output is never overwritten')
    p.add_argument('--start',type=int,default=500)
    p.add_argument('--stop',type=int,default=5000)
    p.add_argument('--every',type=int,default=500)
    p.add_argument('--device',default='cuda')
    p.add_argument('--snr',type=float,default=10.)
    p.add_argument('--channel',choices=['awgn','none'],default='awgn')
    p.add_argument('--trials',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--blocks-per-batch',type=int,default=32)
    p.add_argument('--views',type=int,default=8,help='fixed spaced cameras; 0 uses all cameras of selected split')
    p.add_argument('--split',choices=['train','test'],default='test')
    p.add_argument('--resolution',type=int,default=2)
    p.add_argument('--images',default='images')
    p.add_argument('--white-background',action='store_true')
    return p


@torch.no_grad()
def evaluate_history(args):
    from .cli import device_for
    from .rendering import load_cameras, RenderReference
    from .render_objective import spaced_indices
    if not args.device.startswith('cuda'):
        raise ValueError('scene rendering requires CUDA')
    if min(args.trials,args.blocks_per_batch)<1 or args.views<0 or not math.isfinite(args.snr):
        raise ValueError('invalid trials, batch size, views or SNR')
    paths=checkpoint_paths(args.training,args.start,args.stop,args.every)
    out=Path(args.out) if args.out is not None else Path(args.training)/'render_history'
    if out.exists():
        raise FileExistsError(f'Use a new evaluation directory: {out}')
    # Keep evaluation files in a separate subdirectory, not alongside weights.
    # The experiment can now be copied as one complete directory.
    device=device_for(args.device)
    raw,degree=read_ply(args.ply)
    model=load_checkpoint(paths[0][1],device).eval()
    if model.cfg.sh_degree != degree or model.cfg.position_delivery != 'learned':
        raise ValueError('expected matching learned-XYZ checkpoint and PLY SH degree')
    config=model.cfg.to_dict()
    mean,std=model.attr_mean.detach().cpu().clone(),model.attr_std.detach().cpu().clone()
    raw,geometry,_=prepare(raw,model.cfg.morton_bits)
    blocks,ids=[],[]
    for start in range(0,len(raw),model.cfg.block_size):
        batch=raw[start:start+model.cfg.block_size]
        f,_=to_features(batch.to(device),geometry,model)
        blocks.append(f.cpu())
        ids.append(torch.arange(start,start+len(f)))
    groups=[pad_sequence(blocks[i:i+args.blocks_per_batch],batch_first=True)
            for i in range(0,len(blocks),args.blocks_per_batch)]
    group_ids=[pad_sequence(ids[i:i+args.blocks_per_batch],batch_first=True,padding_value=-1)
               for i in range(0,len(ids),args.blocks_per_batch)]
    all_cameras=load_cameras(args.source,args.resolution,args.white_background,args.images,args.split)
    indices=spaced_indices(len(all_cameras),args.views) if args.views else list(range(len(all_cameras)))
    cameras=[all_cameras[i] for i in indices]
    del all_cameras,blocks,ids
    reference=RenderReference(raw,degree,args.white_background,'source')
    out.mkdir(parents=True,exist_ok=False)
    record=vars(args).copy()
    record['out']=str(out)
    record.update(scope='offline checkpoint evaluation, no optimization; same scene, not unseen-scene evidence',
                  source_gaussians=len(raw),checkpoint_steps=[s for s,_ in paths],
                  camera_indices=indices,camera_names=[str(getattr(c,'image_name',i)) for i,c in enumerate(cameras)],
                  seed_protocol='same validation RNG reset per layout/trial for every checkpoint; mixed layout fixed',
                  metrics='source: original PLY render; photo: photographs; dB metrics averaged per view/trial',
                  images='trial 0 only; Photo | Source PLY | Received | absolute source error x4',
                  rate_scope='JSCC payload; no XYZ side stream; existing global bbox/model/tier metadata not fully charged')
    (out/'evaluation_config.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    results=[]
    for index,(step,path) in enumerate(paths):
        if index:
            del model
            model=load_checkpoint(path,device).eval()
        if model.cfg.to_dict()!=config or not torch.equal(model.attr_mean.cpu(),mean) or not torch.equal(model.attr_std.cpu(),std):
            raise ValueError(f'architecture or feature statistics changed at {path}; cannot reuse encoding inputs')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('step') != step:
            raise ValueError(f'checkpoint step does not match filename: {path}')
        expected=saved.get('training',{}).get('source_gaussians')
        if expected is not None and expected != len(raw):
            raise ValueError('PLY Gaussian count differs from checkpoint training scene')
        del saved
        before=file_hash(path)
        model.requires_grad_(False)
        result=validate_render(model,groups,group_ids,raw,geometry,cameras,reference,args.snr,args.channel,
                               args.trials,args.seed,out,step,'offline_bootstrap',white_background=args.white_background)
        if file_hash(path)!=before:
            raise RuntimeError(f'Checkpoint changed during evaluation: {path}')
        result.update(checkpoint=path.name,checkpoint_sha256=before)
        results.append(result)
        (out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
        flat=[dict({k:v for k,v in e.items() if k not in ('views','tier_counts')},step=r['step'],
                   checkpoint=r['checkpoint'],snr=args.snr,channel=args.channel,trials=args.trials,views=len(cameras))
              for r in results for e in r['layouts']]
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]))
            writer.writeheader();writer.writerows(flat)
        print(f'[{index+1}/{len(paths)}] checkpoint {step}: saved images and metrics; weights unchanged',flush=True)
    plot_history(flat,out)
    print(f'Saved checkpoint render history: {out}',flush=True)
    return results


def main(argv=None):
    evaluate_history(build_parser().parse_args(argv))
