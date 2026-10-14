"""Matched Truck channel-capacity controls with one immutable semantic teacher."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from .modular_experiment import parser as training_parser, json_write
from .modular_plots import read_rows
from .pointnet_context_controls import summarize as summarize_quality

GROUPS=('baseline','residual','progressive')


def parser():
    p=training_parser()
    p.description=__doc__
    p.set_defaults(semantic_bootstrap_steps=0,semantic_render_steps=0,
                   freeze_semantic=True,color_metrics=True)
    p.add_argument('--groups',nargs='+',choices=GROUPS,default=['baseline','residual'])
    p.add_argument('--resume-suite',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    return p


def build_plan(args):
    if not args.init_semantic or args.resume or args.clean_only or args.with_direct_control:
        raise ValueError('controls require --init-semantic; use --resume-suite to continue')
    if args.semantic_bootstrap_steps or args.semantic_render_steps or args.semantic_color_every:
        raise ValueError('controls do not retrain the semantic representation')
    if len(set(args.groups))!=len(args.groups): raise ValueError('duplicate groups')
    run=Path(args.run).resolve(); checkpoint=Path(args.init_semantic).resolve()
    if not (run/'training.json').is_file(): raise FileNotFoundError(run/'training.json')
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    # Copy the actual semantic dimensions/geometry, not generic CLI defaults.
    source=json.loads((run/'training.json').read_text(encoding='utf-8'))
    cfg=source.get('codec_config',{})
    teacher_batch=source.get('scene',source).get('blocks_per_batch')
    if args.blocks_per_batch is not None and args.blocks_per_batch!=teacher_batch:
        raise ValueError('retain the teacher batch size for scene-identity validation; omit --blocks-per-batch')
    common=['--run',str(run),'--init-semantic',str(checkpoint),'--freeze-semantic','--color-metrics',
            '--semantic-bootstrap-steps','0','--semantic-render-steps','0',
            '--context-variant',cfg.get('context_variant','baseline'),
            '--latent-dim',str(cfg.get('latent_dim',48)),
            '--decoder-geometry',cfg.get('decoder_geometry','xyz')]
    for key in ('context_centers','context_neighbors','context_radii'):
        if key in cfg: common+=['--'+key.replace('_','-')]+[str(v) for v in cfg[key]]
    for key in ('manifest','device','blocks_per_batch','train_snrs','channel_latent_weight',
                'validate_every','save_every','seed','cpu_threads','channel_hidden',
                'channel_bootstrap_steps','channel_render_steps',
                'semantic_encoder_lr','semantic_decoder_lr','channel_encoder_lr','channel_decoder_lr'):
        value=getattr(args,key)
        if value is not None:
            common+=['--'+key.replace('_','-')]+[str(v) for v in (value if isinstance(value,(list,tuple)) else [value])]
    # Do not change historical default suite commands/identity.
    if tuple(args.rates)!=(4,12,24):
        common+=['--rates',*[str(v) for v in args.rates]]
    jobs=[{'group':group,'command':[sys.executable,'-u','-m','gaussian_jscc.modular_experiment',
           *common,'--out',str(Path(args.out).resolve()/group),'--channel-variant',group]} for group in args.groups]
    plan={'version':1,'jobs':jobs,'groups':args.groups,'rates':[0,*args.rates],
            'teacher_file_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'teacher':str(checkpoint),'semantic_modules':'frozen in both communication phases',
            'losses':('latent MSE then source render MSE + unchanged latent MSE' if args.channel_render_steps
                      else 'latent MSE only; render evaluation without updates'),
            'training_layout_cycle':['1','2','3','mixed'],'matched_rng_seed':args.seed+17,
            'parameter_and_compute_matched':False,'power_normalization':'unchanged per-point per-segment'}
    if 'progressive' in args.groups:
        plan['progressive_design']={'encoder':'unchanged residual encoder',
            'decoder':'base latent + suffix-conditioned latent correction + suffix-conditioned latent correction',
            'previous_latent_gradients':'joint; not detached','permanent_stage_freezing':False,
            'extra_side_information':False,'monotonic_quality_guarantee':False}
    return plan


def semantic_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        if name.startswith(('semantic_encoder.','semantic_decoder.')) or name in ('attr_mean','attr_std'):
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def summarize(out,groups,teacher):
    out=Path(out)
    expected=semantic_digest(torch.load(teacher,map_location='cpu',weights_only=True)['state_dict'])
    audit={}
    for group in groups:
        latest=out/group/'checkpoints/latest/codec.pt'
        if latest.exists():
            saved=torch.load(latest,map_location='cpu',weights_only=True)
            unchanged=semantic_digest(saved['state_dict'])==expected
            if not unchanged: raise RuntimeError('semantic weights/statistics changed: '+group)
            audit[group]={'semantic_weights_unchanged':unchanged,'step':saved['step']}
    result=summarize_quality(out,groups,ignore_config_keys=('channel_variant','channel_hidden'))
    result.update(semantic_audit=audit,teacher_semantic_sha256=expected,
                  interpretation='same frozen semantics and symbols; compare quality, latent NMSE and color across ALL tiers')
    # Existing quality plots plus per-tier latent and color curves, no extra renders.
    plot_channel_curves(out,groups)
    json_write(out/'comparison/summary.json',result)
    return result


def plot_channel_curves(out,groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    for group in groups:
        rows=read_rows(out/group/'metrics/validation.jsonl')
        for col,tier in enumerate(('1','2','3')):
            selected={r['step']:r for r in rows if r['path']=='awgn' and r['layout']==tier and r['actual_snr']==10}
            selected=[selected[s] for s in sorted(selected)]
            for row,key in enumerate(('source_psnr','latent_nmse','source_color_rg_error_rmse')):
                usable=[r for r in selected if r.get(key) is not None]
                if usable: axes[row,col].plot([r['step'] for r in usable],[r[key] for r in usable],label=group)
    for row,key in enumerate(('Source PSNR (dB)','Latent NMSE','R-G error RMSE')):
        for col in range(3):
            ax=axes[row,col]; ax.set(title=f'10 dB q{col+1}',xlabel='Channel optimizer updates',ylabel=key)
            ax.grid(alpha=.2)
            if ax.lines: ax.legend()
    fig.savefig(out/'comparison/channel_curves.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for group in groups:
        rows=read_rows(out/group/'metrics/loss.jsonl')
        for ax,phase in zip(axes,('channel_bootstrap','channel_render')):
            for tier in ('1','2','3','mixed'):
                selected=[r for r in rows if r['phase']==phase and r['layout']==tier]
                if not selected: continue
                # Smooth independently per tier; never mix different rate objectives.
                xs=[]; ys=[]
                for start in range(0,len(selected),25):
                    window=selected[start:start+25]
                    xs.append(window[-1]['phase_step']); ys.append(sum(r['latent_mse'] for r in window)/len(window))
                ax.plot(xs,ys,label=f'{group}/q{tier}')
            ax.set(title=phase,xlabel='Phase optimizer updates',ylabel='Latent MSE'); ax.grid(alpha=.2)
    for ax in axes:
        if ax.lines: ax.legend(fontsize=7)
    fig.savefig(out/'comparison/latent_training.png',dpi=140); plt.close(fig)


def main(args):
    plan=build_plan(args); out=Path(args.out).resolve()
    if args.dry_run:
        print(json.dumps(plan,indent=2)); return plan
    if args.summarize_only: return summarize(out,args.groups,plan['teacher'])
    if out.exists():
        if not args.resume_suite: raise FileExistsError('use a NEW output directory or --resume-suite')
        if json.loads((out/'plan.json').read_text())!=plan: raise ValueError('resume suite plan differs')
    else:
        out.mkdir(parents=True); json_write(out/'plan.json',plan)
    started=time.perf_counter()
    for job in plan['jobs']:
        folder=out/job['group']; command=list(job['command'])
        if (folder/'complete.json').exists(): continue
        if folder.exists():
            checkpoint=folder/'checkpoints/latest/codec.pt'
            if not checkpoint.exists(): raise ValueError('no resumable checkpoint: '+str(folder))
            at=command.index('--init-semantic'); del command[at:at+2]
            command+=['--resume',str(checkpoint)]
        print(f"Starting {job['group']}; log: {out/(job['group']+'.log')}",flush=True)
        with (out/(job['group']+'.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            json_write(out/'status.json',{'status':'failed','group':job['group'],'returncode':result.returncode})
            raise RuntimeError(f"{job['group']} failed; inspect {out/(job['group']+'.log')}")
        summarize(out,args.groups,plan['teacher'])
    result=summarize(out,args.groups,plan['teacher'])
    json_write(out/'complete.json',{'status':'complete','groups':args.groups,'elapsed_seconds':time.perf_counter()-started})
    print(f'Complete: {out}/comparison/summary.json',flush=True)
    return result


if __name__=='__main__': main(parser().parse_args())
