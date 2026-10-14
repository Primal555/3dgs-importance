"""Truck 2x2 spatial-context controls: baseline / encoder / decoder / both."""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from .modular_experiment import parser as training_parser, json_write, PHASES
from .modular_training import MODULES
from .modular_plots import read_rows

GROUPS = ('baseline','encoder','decoder','both')


def parser():
    p = training_parser()
    p.description = __doc__
    p.add_argument('--groups',nargs='+',choices=GROUPS,default=list(GROUPS))
    p.add_argument('--resume-suite',action='store_true',help='continue incomplete arms from latest; skip completed arms')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--summarize-only',action='store_true')
    return p


def build_plan(args):
    if len(set(args.groups))!=len(args.groups): raise ValueError('duplicate control groups')
    if args.resume or args.init_semantic or args.with_direct_control:
        raise ValueError('context controls start matched random models; use --resume-suite for continuation')
    keys=['run','manifest','device','decoder_geometry','latent_dim','context_centers','context_neighbors','context_radii',
          'blocks_per_batch','train_snrs','channel_latent_weight','validate_every','save_every','seed','cpu_threads']
    keys += [p+'_steps' for p in PHASES]+[p+'_lr' for p in MODULES]
    common=[]
    for key in keys:
        value=getattr(args,key)
        if value is not None:
            common += ['--'+key.replace('_','-')]+[str(v) for v in (value if isinstance(value,(tuple,list)) else [value])]
    jobs=[{'group':group,'command':[sys.executable,'-u','-m','gaussian_jscc.modular_experiment',*common,
                                  '--out',str(Path(args.out)/group),'--context-variant',group]} for group in args.groups]
    return {'version':1,'groups':args.groups,'jobs':jobs,'scene':'truck','seed':args.seed,
            'random_initialization':True,'allocator_training':False,'matched_optimizer_updates':True,
            'capacity_matched':False,'flop_matched':False,
            'geometry_metric':'world Euclidean up to a single scalar; scene bbox from metadata',
            'neighborhood_scope':'within each fixed Morton block; no cross-block halo',
            'mixed':'fixed random positive tiers for validation; not a learned allocation'}


def comparison_signature(record,ignore_config_keys=()):
    cfg={k:v for k,v in record.get('codec_config',{}).items() if not k.startswith('context_') and k not in ignore_config_keys}
    return {**{k:record.get(k) for k in ('scene','phase_budgets','module_lrs','train_snrs','seed','training_objective')},
            'codec_config':cfg}


def summarize(out, groups, *, ignore_config_keys=()):
    """Read saved metrics; no scene rendering, no compression or retraining."""
    out=Path(out); target=out/'comparison'; target.mkdir(parents=True,exist_ok=True)
    records=[]; gains=[]; costs=[]; configurations={}; signatures=[]; states={}
    for group in groups:
        folder=out/group
        states[group]='complete' if (folder/'complete.json').exists() else 'incomplete'
        if not (folder/'training.json').exists(): continue
        cfg=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        configurations[group]=cfg; signatures.append(comparison_signature(cfg,ignore_config_keys))
        rows=read_rows(folder/'metrics/test.jsonl')
        # Exact-resume after final evaluation can leave duplicate endpoint rows.
        rows=list({(r['path'],r['layout'],r['actual_snr']):r for r in rows}.values())
        for r in rows:
            records.append(dict(r,group=group))
        for path,snr in dict.fromkeys((r['path'],r['actual_snr']) for r in rows if r['path']!='clean'):
            by_tier={r['layout']:r for r in rows if r['path']==path and r['actual_snr']==snr}
            if not all(k in by_tier for k in ('1','2','3')): continue
            gains.append({'group':group,'path':path,'snr':snr,
                **{f'q{b}_minus_q{a}_db':by_tier[str(b)]['source_psnr']-by_tier[str(a)]['source_psnr'] for a,b in ((1,2),(2,3))},
                **{f'photo_q{b}_minus_q{a}_db':by_tier[str(b)]['photo_psnr']-by_tier[str(a)]['photo_psnr'] for a,b in ((1,2),(2,3))}})
        losses=read_rows(folder/'metrics/loss.jsonl')
        for phase in PHASES:
            selected=[r for r in losses if r['phase']==phase]
            if not selected: continue
            costs.append({'group':group,'phase':phase,'updates':len(selected),
                          'parameters':sum(cfg['parameters'].values()),
                          'mean_step_seconds':sum(r['step_seconds'] for r in selected)/len(selected),
                          'peak_memory_allocated_bytes':max((r.get('peak_memory_allocated_bytes') or 0 for r in selected))})
    if signatures and any(s!=signatures[0] for s in signatures[1:]):
        raise ValueError('controls are not matched: scene/split/objective/budget/seed/model dimensions differ')
    summary={'status_by_group':states,'enhancement_gains':gains,'costs':costs,'test_records':records,
             'comparison':'same updates/data/seed; parameter count and compute differ',
             'interpretation':'judge absolute q1/q2/q3 quality and gains separately; larger gaps alone are not success'}
    json_write(target/'summary.json',summary)
    keys=['group','path','layout','actual_snr','source_psnr','photo_psnr','source_ssim','photo_ssim',
          'latent_mse','latent_nmse','symbols_per_gaussian','payload_complex_symbols','xyz_bytes','metadata_bytes',
          'total_channel_uses','views','trials','condition_policy']
    keys += ['source_color_r_bias','source_color_g_bias','source_color_b_bias',
             'source_color_rg_error_rmse','source_color_gb_error_rmse']
    with (target/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore'); writer.writeheader(); writer.writerows(records)
    plot_comparison(out,groups,records,gains,costs)
    return summary


def plot_comparison(out,groups,records,gains,costs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=out/'comparison'
    def save(fig,name):
        fig.savefig(target/name,dpi=150); plt.close(fig)
    if records:
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for ax,path in zip(axes,('awgn','finite_noiseless')):
            for group in groups:
                rows=sorted((r for r in records if r['group']==group and r['path']==path and
                             r['actual_snr']==10 and r['layout'] in ('1','2','3')),key=lambda r:int(r['layout']))
                if rows: ax.plot([r['symbols_per_gaussian'] for r in rows],[r['source_psnr'] for r in rows],marker='o',label=group)
            ax.set(title='10 dB AWGN' if path=='awgn' else 'Finite noiseless channel',
                   xlabel='Complex symbols per Gaussian',ylabel='Source PLY PSNR (dB)',xticks=[4,12,24]); ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=8)
        save(fig,'quality_by_tier.png')
    if gains:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,path in zip(axes,('awgn','finite_noiseless')):
            rows=[r for r in gains if r['path']==path and r['snr']==10]
            x=list(range(len(rows)))
            ax.bar([v-.18 for v in x],[r['q2_minus_q1_db'] for r in rows],width=.36,label='q2 - q1')
            ax.bar([v+.18 for v in x],[r['q3_minus_q2_db'] for r in rows],width=.36,label='q3 - q2')
            ax.set(xticks=x,xticklabels=[r['group'] for r in rows],ylabel='PSNR improvement (dB)',title=path)
            ax.axhline(0,color='gray',linewidth=.7); ax.legend(); ax.grid(axis='y',alpha=.2)
        save(fig,'enhancement_gains.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained'); has_curves=False
    for group in groups:
        rows=read_rows(out/group/'metrics/validation.jsonl')
        for ax,path,layout in ((axes[0],'clean','clean'),(axes[1],'awgn','3')):
            by_step={r['step']:r for r in rows if r['path']==path and r['layout']==layout and r['actual_snr'] in (None,10)}
            selected=[by_step[s] for s in sorted(by_step)]
            if selected:
                ax.plot([r['step'] for r in selected],[r['source_psnr'] for r in selected],label=group); has_curves=True
            ax.set(title='Clean representation' if path=='clean' else '10 dB q3',xlabel='Optimizer updates',ylabel='Validation PSNR (dB)'); ax.grid(alpha=.2)
    if has_curves:
        for ax in axes:
            if ax.lines: ax.legend(fontsize=8)
        save(fig,'learning_curves.png')
    else: plt.close(fig)
    if costs:
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for i,group in enumerate(groups):
            rows={r['phase']:r for r in costs if r['group']==group}
            ax.bar([j+(i-(len(groups)-1)/2)*.18 for j in range(len(PHASES))],
                   [rows.get(p,{}).get('mean_step_seconds',0) for p in PHASES],width=.18,label=group)
        ax.set(xticks=list(range(len(PHASES))),xticklabels=PHASES,ylabel='Mean seconds per optimizer update'); ax.legend(); ax.grid(axis='y',alpha=.2)
        save(fig,'training_cost.png')


def main(args):
    plan=build_plan(args); out=Path(args.out).resolve()
    if args.dry_run:
        print(json.dumps(plan,indent=2)); return plan
    if args.summarize_only: return summarize(out,args.groups)
    if out.exists():
        if not args.resume_suite: raise FileExistsError('use a NEW suite directory or --resume-suite')
        previous=json.loads((out/'plan.json').read_text(encoding='utf-8'))
        if previous!=plan: raise ValueError('resume suite plan differs; retain original paths, options and groups')
    else:
        out.mkdir(parents=True); json_write(out/'plan.json',plan)
    started=time.perf_counter()
    for job in plan['jobs']:
        folder=out/job['group']; command=list(job['command'])
        if (folder/'complete.json').exists(): continue
        if folder.exists():
            checkpoint=folder/'checkpoints/latest/codec.pt'
            if not checkpoint.exists(): raise ValueError(f'no resumable checkpoint in {folder}; use a new suite directory')
            command+=['--resume',str(checkpoint)]
        print(f"Starting {job['group']}; log: {out/(job['group']+'.log')}",flush=True)
        with (out/(job['group']+'.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            json_write(out/'status.json',{'status':'failed','group':job['group'],'returncode':result.returncode})
            raise RuntimeError(f"{job['group']} failed; inspect {out/(job['group']+'.log')}")
        summarize(out,args.groups)
    summary=summarize(out,args.groups)
    json_write(out/'complete.json',{'status':'complete','groups':args.groups,'elapsed_seconds':time.perf_counter()-started})
    print(f'Complete: {out}/comparison/summary.json',flush=True)
    return summary


if __name__=='__main__': main(parser().parse_args())
