"""Source-backed multi-SNR curves; validation history and test endpoints stay separate."""
import csv
import json
from pathlib import Path
import statistics


def aggregate(rows):
    groups={}
    for r in rows:
        if r['path']!='awgn' or r['layout'] not in ('1','2','3'): continue
        groups.setdefault((r['scene'],r['snr_db'],r['layout']),[]).append(r)
    result=[]
    for (scene,snr,q),values in sorted(groups.items()):
        row=dict(scene=scene,snr_db=snr,layout=q,scope=values[0]['scope'],trials=len(values),
                 total_channel_uses=values[0]['total_channel_uses'])
        for metric in ('source_psnr','photo_psnr','source_ssim','photo_ssim','source_lpips','photo_lpips','latent_mse'):
            v=[r.get(metric,r.get('diagnostics',{}).get(metric)) for r in values]
            if any(x is None for x in v): continue
            row[metric]=statistics.mean(v); row[metric+'_std']=statistics.pstdev(v)
        result.append(row)
    return result


def write_figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); destination=out/'figures'; destination.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    history=[]; endpoints=[]
    for folder in sorted((out/'evaluation').glob('monitor_*')):
        if not (folder/'progress.json').exists(): continue
        progress=read(folder/'progress.json')
        for file in sorted(folder.glob('*/summary.json')):
            for row in aggregate(read(file)['rows']): history.append(dict(row,**progress))
    for label in ('initial','final'):
        for file in sorted((out/'evaluation'/label).glob('*/summary.json')):
            for row in aggregate(read(file)['rows']): endpoints.append(dict(row,endpoint=label))
    for name,rows in (('training_curves',history),('snr_endpoints',endpoints)):
        if not rows: continue
        fields=sorted(set().union(*(r.keys() for r in rows)))
        with (destination/(name+'.csv')).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

    def save(fig,name):
        fig.tight_layout();fig.savefig(destination/(name+'.png'),dpi=180)
        fig.savefig(destination/(name+'.pdf'));plt.close(fig)

    for scene in sorted({r['scene'] for r in history}):
        rows=[r for r in history if r['scene']==scene]
        for x,label,suffix in (('step','Optimizer updates','updates'),
                               ('optimization_seconds','Optimization time (minutes; excludes evaluation)','time')):
            fig,axes=plt.subplots(2,3,figsize=(13,7))
            for column,q in enumerate(('1','2','3')):
                for axis,metric in zip(axes[:,column],('source_psnr','photo_psnr')):
                    for snr in sorted({r['snr_db'] for r in rows}):
                        subset=sorted([r for r in rows if r['layout']==q and r['snr_db']==snr],key=lambda r:r['step'])
                        xs=[r[x]/(60 if x=='optimization_seconds' else 1) for r in subset]
                        axis.plot(xs,[r[metric] for r in subset],marker='.',label=f'{snr:g} dB')
                    axis.set(title=f'{scene} q{q}',xlabel=label,ylabel=metric+' (dB)');axis.grid(alpha=.25)
                    axis.legend(fontsize=7)
            fig.suptitle('Fixed validation cameras and paired noise; reporting only')
            save(fig,f'{scene}_training_{suffix}')
    for scene in sorted({r['scene'] for r in endpoints}):
        rows=[r for r in endpoints if r['scene']==scene]
        for reference in ('source','photo'):
            fig,axes=plt.subplots(1,3,figsize=(13,3.8))
            for axis,metric in zip(axes,(reference+'_psnr',reference+'_ssim',reference+'_lpips')):
                for q,color in zip(('1','2','3'),('tab:blue','tab:orange','tab:green')):
                    for endpoint,style in (('initial','--'),('final','-')):
                        subset=sorted([r for r in rows if r['layout']==q and r['endpoint']==endpoint and metric in r],key=lambda r:r['snr_db'])
                        if subset: axis.plot([r['snr_db'] for r in subset],[r[metric] for r in subset],
                            style,marker='.',color=color,label=f'q{q} {endpoint}')
                axis.set(xlabel='Channel SNR (dB)',ylabel=metric);axis.grid(alpha=.25);axis.legend(fontsize=7)
            fig.suptitle(scene+': fixed test cameras; reliable-side assumption')
            save(fig,f'{scene}_snr_{reference}')
    for file in (out/'actual_delivery').glob('*/summary.json'):
        results=read(file)['results'];fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        for q in sorted({r['method'] for r in results}):
            rows=sorted([r for r in results if r['method']==q],key=lambda r:r['snr_db'])
            axes[0].plot([r['snr_db'] for r in rows],[r['success_rate'] for r in rows],marker='o',label='q'+q)
            axes[1].plot([r['snr_db'] for r in rows],[r['total_channel_uses']/1e6 for r in rows],marker='o',label='q'+q)
        axes[0].set(ylabel='Complete delivery success rate',ylim=(-.02,1.02))
        axes[1].set(ylabel='Complex channel uses (M)')
        for axis in axes: axis.set_xlabel('Channel SNR (dB)');axis.grid(alpha=.25);axis.legend()
        fig.suptitle(file.parent.name+': actual LDPC/QAM/AWGN; failures charged; no ARQ')
        save(fig,file.parent.name+'_actual_delivery')
    (destination/'definitions.json').write_text(json.dumps(dict(
        history_scope='fixed validation cameras, paired noise, no checkpoint selection',
        endpoint_scope='original test cameras, paired initial/final noise',
        time_axis='sum of optimizer-step wall time; excludes preparation, evaluation and digital delivery',
        initial_condition='frozen pretrained 10dB condition; actual channel SNR varies',
        final_condition='matched actual and conditioned SNR',
        plotted_values='trial means; per-trial values and per-view metrics retained in evaluation JSON',
        cost_unit='complex channel uses, actual files including CRC/header/padding'),indent=2)+'\n',encoding='utf8')
