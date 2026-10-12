"""Plots from saved metrics only; no rerendering or coordinate compression."""
import argparse
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def plot_run(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .modular_training import MODULES,PHASE_MODULES
    out=Path(out); charts=out/'charts'; charts.mkdir(parents=True,exist_ok=True)
    losses=read_rows(out/'metrics/loss.jsonl'); validation=read_rows(out/'metrics/validation.jsonl'); final=read_rows(out/'metrics/test.jsonl')
    paths=[]
    def save(fig,name):
        path=charts/name; fig.savefig(path,dpi=150); plt.close(fig); paths.append(path)
    if losses:
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for ax,phase in zip(axes.flat,PHASE_MODULES):
            rows=[r for r in losses if r['phase']==phase]
            if rows: ax.plot([r['phase_step'] for r in rows],[r['loss'] for r in rows],linewidth=.8)
            ax.set(title=phase,xlabel='Actual phase optimizer updates',ylabel='Local response MSE' if phase.endswith('bootstrap') else 'Source PLY RGB MSE')
            ax.grid(alpha=.2)
        fig.suptitle('Separate objectives: phase boundaries reset Adam, no latent auxiliary loss')
        save(fig,'objectives_by_phase.png')
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for ax,name in zip(axes.flat,MODULES):
            ax.plot([r['step'] for r in losses],[r.get('gradient_groups',{}).get(name,{}).get('before',0) for r in losses],label='gradient norm')
            ax.plot([r['step'] for r in losses],[r.get('updates',{}).get(name,{}).get('update_norm',0) for r in losses],label='parameter update norm')
            ax.set(title=name,xlabel='Global real updates',yscale='symlog'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        save(fig,'module_gradients_updates.png')
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        for phase in PHASE_MODULES:
            rows=[r for r in losses if r['phase']==phase]
            if rows: ax.plot([r['step'] for r in rows],[r['step_seconds'] for r in rows],label=phase)
        ax.set(xlabel='Global real updates',ylabel='Seconds per optimizer update'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        save(fig,'step_seconds.png')
    if validation:
        fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
        for path,layout in dict.fromkeys((r['path'],r['layout']) for r in validation):
            rows=[r for r in validation if r['path']==path and r['layout']==layout and r['actual_snr'] in (None,10)]
            # Each stage boundary may record the same weights; show one value/step.
            by_step={r['step']:r for r in rows}
            series=[by_step[k] for k in sorted(by_step)]
            if series: ax.plot([r['step'] for r in series],[r['source_psnr'] for r in series],label=f'{path}/{layout}')
        ax.set(xlabel='Global real updates',ylabel='Mean per-view source PSNR (dB)',title='Same fixed validation views; clean remeasured every time')
        ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=3)
        save(fig,'validation_paths.png')
    if final:
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,metric in zip(axes,('source_psnr','photo_psnr')):
            for layout in ('1','2','3','mixed'):
                rows=sorted((r for r in final if r['path']=='awgn' and r['layout']==layout),key=lambda r:r['actual_snr'])
                if rows and metric in rows[0]:
                    ax.errorbar([r['actual_snr'] for r in rows],[r[metric] for r in rows],
                        yerr=[r.get(metric+'_std_trials',0) for r in rows],marker='o',capsize=3,label=layout)
            for path in ('clean','finite_noiseless'):
                for r in final:
                    if r['path']==path and (path=='clean' or r['layout']=='3') and metric in r:
                        ax.axhline(r[metric],linestyle='--',label=path+('/q3' if path!='clean' else ''))
            ax.set(xlabel='Actual AWGN SNR (dB)',ylabel=metric+' (dB)'); ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
        fig.suptitle('All test views; error bars = standard deviation across noise-trial view means')
        save(fig,'psnr_snr.png')
    return paths


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('out')
    print('\n'.join(map(str,plot_run(p.parse_args().out))))
