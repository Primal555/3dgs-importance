"""Plots from saved metrics only; no rerendering or coordinate compression."""
import argparse
import json
import math
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def quality_series(rows,path,layout,key):
    """Last measured value per step; missing legacy metrics are NOT zero."""
    selected={r['step']:r for r in rows if r['path']==path and r['layout']==layout
              and r['actual_snr'] in (None,10) and _finite_metric(r,key)}
    return [selected[k] for k in sorted(selected)]


def _finite_metric(row,key):
    return isinstance(row.get(key),(int,float)) and math.isfinite(row[key])


def _plot_perceptual_quality(validation,final,save):
    """Chart contract: observed updates/SNR vs quality, separate references.

    Six validation panels separate clean/noiseless/AWGN rather than mixing
    nine traces. Tiers keep explicit categorical colors and distinct markers;
    sparse checkpoint series use dots, not an invented smooth training curve.
    Noise error bars are trial view-mean std, never a training-seed CI.
    """
    import matplotlib.pyplot as plt
    styles={'1':('#356d9b','o'),'2':('#d88931','s'),'3':('#7b8139','^'),
            'mixed':('#b06a8e','D'),'clean':('#444444','x')}
    for metric,label in (('ssim','SSIM (higher is better)'),('lpips','LPIPS/VGG v0.1 (lower is better)')):
        keys=[ref+'_'+metric for ref in ('source','photo')]
        if any(_finite_metric(r,key) for r in validation for key in keys):
            fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
            for row,path in enumerate(('clean','finite_noiseless','awgn')):
                layouts=('clean',) if path=='clean' else ('1','2','3','mixed')
                for ax,ref,key in zip(axes[row],('Source PLY','Photos'),keys):
                    for layout in layouts:
                        series=quality_series(validation,path,layout,key)
                        if not series: continue
                        color,marker=styles[layout]
                        ax.plot([r['step'] for r in series],[r[key] for r in series],color=color,marker=marker,
                                linestyle='-' if len(series)>=8 else 'None',markersize=4,
                                label='clean' if layout=='clean' else 'q'+layout if layout!='mixed' else 'mixed')
                    ax.set(title=f'{path} vs {ref}',xlabel='Global real optimizer updates',ylabel=label)
                    ax.grid(alpha=.2)
                    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8,ncol=2)
                    else: ax.text(.5,.5,'No recorded metric',ha='center',va='center',transform=ax.transAxes)
            fig.suptitle('Fixed validation views; physical AWGN 10 dB; dots = measured checkpoints')
            save(fig,f'validation_{metric}.png')
        awgn=[r for r in final if r['path']=='awgn']
        if any(_finite_metric(r,key) for r in awgn for key in keys):
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
            for ax,ref,key in zip(axes,('Source PLY','Photos'),keys):
                for layout in ('1','2','3','mixed'):
                    rows=sorted((r for r in awgn if r['layout']==layout and _finite_metric(r,key)),
                                key=lambda r:r['actual_snr'])
                    if not rows: continue
                    color,marker=styles[layout]
                    ax.errorbar([r['actual_snr'] for r in rows],[r[key] for r in rows],
                                yerr=[r.get(key+'_std_trials',0.) for r in rows],color=color,marker=marker,
                                capsize=3,label='q'+layout if layout!='mixed' else 'mixed')
                for path in ('clean','finite_noiseless'):
                    reference=next((r for r in final if r['path']==path and _finite_metric(r,key)
                                    and (path=='clean' or r['layout']=='3')),None)
                    if reference:
                        ax.axhline(reference[key],color='#444444',linestyle=':' if path=='clean' else '--',
                                   label='clean' if path=='clean' else 'noiseless q3')
                ax.set(title=f'vs {ref}',xlabel='Physical AWGN SNR (dB)',ylabel=label)
                ax.grid(alpha=.2)
                if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
            policies=', '.join(dict.fromkeys(r.get('condition_policy','matched') for r in awgn))
            fig.suptitle(f'Test views; conditions: {policies}; bars = noise-trial view-mean std')
            save(fig,f'{metric}_snr.png')


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
            for objective in dict.fromkeys(r.get('objective','Total objective') for r in rows):
                selected=[r for r in rows if r.get('objective','Total objective')==objective]
                ax.plot([r['phase_step'] for r in selected],[r['loss'] for r in selected],linewidth=.8,label=objective)
            if phase=='channel_render':
                for key,label in [('render_loss','Render distortion'),('weighted_latent_loss','Weighted latent MSE')]:
                    selected=[r for r in rows if key in r]
                    if selected: ax.plot([r['phase_step'] for r in selected],[r[key] for r in selected],linewidth=.8,label=label)
            for key,label in [('image_mse','Image MSE'),('image_aux_weighted','Weighted image perceptual loss')]:
                selected=[r for r in rows if key in r]
                if selected:ax.plot([r['phase_step'] for r in selected],[r[key] for r in selected],linewidth=.8,label=label)
            if rows: ax.legend(fontsize=8)
            ax.set(title=phase,xlabel='Actual phase optimizer updates',ylabel='Objective value (phase-specific)')
            ax.grid(alpha=.2)
        fig.suptitle('Phase-specific objectives; communication render loss components shown separately')
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
        quality_rows={r['step']:r for r in validation if r['path']=='clean'
                      and all(k in r for k in ('source_psnr','photo_psnr','source_ssim','photo_ssim'))}
        if quality_rows:
            series=[quality_rows[k] for k in sorted(quality_rows)]
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
            for ax,metric,label in zip(axes,('psnr','ssim'),('PSNR (dB)','SSIM')):
                for reference in ('source','photo'):
                    ax.plot([r['step'] for r in series],[r[reference+'_'+metric] for r in series],label=reference)
                ax.set(xlabel='Actual optimizer updates',ylabel=label); ax.grid(alpha=.2); ax.legend(fontsize=8)
            fig.suptitle('Clean reconstruction vs source PLY and photos; fixed held-out validation cameras')
            save(fig,'clean_quality.png')
        color_rows={r['step']:r for r in validation if r['path']=='clean' and 'source_color_rg_error_rmse' in r}
        if color_rows:
            series=[color_rows[k] for k in sorted(color_rows)]
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
            for key,label in [('rg_error_rmse','R-G error RMSE'),('gb_error_rmse','G-B error RMSE')]:
                axes[0].plot([r['step'] for r in series],[r['source_color_'+key] for r in series],label=label)
            for channel in 'rgb':
                axes[1].plot([r['step'] for r in series],[r[f'source_color_{channel}_bias'] for r in series],label=channel.upper()+' bias')
            axes[1].axhline(0,color='gray',linestyle='--',linewidth=.8)
            for ax in axes:
                ax.set(xlabel='Actual optimizer updates',ylabel='RGB units (0-1 scale)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
            fig.suptitle('Clean color errors vs source PLY; bias may cancel, not perceptual Delta-E')
            save(fig,'color_errors.png')
        latent_rows=[r for r in validation if r['path']!='clean' and r['actual_snr']==10 and 'latent_nmse' in r]
        if latent_rows:
            fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
            for path,layout in dict.fromkeys((r['path'],r['layout']) for r in latent_rows):
                if path=='finite_noiseless' and layout!='3': continue
                by_step={r['step']:r for r in latent_rows if r['path']==path and r['layout']==layout}
                series=[by_step[k] for k in sorted(by_step)]
                for ax,key in zip(axes,('latent_mse','latent_nmse')):
                    ax.plot([r['step'] for r in series],[r[key] for r in series],label=f'{path}/{layout}',
                            linestyle='--' if path=='finite_noiseless' else '-')
            for ax,key in zip(axes,('latent_mse','latent_nmse')):
                ax.set(xlabel='Global real updates',ylabel=key)
                ax.set_yscale('symlog',linthresh=1e-6); ax.grid(alpha=.2); ax.legend(fontsize=8)
            fig.suptitle('Fixed validation points; NMSE = latent MSE / target mean-square power')
            save(fig,'latent_reconstruction.png')
    if any(r['path']=='awgn' for r in final):
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
        policies=', '.join(dict.fromkeys(r.get('condition_policy','matched') for r in final if r['path']=='awgn'))
        fig.suptitle(f'All test views; SNR conditions: {policies}; error bars = noise-trial view-mean standard deviation')
        save(fig,'psnr_snr.png')
    _plot_perceptual_quality(validation,final,save)
    return paths


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('out')
    print('\n'.join(map(str,plot_run(p.parse_args().out))))
