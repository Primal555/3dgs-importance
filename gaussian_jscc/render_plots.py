"""Static experiment figures for the render-first trainer.

Chart contract: do held-out image metrics improve at a fixed channel/layout?
PNG/SVG in charts/, CSV retains trial/view counts, payload and phase. Phase
objectives have independent axes; validation compares fixed observations across
phases. >=8 checks use lines; sparse checks use markers, not implied trends.
Tier identity uses blue/gold/orange/olive plus distinct markers, mask neutral.
Optimizer diagnostics use separate axes, not scalar-loss-as-gradient claims.
"""
from pathlib import Path
from .plots import _read_jsonl, _plt, _finish, _save, _manifest, _write_csv, _trace


def plot_render_training(training_dir, output_dir=None):
    root = Path(training_dir)
    out = Path(output_dir) if output_dir else root/'charts'
    out.mkdir(parents=True,exist_ok=True)
    rows = _read_jsonl(root/'loss.jsonl')
    plt = _plt()
    charts = []
    phases = list(dict.fromkeys(r['phase'] for r in rows))
    fig,axes = plt.subplots(1,len(phases),figsize=(6*len(phases),4),squeeze=False)
    fig.suptitle('Training objectives by phase | independent scales')
    for ax,phase in zip(axes[0],phases):
        subset = [r for r in rows if r['phase']==phase]
        if len(subset)<8:
            ax.plot([r['step'] for r in subset],[r['loss'] for r in subset],linestyle='none',marker='o',color='#2F6B9A',label='Objective')
        else:
            _trace(ax,subset,'loss','Objective','#2F6B9A')
        if phase=='joint':
            _trace(ax,subset,'image_mse','Image MSE','#D96C2F','--')
            _trace(ax,subset,'rate_loss','Payload penalty','#59636E',':')
        ax.set(title=phase,xlabel='Optimization step',ylabel='Normalized-feature SmoothL1' if phase=='bootstrap' else 'RGB MSE + rate (joint only)')
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    charts += _finish(fig,out/'training_objectives')
    fig,axes = plt.subplots(1,3,figsize=(16,4.5))
    fig.suptitle('Optimizer diagnostics | measured gradients and actual Adam updates')
    colors = ('#2F6B9A','#D96C2F','#59636E')
    for phase,color in zip(phases,colors):
        subset = [r for r in rows if r['phase']==phase]
        for ax,key in zip(axes,('grad_norm','update_norm','step_seconds')):
            _trace(ax,subset,key,phase,color)
    for ax,title in zip(axes,('Gradient L2 before clipping','Parameter update L2','Training step seconds (validation excluded)')):
        ax.set(title=title,xlabel='Optimization step',yscale='symlog')
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        else:
            ax.text(.5,.5,'Not recorded',transform=ax.transAxes,ha='center')
    charts += _finish(fig,out/'optimization')
    validation = root/'validation.jsonl'
    if validation.exists():
        records = _read_jsonl(validation)
        # A phase boundary can be evaluated twice at identical weights. Keep
        # the latest observation per step/layout, while CSV retains phase.
        indexed = {}
        for r in records:
            for entry in r['layouts']:
                indexed[r['step'],entry['layout']] = {k:v for k,v in entry.items() if k!='views'} | {
                    'step':r['step'],'phase':r['phase'],'snr':r['snr'],'channel':r['channel'],
                    'trials':r['trials'],'validation_views':r['validation_views']}
        flattened = list(indexed.values())
        _write_csv(out/'validation_metrics.csv',flattened,list(flattened[0]))
        fig,axes = plt.subplots(2,3,figsize=(16,8))
        first=records[0]
        fig.suptitle(f'Fixed validation | {first["channel"]}, {first["snr"]:g} dB, '
                     f'{first["validation_views"]} held-out views x {first["trials"]} noise trials')
        keys=('source_mse','source_psnr','source_ssim','photo_psnr','photo_ssim','xyz_rmse_retained')
        titles=('Source-render RGB MSE (lower better)','Source-render PSNR (dB)','Source-render SSIM',
                'Photo PSNR (dB)','Photo SSIM','XYZ RMSE: diagnostic only, scene units')
        styles={'1':('#2F6B9A','o'),'2':('#D8A72E','s'),'3':('#D96C2F','^'),
                'mixed':('#737A36','D'),'mask':('#59636E','x')}
        for label,(color,marker) in styles.items():
            subset=sorted([r for r in flattened if r['layout']==label],key=lambda r:r['step'])
            if not subset:
                continue
            for ax,key,title in zip(axes.flat,keys,titles):
                points=[r for r in subset if r.get(key) is not None]
                ax.plot([r['step'] for r in points],[r[key] for r in points],color=color,marker=marker,
                        linestyle='-' if len(points)>=8 else 'none',markersize=4,label='q'+label)
                ax.set(title=title,xlabel='Optimization step')
        for ax,key in ((axes[1,0],'reference_photo_psnr'),(axes[1,1],'reference_photo_ssim')):
            ax.axhline(flattened[0][key],color='#59636E',linestyle='--',label='Source PLY vs photo')
        handles,labels=axes[1,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=len(labels),fontsize=9)
        fig.tight_layout(rect=(0,.055,1,.96))
        charts += _save(fig,out/'validation_quality')
        plt.close(fig)
        latest=records[-1]['layouts']
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        fig.suptitle(f'Last validation, step {records[-1]["step"]} | not necessarily best checkpoint')
        for ax,key,title in zip(axes,('symbols_per_source_gaussian','source_psnr'),
                                ('Payload complex symbols / source Gaussian','Source-render PSNR (dB)')):
            ax.bar(['q'+r['layout'] for r in latest],[r[key] for r in latest],color='#2F6B9A')
            ax.set_title(title)
            for i,r in enumerate(latest):
                ax.annotate(f'{r[key]:.2f}',(i,r[key]),xytext=(0,3),textcoords='offset points',ha='center')
            ax.margins(y=.15)
        charts += _finish(fig,out/'last_layout_comparison')
    return _manifest(out,'render_first_training',root,charts,[
        'Bootstrap is initialization, not communication fidelity; do not compare its scale to image MSE.',
        'Validation noise/views/layouts are fixed; sparse histories use markers only.',
        'Source-render and photo metrics have different references. MSE/PSNR use unclipped RGB; SSIM displayed RGB.',
        'Rate chart includes payload only. Reliable metadata overhead is counted by the independent packet benchmark.',
        'These are validation views used for selection, not a final held-out test.'])
