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


def _layout_key(label):
    return (0,int(label)) if label.isdigit() else (1,label)


def _layout_label(entry):
    label = 'q'+entry['layout']
    if entry['layout'].isdigit() and entry.get('symbols_per_source_gaussian') is not None:
        label += f' / {entry["symbols_per_source_gaussian"]:g} symbols'
    return label


def plot_allocation_history(root, out):
    path = root/'allocation_history.jsonl'
    if not path.exists():
        return []
    records = _read_jsonl(path)
    if not records:
        return []
    plt = _plt()
    steps = [r['step'] for r in records]
    rates = records[-1]['rates']
    fig,axes = plt.subplots(1,3,figsize=(17,4.5))
    rows = []
    for i,rate in enumerate(rates):
        label = f'q{i}: {rate} symbols'
        axes[0].plot(steps,[r['hard_tier_shares'][i] for r in records],label=label)
        axes[1].plot(steps,[r['hard_tier_counts'][i] for r in records],label=label)
        axes[1].plot(steps,[r['expected_tier_counts'][i] for r in records],linestyle='--',alpha=.6)
        for r in records:
            rows.append({'step':r['step'],'tier':i,'symbols':rate,'hard_count':r['hard_tier_counts'][i],
                         'hard_share':r['hard_tier_shares'][i],'expected_count':r['expected_tier_counts'][i]})
    axes[0].set(title='Hard deployment shares',ylim=(0,1),xlabel='Optimization step')
    axes[1].set(title='Point counts: solid hard, dashed expected',xlabel='Optimization step')
    axes[2].plot(steps,[r['mean_max_probability'] for r in records],label='Mean max probability')
    axes[2].plot(steps,[r['mean_entropy_nats'] for r in records],label='Mean entropy (nats)')
    axes[2].set(title='Decision confidence (not image quality)',xlabel='Optimization step')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle('Learned per-Gaussian categorical allocation | q0 sends no Gaussian payload/XYZ')
    charts = _finish(fig,out/'allocation_history')
    _write_csv(out/'allocation_history.csv',rows,list(rows[0]))
    last=records[-1]
    if 'existence_decile_tier_counts' in last:
        import numpy as np
        table=np.asarray(last['existence_decile_tier_counts'],dtype=float)
        shares=table/np.maximum(1,table.sum(1,keepdims=True))
        fig,ax=plt.subplots(figsize=(8,5))
        im=ax.imshow(shares,vmin=0,vmax=1,aspect='auto',cmap='Blues',origin='lower')
        ax.set(xticks=range(len(rates)),xticklabels=[f'q{i}' for i in range(len(rates))],
               yticks=range(10),yticklabels=[f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)],
               xlabel='Learned hard tier',ylabel='Original existence probability interval',
               title='Existence prior vs learned allocation | row-normalized counts')
        fig.colorbar(im,ax=ax,label='Fraction within existence interval')
        charts += _finish(fig,out/'existence_vs_allocation')
    return charts


def plot_rate_sweep(records, out):
    """Measured uniform prefixes only; mixed layouts are NOT intermediate rates.

    Connecting segments aid reading, not claims about untrained cutoffs. All
    checkpoints keep their data in CSV; figures include latest phase endpoints.
    """
    plt = _plt()
    indexed = {}
    endpoints = {}
    for record in records:
        endpoints[record['phase']] = record
        for e in record['layouts']:
            if not e['layout'].isdigit():
                continue
            indexed[record['step'],e['layout']] = {
                'step':record['step'],'phase':record['phase'],'tier':e['layout'],
                'complex_symbols_per_gaussian':e['symbols_per_source_gaussian'],
                **{key:e.get(key) for key in ('source_mse','source_psnr','source_ssim',
                   'photo_psnr','photo_ssim','payload_plus_position_uses_per_source_gaussian')},
                'snr':record['snr'],'channel':record['channel'],
                'validation_views':record['validation_views'],'trials':record['trials']}
    rows = sorted(indexed.values(),key=lambda r:(r['step'],r['complex_symbols_per_gaussian']))
    if not rows:
        return []
    _write_csv(out/'rate_distortion.csv',rows,list(rows[0]))
    fig,axes = plt.subplots(1,3,figsize=(16,4.5))
    for phase,record in endpoints.items():
        subset = sorted((e for e in record['layouts'] if e['layout'].isdigit()),
                        key=lambda e:e['symbols_per_source_gaussian'])
        for ax,key in zip(axes,('source_psnr','source_mse','source_ssim')):
            ax.plot([e['symbols_per_source_gaussian'] for e in subset],[e[key] for e in subset],
                    marker='o',label=f'{phase}: step {record["step"]}')
    for ax,title in zip(axes,('PSNR vs Source PLY (dB)','MSE vs Source PLY (lower better)','SSIM vs Source PLY')):
        ax.set(title=title,xlabel='Attribute payload: complex symbols / Gaussian')
        ax.set_xticks(sorted(set(r['complex_symbols_per_gaussian'] for r in rows)))
        ax.legend(fontsize=8)
    fig.suptitle('Learned prefix rate-distortion | phase endpoints; measured cutoffs only')
    charts = _finish(fig,out/'rate_distortion')
    last = records[-1]
    gains = last.get('prefix_gains',[])
    if gains:
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        labels=[f'{g.get("from_complex_symbols",g["from"]):g} -> {g.get("to_complex_symbols",g["to"]):g}'
                if 'from_complex_symbols' in g else f'q{g["from"]} -> q{g["to"]}' for g in gains]
        for ax,key,title in zip(axes,('source_mse_reduction_per_extra_symbol','source_psnr_gain_db'),
                               ('MSE reduction / additional complex symbol per Gaussian','PSNR gain (dB)')):
            values=[g.get(key) for g in gains]
            if any(v is None for v in values):
                ax.text(.5,.5,'Not recorded in this run',transform=ax.transAxes,ha='center')
            else:
                ax.bar(labels,values,color='#2F6B9A')
                ax.tick_params(axis='x',labelrotation=35)
            ax.axhline(0,color='#59636E',linestyle='--')
            ax.set(title=title,xlabel='Adjacent measured prefix lengths')
        fig.suptitle(f'Incremental benefit | step {last["step"]}; paired views/noise, not a monotonicity guarantee')
        charts += _finish(fig,out/'rate_marginal_gain')
    return charts


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
            _trace(ax,subset,'rate_loss','Communication cost penalty','#59636E',':')
        label = ('Isolated Gaussian RGB response MSE' if subset[0].get('bootstrap_objective')=='local-response'
                 else 'Normalized-feature SmoothL1') if phase=='bootstrap' else 'RGB MSE + rate (joint only)'
        ax.set(title=phase,xlabel='Optimization step',ylabel=label)
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
    bootstrap = root/'bootstrap_validation.jsonl'
    if bootstrap.exists():
        checks = _read_jsonl(bootstrap)
        fig,ax = plt.subplots(figsize=(7,4))
        for tier in sorted({e['layout'] for r in checks for e in r['layouts']},key=_layout_key):
            points = [(r['step'], e.get('loss',e.get('feature_loss'))) for r in checks
                      for e in r['layouts'] if e['layout']==tier]
            ax.plot([p[0] for p in points],[p[1] for p in points],label='q'+tier,marker='o',markersize=3,
                    linestyle='-' if len(points)>=8 else 'none')
        ax.set(title='Attribute initialization: held-out block objective',xlabel='Optimization step',
               ylabel='Local response MSE' if checks[0].get('objective')=='local-response' else 'Feature loss')
        ax.legend()
        charts += _finish(fig,out/'bootstrap_validation')
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
        labels=sorted({r['layout'] for r in flattened},key=_layout_key)
        for index,label in enumerate(labels):
            color,marker=styles.get(label,(plt.get_cmap('tab20')(index%20),('o','s','^','D','v','P','X','<')[index%8]))
            subset=sorted([r for r in flattened if r['layout']==label],key=lambda r:r['step'])
            if not subset:
                continue
            for ax,key,title in zip(axes.flat,keys,titles):
                points=[r for r in subset if r.get(key) is not None]
                ax.plot([r['step'] for r in points],[r[key] for r in points],color=color,marker=marker,
                        linestyle='-' if len(points)>=8 else 'none',markersize=4,label=_layout_label(subset[0]))
                ax.set(title=title,xlabel='Optimization step')
        for ax,key in ((axes[1,0],'reference_photo_psnr'),(axes[1,1],'reference_photo_ssim')):
            ax.axhline(flattened[0][key],color='#59636E',linestyle='--',label='Source PLY vs photo')
        handles,labels=axes[1,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=min(5,len(labels)),fontsize=8)
        fig.tight_layout(rect=(0,.10,1,.96))
        charts += _save(fig,out/'validation_quality')
        plt.close(fig)
        latest=records[-1]['layouts']
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        fig.suptitle(f'Last validation, step {records[-1]["step"]} | not necessarily best checkpoint')
        for ax,key,title in zip(axes,('symbols_per_source_gaussian','source_psnr'),
                                ('Payload complex symbols / source Gaussian','Source-render PSNR (dB)')):
            ax.bar([_layout_label(r) for r in latest],[r[key] for r in latest],color='#2F6B9A')
            ax.tick_params(axis='x',labelrotation=45)
            ax.set_title(title)
            for i,r in enumerate(latest):
                ax.annotate(f'{r[key]:.2f}',(i,r[key]),xytext=(0,3),textcoords='offset points',ha='center')
            ax.margins(y=.15)
        charts += _finish(fig,out/'last_layout_comparison')
        gains = [{**g,'step':r['step'],'phase':r['phase']} for r in records
                 for g in r.get('prefix_gains',[])]
        if gains:
            _write_csv(out/'prefix_gains.csv',gains,list(gains[0]))
            fig,axes=plt.subplots(1,2,figsize=(12,4.5))
            fig.suptitle('Progressive prefixes | paired views and symbol noise')
            pairs=list(dict.fromkeys((g['from'],g['to']) for g in gains))
            for index,(lower,upper) in enumerate(pairs):
                color=plt.get_cmap('tab10')(index%10)
                subset=[g for g in gains if g['from']==lower and g['to']==upper]
                for ax,key in zip(axes,('source_psnr_gain_db','paired_view_trial_improved_fraction')):
                    ax.plot([g['step'] for g in subset],[g[key] for g in subset],
                            marker='o',linestyle='-' if len(subset)>=8 else 'none',
                            color=color,label=f'q{lower} -> q{upper}')
                    ax.set_xlabel('Optimization step')
                    ax.legend()
            axes[0].axhline(0,color='#59636E',linestyle='--')
            axes[0].set_ylabel('Source-render PSNR gain (dB); positive is better')
            axes[1].set(ylabel='Fraction of paired observations with lower MSE',ylim=(-.02,1.02))
            charts += _finish(fig,out/'prefix_gains')
        charts += plot_rate_sweep(records,out)
    charts += plot_allocation_history(root,out)
    return _manifest(out,'render_first_training',root,charts,[
        'Bootstrap is initialization, not communication fidelity; do not compare its scale to image MSE.',
        'Validation noise/views/layouts are fixed; sparse histories use markers only.',
        'Source-render and photo metrics have different references. MSE/PSNR use unclipped RGB; SSIM displayed RGB.',
        'Rate chart includes payload only. Reliable metadata overhead is counted by the independent packet benchmark.',
        'These are validation views used for selection, not a final held-out test.'])
