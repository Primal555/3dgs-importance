"""Equal-scene metrics and compact plots; never pools unlike tiers/references."""
import math
from collections import defaultdict
from pathlib import Path

from .modular_plots import read_rows


def macro_metrics(measured):
    groups = defaultdict(dict)
    for scene,rows in measured.items():
        for row in rows:
            key = (row['step'],row['phase'],row['path'],row['layout'],row['actual_snr'])
            if scene in groups[key]: raise ValueError('duplicate scene/path in macro snapshot')
            groups[key][scene] = row
    result = []
    for (step,phase,path,layout,snr),rows in groups.items():
        if set(rows) != set(measured): raise ValueError('macro metrics require every scene on the same path')
        keys = ('source_psnr','source_ssim','source_lpips','photo_psnr','photo_ssim','photo_lpips',
                'latent_mse','latent_nmse')
        metrics = {}
        for key in keys:
            values = [r.get(key) for r in rows.values()]
            if all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
                metrics[key] = sum(values)/len(values)
        result.append(dict(step=step,phase=phase,path=path,layout=layout,actual_snr=snr,
            scenes=list(rows),scene_count=len(rows),aggregation='arithmetic mean of per-scene view means',
            **metrics))
    return result


def repair_macro_logs(out, step):
    # Archive rather than silently discard metrics ahead of the checkpoint.
    import json
    import time
    for name in ('macro_validation','macro_test'):
        path = Path(out)/'metrics'/f'{name}.jsonl'
        if not path.exists(): continue
        lines = path.read_text(encoding='utf-8').splitlines()
        newer = [line for line in lines if json.loads(line)['step']>step]
        if newer:
            path.with_name(f'{name}.uncheckpointed_{time.time_ns()}.jsonl').write_text('\n'.join(newer)+'\n',encoding='utf-8')
            path.write_text('\n'.join(line for line in lines if json.loads(line)['step']<=step)+'\n',encoding='utf-8')


def plot_run(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .modular_training import PHASE_MODULES
    from .modular_experiment import json_write
    out = Path(out); charts = out/'charts'; charts.mkdir(parents=True,exist_ok=True)
    losses = read_rows(out/'metrics/loss.jsonl')
    scenes = list(dict.fromkeys(r['scene'] for r in losses))
    summaries = {}
    fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,phase in zip(axes.flat,PHASE_MODULES):
        summaries[phase] = {}
        for scene in scenes:
            rows = [r for r in losses if r['scene']==scene and r['phase']==phase]
            summaries[phase][scene] = dict(updates=len(rows),total_step_seconds=sum(r['step_seconds'] for r in rows))
            # Attribute-response and every-fourth color objectives have different units.
            for objective in dict.fromkeys(r['objective'] for r in rows):
                part = [r for r in rows if r['objective']==objective]
                ax.plot([r['scene_step'] for r in part],[r['loss'] for r in part],linewidth=.8,
                        label=f'{scene}/{objective}')
        ax.set(title=phase,xlabel='Real updates per scene in phase',ylabel='Phase-specific objective')
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=6)
    fig.savefig(charts/'objectives_by_scene.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for root,label in [(out/'scenes'/s,s) for s in scenes]+[(out,'macro')]:
        rows = read_rows(root/'metrics'/('macro_validation.jsonl' if label=='macro' else 'validation.jsonl'))
        for row_index,reference in enumerate(('source','photo')):
            for ax,metric in zip(axes[row_index],('psnr','ssim','lpips')):
                for path,layout in (('clean','clean'),('awgn','1'),('awgn','2'),('awgn','3')):
                    series = {r['step']:r for r in rows if r['path']==path and r['layout']==layout and reference+'_'+metric in r}
                    part = [series[k] for k in sorted(series)]
                    if part: ax.plot([r['step'] for r in part],[r[reference+'_'+metric] for r in part],
                                     linewidth=1.8 if label=='macro' else .8,label=f'{label}/{path}/{layout}')
                ax.set(title=f'{reference} {metric}',xlabel='Global real optimizer updates')
                ax.grid(alpha=.2)
                if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=5,ncol=2)
    fig.savefig(charts/'quality_by_scene_and_macro.png',dpi=150); plt.close(fig)
    json_write(out/'metrics/phase_summary.json',summaries)
