"""Read saved one-pair experiment logs; no scene decoding or rendering."""
import json
from pathlib import Path


def plot_run(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);charts=out/'charts';charts.mkdir(parents=True,exist_ok=True)
    def read(name):
        path=out/'metrics'/name
        return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()] if path.exists() else []
    def save(fig,name):fig.savefig(charts/name,dpi=140);plt.close(fig)
    losses=read('loss.jsonl');validation=read('validation.jsonl')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,obj in zip(axes,('local_response_mse','directional_color_mse_unclipped_v1','source_ply_rgb_mse')):
        rows=[r for r in losses if r['objective']==obj]
        ax.plot([r['step'] for r in rows],[r['loss'] for r in rows]);ax.set(title=obj,xlabel='Optimizer updates');ax.grid(alpha=.2)
    save(fig,'objectives.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for path in ('finite_noiseless','awgn'):
        for layout in ('1','2','3','mixed'):
            rows={r['step']:r for r in validation if r['path']==path and r['layout']==layout}
            series=[rows[s] for s in sorted(rows)]
            for ax,key in zip(axes,('source_psnr','source_color_rg_error_rmse')):
                ax.plot([r['step'] for r in series],[r[key] for r in series],label=path+'/q'+layout,
                        linestyle='--' if path=='finite_noiseless' else '-')
    for ax,label in zip(axes,('Source PSNR (dB)','R-G error RMSE')):
        ax.set(xlabel='Optimizer updates',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=6)
    save(fig,'quality.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    gains=read('prefix_gains.jsonl')
    for lower,upper in (('1','2'),('2','3')):
        by_step={r['step']:r for r in gains if r['path']=='awgn' and r['lower']==lower and r['upper']==upper and r['phase']!='test'}
        rows=[by_step[s] for s in sorted(by_step)]
        axes[0].plot([r['step'] for r in rows],[r['source_psnr_gain_db'] for r in rows],label='q'+lower+' to q'+upper)
    axes[0].set(ylabel='Enhancement PSNR gain (dB)');axes[0].legend()
    axes[1].plot([r['step'] for r in losses],[r['step_seconds'] for r in losses]);axes[1].set(ylabel='Seconds per update')
    for ax in axes:ax.set(xlabel='Optimizer updates');ax.grid(alpha=.2)
    save(fig,'enhancement_runtime.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for group in dict.fromkeys(g for r in losses for g in r['gradient_norms']):
        axes[0].plot([r['step'] for r in losses],[r['gradient_norms'].get(group,0) for r in losses],label=group)
        axes[1].plot([r['step'] for r in losses],[r['updates'].get(group,{}).get('update_norm',0) for r in losses],label=group)
    for ax,label in zip(axes,('Gradient norm','Parameter update norm')):
        ax.set(xlabel='Optimizer updates',ylabel=label,yscale='symlog');ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,'gradients_updates.png')
