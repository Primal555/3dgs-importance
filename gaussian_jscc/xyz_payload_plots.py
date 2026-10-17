"""Plot recorded coordinates and branch losses; no decoding or compression."""
from pathlib import Path
from .modular_plots import plot_run as common_plots,read_rows


def plot_run(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); paths=common_plots(out)
    rows=read_rows(out/'metrics/validation.jsonl')
    if rows:
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        keys=('world_distance_rmse','world_distance_p95','world_bias_norm','payload_xyz_mse')
        labels=('XYZ distance RMSE (scene units)','XYZ distance P95 (scene units)',
                'Norm of mean XYZ error (scene units)','3D normalized XYZ payload MSE')
        for path,layout in dict.fromkeys((r['path'],r['layout']) for r in rows):
            by_step={r['step']:r for r in rows if r['path']==path and r['layout']==layout}
            series=[by_step[s] for s in sorted(by_step)]
            for ax,key in zip(axes.flat,keys):
                selected=[r for r in series if isinstance(r.get(key),(int,float))]
                if selected: ax.plot([r['step'] for r in selected],[r[key] for r in selected],label=f'{path}/{layout}',linewidth=.9)
        for ax,label in zip(axes.flat,labels):
            ax.set(xlabel='Real optimizer updates',ylabel=label,yscale='symlog')
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7,ncol=2)
        fig.suptitle('XYZ is inside payload; clean bypass is not finite-rate communication')
        target=out/'charts/position_precision.png'; fig.savefig(target,dpi=140); plt.close(fig); paths.append(target)
    losses=read_rows(out/'metrics/loss.jsonl')
    if any('payload_xyz_mse' in r for r in losses):
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,phase in zip(axes,('channel_bootstrap','channel_render')):
            selected=[r for r in losses if r['phase']==phase]
            for key,label in (('payload_xyz_mse','XYZ MSE'),('payload_attr_mse','Attribute latent MSE'),('latent_mse','Flat 51D MSE (diagnostic)')):
                points=[r for r in selected if key in r]
                if points: ax.plot([r['phase_step'] for r in points],[r[key] for r in points],label=label,linewidth=.7)
            ax.set(title=phase,xlabel='Real phase updates',ylabel='Separate MSE objectives',yscale='symlog'); ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
        target=out/'charts/payload_branch_errors.png'; fig.savefig(target,dpi=140); plt.close(fig); paths.append(target)
    return paths
