"""Plots keep phase objectives separate and distinguish clean from transmitted."""
import json
from pathlib import Path


def plot_run(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    def read(name):
        path = out/name
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
    loss, validation, renders = read('loss.jsonl'), read('validation.jsonl'), read('render_validation.jsonl')
    charts = out/'charts'
    charts.mkdir(exist_ok=True)
    phases = [p for p in ('representation', 'adapter', 'joint') if any(r['phase'] == p for r in loss)]
    if phases:
        fig, axes = plt.subplots(2, len(phases), squeeze=False, figsize=(5*len(phases), 7))
        for index, phase in enumerate(phases):
            rows = [r for r in loss if r['phase'] == phase]
            x = [r['step'] for r in rows]
            axes[0, index].plot(x, [r['loss'] for r in rows], label='Total objective', alpha=.7)
            for term in rows[0]['terms']:
                axes[0, index].plot(x, [r['terms'][term] for r in rows], label=term)
            for module in rows[0]['module_grad_norms']:
                axes[1, index].plot(x, [r['module_grad_norms'][module] for r in rows], label=module)
            axes[0, index].set_title(phase)
            axes[1, index].set_yscale('symlog', linthresh=.01)
            for axis in axes[:, index]:
                axis.grid(alpha=.2)
                axis.legend(fontsize=7)
                axis.set_xlabel('Optimization step')
        axes[0, 0].set_ylabel('Phase-specific objective')
        axes[1, 0].set_ylabel('Gradient L2 before clipping')
        fig.tight_layout()
        fig.savefig(charts/'training_by_phase.png', dpi=150)
        plt.close(fig)
    if validation:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for route in ('clean', 'communication'):
            for axis, key in zip(axes, ('loss', 'xyz_pooled_world_rmse')):
                axis.plot([r['step'] for r in validation], [r[route][key] for r in validation], label=route)
                axis.set_title(key)
                axis.set_xlabel('Optimization step')
                axis.grid(alpha=.2)
                axis.legend()
        fig.tight_layout()
        fig.savefig(charts/'heldout_reconstruction.png', dpi=150)
        plt.close(fig)
    if renders:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for row, target in enumerate(('source', 'photo')):
            for col, metric in enumerate(('psnr', 'ssim')):
                axis = axes[row, col]
                for route in ('clean', 'communication'):
                    axis.plot([r['step'] for r in renders], [r[route][target+'_'+metric] for r in renders], label=route)
                if target == 'photo':
                    baseline = [sum(v['reference_photo'][metric] for v in r['views'])/len(r['views']) for r in renders]
                    axis.plot([r['step'] for r in renders], baseline, '--', label='Source PLY reference')
                axis.set_title(f'{metric.upper()} vs {target}')
                axis.set_xlabel('Optimization step')
                axis.legend()
                axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(charts/'render_quality.png', dpi=150)
        plt.close(fig)
    axis_rows = [r for r in validation if 'axis_native_radius_p50' in r['clean']]
    if axis_rows:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        x = [r['step'] for r in axis_rows]
        for kind in ('native', 'effective'):
            for quantile in ('p50', 'p95'):
                axes[0, 0].plot(x, [r['clean'][f'axis_{kind}_radius_{quantile}'] for r in axis_rows], label=f'{kind} {quantile}')
            axes[0, 1].plot(x, [r['clean'][f'axis_{kind}_within_one_fraction'] for r in axis_rows], label=kind)
        axes[0, 0].set_yscale('symlog', linthresh=1)
        axes[0, 0].set_title('Clean heldout ellipsoid error (mean block quantiles)')
        axes[0, 1].set_title('Fraction within one teacher ellipsoid (not pixels)')
        for key in ('axis_clamped_fraction', 'axis_affected_point_fraction'):
            axes[1, 0].plot(x, [r['clean'][key] for r in axis_rows], label=key)
        axes[1, 0].set_title('Heldout axes/points affected by fixed scale floor')
        profiled = [r for r in loss if 'position' in r.get('objective_module_grad_norms', {})]
        for term in ('position', 'shape', 'appearance'):
            for module in ('representation_encoder', 'representation_decoder'):
                axes[1, 1].plot([r['step'] for r in profiled],
                    [r['objective_module_grad_norms'][term][module] for r in profiled],
                    label=term+' / '+module.replace('representation_', ''))
        axes[1, 1].set_yscale('symlog', linthresh=.01)
        axes[1, 1].set_title('Weighted objective gradient L2 (before clipping)')
        for axis in axes.flat:
            axis.set_xlabel('Optimization step')
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(charts/'teacher_axis_diagnostics.png', dpi=150)
        plt.close(fig)
