"""Separate paired test endpoints from intermediate validation and oracle swaps."""
import json
from pathlib import Path

from .channel_audit import write_csv
from .modular_experiment import json_write


def summarize(out, modes):
    out = Path(out); rows = []; ablations = []; histories = []; view_sets = set()
    for mode in modes:
        folder = out/mode
        for label in ('initial', 'channel_end', 'final'):
            root = folder/'evaluation'/label
            if not (root/'complete.json').exists(): continue
            meta = json.loads((root/'complete.json').read_text(encoding='utf-8'))
            view_sets.add(tuple(meta['view_names']))
            power = json.loads((root/'power.json').read_text(encoding='utf-8'))
            data = [json.loads(s) for s in (root/'metrics/validation.jsonl').read_text().splitlines()]
            clean = next(r for r in data if r['path'] == 'clean')
            for tier in (1, 2, 3):
                row = dict(mode=mode, endpoint=label, step=meta['step'], tier=tier, views=len(meta['view_names']),
                    clean_psnr=clean['source_psnr'], mean_complex_power=power['prefixes'][tier-1]['mean_complex_power'],
                    power_budget_ok=power['within_average_power_budget'])
                for path, key in (('finite_noiseless', 'noiseless'), ('awgn', 'awgn10')):
                    r = next(r for r in data if r['path'] == path and r['layout'] == str(tier))
                    for metric in ('source_mse', 'source_psnr', 'source_ssim', 'source_lpips', 'photo_psnr', 'photo_ssim', 'photo_lpips', 'latent_nmse'):
                        row[key+'_'+metric] = r.get(metric)
                rows.append(row)
        path = folder/'ablation/results.json'
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            ablations.extend(dict(mode=mode, **{k: v for k, v in r.items() if k != 'per_view_trial'}) for r in data['rows'])
        for root in sorted((folder/'evaluation').glob('*')) if (folder/'evaluation').exists() else []:
            if not (root/'complete.json').exists(): continue
            meta = json.loads((root/'complete.json').read_text())
            if meta['scope'] != 'validation': continue
            data = [json.loads(s) for s in (root/'metrics/validation.jsonl').read_text().splitlines()]
            histories.extend(dict(mode=mode, **r) for r in data if r['path'] == 'awgn' and r['layout'] in ('1', '2', '3'))
    if len(view_sets) > 1: raise RuntimeError('test endpoints use different camera sets')
    for row in rows:
        paired_rows = [r for r in rows if r['endpoint'] == row['endpoint'] and r['tier'] == row['tier']]
        pair = [r['mean_complex_power'] for r in paired_rows]
        row['pair_power_matched_5pct'] = (max(pair)/max(min(pair), 1e-30) <= 1.05
            if {r['mode'] for r in paired_rows} == {'point', 'block'} else None)
    comparison = out/'comparison'; comparison.mkdir(parents=True, exist_ok=True)
    write_csv(comparison/'test_endpoints.csv', rows); write_csv(comparison/'attribute_replacements.csv', ablations)
    result = dict(test_rows=rows, ablation_rows=ablations, test_view_names=list(next(iter(view_sets))) if view_sets else [],
        interpretation='test endpoints paired by cameras/noise; intermediate validation is plotted separately; replacements are oracle diagnostics, not deployment results')
    json_write(comparison/'summary.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    labels = ('initial', 'channel_end', 'final')
    for tier, ax in enumerate(axes, 1):
        for mode in modes:
            selected = [r for label in labels for r in rows if r['mode'] == mode and r['tier'] == tier and r['endpoint'] == label]
            for key, style in (('awgn10_source_psnr', '-o'), ('noiseless_source_psnr', '--o')):
                if selected: ax.plot([r['endpoint'] for r in selected], [r[key] for r in selected], style, label=mode+'/'+key.split('_')[0])
        ax.set(title=f'q{tier}: paired test endpoints', ylabel='Source PSNR (dB)'); ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=7)
    fig.savefig(comparison/'test_endpoints.png', dpi=150); plt.close(fig)
    if histories:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
        for tier, ax in enumerate(axes, 1):
            for mode in modes:
                selected = sorted((r for r in histories if r['mode'] == mode and r['layout'] == str(tier)), key=lambda r: r['step'])
                if selected: ax.plot([r['step'] for r in selected], [r['source_psnr'] for r in selected], '-o', label=mode)
            ax.set(title=f'q{tier}: original heldout validation', xlabel='Cumulative optimizer updates', ylabel='Source PSNR (dB)'); ax.grid(alpha=.2)
            if ax.lines: ax.legend()
        fig.savefig(comparison/'validation_curves.png', dpi=150); plt.close(fig)
    return result
