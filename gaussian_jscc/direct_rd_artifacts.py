"""Small analysis artifacts; checkpoints remain separate from the review bundle."""
import csv
import json
from pathlib import Path
import tarfile

from .modular_experiment import json_write


def read_rows(path):
    if not path.is_file(): return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(out):
    out = Path(out); folder = out/'review'; folder.mkdir(exist_ok=True)
    completion = json.loads((out/'complete.json').read_text(encoding='utf-8'))
    events = {r['event']: r for r in read_rows(out/'metrics/validation.jsonl')}
    endpoints = {name: {k: v for k, v in events[name].items() if k != 'per_view_trial'}
                 for name in ('initial', 'latest', 'uniform_q2', 'uniform_q3')}
    latest, q2 = endpoints['latest'], endpoints['uniform_q2']
    saving = 1-latest['total_channel_uses']/q2['total_channel_uses']
    summary = dict(completion=completion, endpoints=endpoints,
        vs_q2=dict(cost_saving_fraction=saving, source_psnr_delta=latest['source_psnr']-q2['source_psnr'],
                   photo_psnr_delta=latest['photo_psnr']-q2['photo_psnr'], rd_delta=latest['rd_score']-q2['rd_score'],
                   source_quality_cost_goal=latest['source_psnr'] >= q2['source_psnr'] and saving > 0,
                   photo_quality_cost_goal=latest['photo_psnr'] >= q2['photo_psnr'] and saving > 0),
        scope='paired calibration RD acceptance; independent test endpoints; reliable-side net bits/use assumption',
        resume='checkpoints/latest/state.pt plus original CODEC run and scene data; no old replay required')
    json_write(folder/'summary.json', summary)
    keys = ('event', 'step', 'source_psnr', 'source_ssim', 'source_lpips', 'photo_psnr', 'photo_ssim',
            'photo_lpips', 'source_distortion', 'total_channel_uses', 'payload_complex_symbols',
            'coordinate_channel_uses', 'metadata_channel_uses', 'scale_channel_uses', 'rd_score')
    with (folder/'endpoints.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        writer.writerows({k: row.get(k) for k in keys} for row in endpoints.values())
    audits = read_rows(out/'metrics/audits.jsonl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if audits:
        axes[0].plot([0]+[r['step'] for r in audits], [audits[0]['before_rd']]+[r['after_rd'] for r in audits])
    axes[0].set(xlabel='Candidate windows', ylabel='Accepted calibration RD')
    bottom = [0., 0., 0.]; names = ('uniform_q2', 'latest', 'uniform_q3')
    for tier, color in enumerate(('#888888', '#2878b5', '#e79429', '#9467bd')):
        values = [endpoints[name]['tier_counts'][tier]/sum(endpoints[name]['tier_counts'])*100 for name in names]
        axes[1].bar(names, values, bottom=bottom, color=color, label='q'+str(tier))
        bottom = [a+b for a, b in zip(bottom, values)]
    axes[1].set(ylabel='Actual deployment share (%)', ylim=(0, 100)); axes[1].legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(folder/'allocation_summary.png', dpi=160); plt.close(fig)
    bundle = out/'review.tar.gz'; temporary = out/'review.tar.gz.tmp'
    files = [out/name for name in ('complete.json', 'plan.json', 'training.json', 'preflight.json',
                                  'deployment/latest_tiers.npy')]
    for section in ('review', 'metrics', 'evaluation/initial', 'evaluation/latest', 'evaluation/uniform_q3'):
        files.extend(p for p in (out/section).rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.png'))
    with tarfile.open(temporary, 'w:gz') as archive:
        for path in files:
            if path.is_file(): archive.add(path, arcname=str(Path(out.name)/path.relative_to(out)), recursive=False)
    temporary.replace(bundle)
    print(f'Review bundle: {bundle} ({bundle.stat().st_size/2**20:.2f} MiB)', flush=True)
    return summary


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('out')
    summarize(p.parse_args().out)
