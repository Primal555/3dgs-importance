"""Compact true-deployment rate/distortion, coverage and timing artifacts."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from .budgeted_allocation_evaluation import tier_hash
from .modular_experiment import json_write


def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()] if path.exists() else []


def summarize(out, result=None):
    out=Path(out)
    if result is None: result=json.loads((out/'complete.json').read_text(encoding='utf-8'))
    rows=read_rows(out/'metrics/validation.jsonl'); losses=read_rows(out/'metrics/loss.jsonl')
    controls=read_rows(out/'metrics/control.jsonl'); audit=read_rows(out/'metrics/audit.jsonl')
    for row in rows:
        if row['budget_feasible'] != (row['total_channel_uses'] <= row['budget']):
            raise ValueError('reported feasibility disagrees with measured budget')
    selected=next(r for r in rows if r['event']=='selected')
    q=torch.from_numpy(np.load(out/'deployment/selected_tiers.npy',allow_pickle=False).astype(np.int64))
    if (result['budget_feasible'] != selected['budget_feasible']
            or result['selected_total_channel_uses'] != selected['total_channel_uses']
            or tier_hash(q) != result['tier_sha256'] or result['tier_sha256'] != selected['tier_sha256']
            or selected['codec_id'] != result['codec_id']
            or torch.bincount(q,minlength=4).tolist() != selected['tier_counts']):
        raise ValueError('selected deployment/checkpoint budget or map binding mismatch')
    endpoints=[r for r in rows if r['event'] in ('initial','latest','selected')]
    elapsed=[r['step_seconds'] for r in losses]
    timing=dict(optimization_step_p50_seconds=float(np.percentile(elapsed,50)) if elapsed else None,
                optimization_step_p90_seconds=float(np.percentile(elapsed,90)) if elapsed else None,
                evaluation_seconds=sum(r.get('evaluation_seconds',0.) for r in rows),
                control_cost_seconds=sum(r.get('cost_seconds',0.) for r in controls))
    comparisons=[]
    for a,b in itertools.combinations(audit,2):
        if a['step']==b['step'] and a['source_row']==b['source_row']:
            comparisons.append((a['measured_delta_mse']-b['measured_delta_mse'])*
                               (a['first_order_delta_mse']-b['first_order_delta_mse'])>=0)
    summary=dict(result=result,units='total complex channel uses',endpoints=endpoints,timing=timing,
        audit=dict(samples=len(audit),sign_agreement=float(np.mean([r['sign_agreement'] for r in audit])) if audit else None,
                   within_point_rank_agreement=float(np.mean(comparisons)) if comparisons else None),
        scope='fixed frozen codec; calibration-selected map; source/photo metrics kept separate',
        proxies='per-update expected/sample side cost is approximate; controller and feasibility use measured ten-draw deployments')
    json_write(out/'comparison/summary.json',summary)
    keys=('event','step','source_psnr','source_ssim','source_lpips','photo_psnr','photo_ssim','photo_lpips',
          'total_channel_uses','payload_complex_symbols','xyz_bytes','metadata_bytes','scale_stream_bytes',
          'budget','budget_feasible','codec_id','tier_sha256')
    with (out/'comparison/endpoints.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r.get(k) for k in keys} for r in endpoints)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'charts';folder.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(9,4));price=ax.twinx()
    if losses:
        for key,label in (('expected_cost_proxy','Single-draw expected proxy'),('sampled_cost_proxy','Sampled proxy')):
            ax.plot([r['step'] for r in losses],[r[key] for r in losses],label=label,alpha=.7)
        price.plot([r['step'] for r in losses],[r['price'] for r in losses],color='purple',label='Budget price',alpha=.7)
    if controls: ax.plot([r['step'] for r in controls],[r['total_channel_uses'] for r in controls],'k.-',label='Measured ten-draw deployment')
    ax.axhline(result['budget'],color='red',linestyle='--',label='Explicit budget')
    ax.set(xlabel='Actual allocation updates',ylabel='Total complex channel uses');price.set_ylabel('Budget price')
    ax.legend(loc='upper left',fontsize=8);price.legend(loc='upper right',fontsize=8);fig.tight_layout()
    fig.savefig(folder/'budget_cost_price.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));bottom=np.zeros(len(endpoints))
    for tier,color in enumerate(('#888888','#2878b5','#e79429','#9467bd')):
        values=np.array([r['tier_counts'][tier]/sum(r['tier_counts'])*100 for r in endpoints])
        ax.bar([r['event'] for r in endpoints],values,bottom=bottom,label='q'+str(tier),color=color);bottom+=values
    ax.set(ylabel='Actual ten-draw deployment share (%)',ylim=(0,100));ax.legend(ncol=4);fig.tight_layout()
    fig.savefig(folder/'tier_shares.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    if losses:
        ax.plot([r['step'] for r in losses],[r['evaluated_rows'] for r in losses],label='Original rows evaluated at least once')
        other=ax.twinx();other.plot([r['step'] for r in losses],[r['complete_coverages'] for r in losses],color='orange')
        other.set_ylabel('Complete coverages over ALL source rows')
    ax.set(xlabel='Actual allocation updates',ylabel='Positive-tier candidate coverage');ax.legend();fig.tight_layout()
    fig.savefig(folder/'candidate_coverage.png',dpi=160);plt.close(fig)
    return summary
