"""Small component/evidence diagnostics; no extra renders or scene exports."""
import csv
from pathlib import Path
import torch
from .budgeted_allocation_report import read_rows
from .modular_experiment import json_write


def summarize_structured(out, selected):
    out = Path(out); step = selected['progress']['step']
    evidence = selected['blocks']['evidence']; data = selected['blocks']['structured']['tensors']
    half_life = evidence['settings'][3]; kept = selected['latest_q'] > 0
    weight = evidence['weight']*torch.exp2(-(step-evidence['evidence_step']).double()/half_life)
    keep_weight = data['keep_weight']*torch.exp2(-(step-data['keep_step']).double()/half_life)
    def stats(value):
        value = value.double().flatten()
        return dict(mean=float(value.mean()) if value.numel() else 0.,
            zero_fraction=float((value==0).double().mean()) if value.numel() else 0.,
            quantiles=torch.quantile(value, torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)).tolist() if value.numel() else [])
    summary = dict(selected_step=step, primary_observation_rule='one resolved observation per action per step',
        history_units='component action deltas divided by fixed D_ref; rate is not stored in history',
        reference=selected['objective_reference'], positive_effective_weight=stats(weight[kept]),
        positive_observations=stats(evidence['observations'][kept]),
        keep_effective_weight_by_positive_tier=[stats(keep_weight[:, i]) for i in range(3)],
        keep_observations_by_positive_tier=[int(data['keep_observations'][:, i].sum()) for i in range(3)],
        exact_checks_by_branch={b:int(data[f'{b}_exact_count'].sum()) for b in ('positive', 'keep')},
        diagnostic_note='exact checks replace same-step proxy before primary accumulation; no extra primary weight')
    json_write(out/'comparison/structured_evidence.json', summary)
    rows = read_rows(out/'metrics/validation.jsonl')
    keys = ('event','step','total_channel_uses','source_mse','source_structure_error','source_distortion',
            'lambda_s','source_psnr','source_ssim','source_lpips','photo_psnr','photo_ssim','photo_lpips','rd_score')
    with (out/'comparison/structured_controls.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        writer.writerows({k:r.get(k) for k in keys} for r in rows
            if r['event'] in ('initial','selected','latest','uniform_q2','uniform_q3'))
    losses = read_rows(out/'metrics/loss.jsonl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    if losses:
        steps = [r['step'] for r in losses]
        for key, label in (('source_mse','MSE'), ('source_structure_error','lambda * (1-SSIM)')):
            axes[0].plot(steps, [r[key]*(r['lambda_s'] if key=='source_structure_error' else 1.) for r in losses], label=label)
        axes[1].plot(steps, [r['positive_history']['mean'] for r in losses], label='Retained positive effective weight')
        axes[1].plot(steps, [r['keep_history']['mean'] for r in losses], label='Keep history effective weight / slot')
        axes[2].plot(steps, [r['finite_sign_disagreements']/max(1,r['finite_nonzero_pairs']) for r in losses],
            label='Checked mixed-RD sign mismatch fraction')
    for ax in axes: ax.legend(fontsize=8)
    axes[0].set_ylabel('Raw distortion contribution'); axes[1].set_ylabel('Decayed observations')
    axes[2].set(xlabel='Actual allocation updates', ylabel='Selected checks only')
    fig.tight_layout(); fig.savefig(out/'charts/structured_evidence.png', dpi=160); plt.close(fig)
    return summary
