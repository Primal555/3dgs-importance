"""Separate historical preview, current exact replacement and action supervision."""
import json
from pathlib import Path

import torch


def current_exact_summary(costs, components, rates, old_tiers, exact, lambda_s):
    rows,slots=exact.nonzero(as_tuple=True)
    if not len(rows):
        return dict(positive_current_exact_replacements=0,positive_current_exact_gap_max_abs_error=0.)
    old=old_tiers[rows].to(costs.device)-1
    rows,slots=rows.to(costs.device),slots.to(costs.device)
    expected=components[rows,slots,0]+lambda_s*components[rows,slots,1]+rates[rows,slots]-rates[rows,old]
    observed=costs[rows,slots]-costs[rows,old]
    return dict(positive_current_exact_replacements=len(rows),
        positive_current_exact_gap_max_abs_error=float((observed-expected).abs().max()))


def audit_feedback(run):
    root=Path(run)
    metrics=root/'allocation/metrics' if (root/'allocation/metrics').is_dir() else root/'metrics'
    loss,audit=metrics/'loss.jsonl',metrics/'audit.jsonl'
    if not loss.is_file() or not audit.is_file():
        raise FileNotFoundError('allocation metrics/loss.jsonl and audit.jsonl required')
    historical=labels=checks=steps=logged=0
    logged_present=False;max_error=0.;pairs=unchanged=0
    for line in loss.read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        r=json.loads(line);steps+=1
        historical+=int(r.get('positive_correction_reused_actions',0))
        labels+=int(r.get('neural_exact_training_groups',0))
        checks+=int(r.get('finite_positive_corrections',0))
        if 'positive_current_exact_replacements' in r:
            logged_present=True;logged+=r['positive_current_exact_replacements']
            max_error=max(max_error,r['positive_current_exact_gap_max_abs_error'])
    with audit.open(encoding='utf8') as stream:
        for line in stream:
            if not line.strip():continue
            r=json.loads(line)
            if r.get('feedback_kind')!='positive':continue
            pairs+=1
            unchanged+=r['first_order_delta_rd']==r['calibrated_delta_rd']
    return dict(version='feedback_paths_v1',run=str(root.resolve()),steps=steps,
        audit_scope='network versus historical preview; before current exact replacement',
        historical_reuse=dict(actions=historical,audited_positive_pairs=pairs,unchanged_predictions=unchanged),
        current_exact_replacement=dict(measured_positive_actions=checks,
            logged_actions=logged if logged_present else None,
            maximum_gap_error=max_error if logged_present else None,
            telemetry_available=logged_present),
        observed_action_training_labels=labels,
        training_scope='label counts only; prediction improvements require independent validation')
