"""Positive-tier supervision in its own fixed numeric units."""
import math

import torch
from torch.nn import functional as F

from .neural_correction_learning import CorrectionLearner
from .spatial_benefit import pack_inputs

OBJECTIVE = 'positive_pair_rms_with_q0_auxiliary_v1'


def _pairs(device):
    return torch.triu_indices(3, 3, offset=1, device=device)


def fit_positive_scales(records, original_rd_scale=None, original_tie_fraction=.02):
    """Weighted RMS of positive pairs, fitted once on training records only."""
    component_squares = torch.zeros(2, dtype=torch.float64)
    rd_squares = 0.
    total = 0.
    counts = dict(full_state=0, action_gap=0)
    comparisons = ignored = 0
    for g in records:
        if g.get('validation', False):
            raise ValueError('validation labels cannot fit positive scales')
        weight = float(g.get('importance_weight', 1.))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('positive source importance required')
        if g.get('supervision') == 'action_gap':
            gap = torch.as_tensor(g['delta_values']).double().reshape(1, 2)
            rate = torch.tensor([g['rate_delta']], dtype=torch.float64)
            counts['action_gap'] += 1
        else:
            values = torch.stack([v['values'].double() for v in g['views']])
            i, j = _pairs(values.device)
            gap = values[:, j]-values[:, i]
            r = torch.as_tensor(g['rates']).double()
            rate = r[j]-r[i]
            counts['full_state'] += 1
        if not torch.isfinite(gap).all() or not torch.isfinite(rate).all():
            raise ValueError('finite positive labels required')
        component_squares += gap.square().reshape(-1, 2).mean(0).cpu()*weight
        # Regression is per view; allocator directions use the camera-window mean.
        window_gap = gap.sum(-1).mean(0)+rate if gap.ndim == 3 else gap.sum(-1)+rate
        rd_squares += float(window_gap.square().mean())*weight
        comparisons += window_gap.numel()
        if original_rd_scale is not None:
            ignored += int((window_gap.abs()/float(original_rd_scale) <= original_tie_fraction).sum())
        total += weight
    if not total:
        raise ValueError('nonempty training labels required')
    return dict(components=(component_squares/total).sqrt().clamp_min(1e-8),
                rd=torch.tensor(max(math.sqrt(rd_squares/total), 1e-8), dtype=torch.float64),
                counts=counts, fit_scope='fixed training subset only', floor=1e-8,
                positive_comparisons=comparisons,
                original_ignored_comparisons=ignored if original_rd_scale is not None else None)


def direction_kl(prediction, target, rates, scale):
    """Continuous gap targets; no hard near-tie ignore band."""
    scale = torch.as_tensor(scale, device=prediction.device, dtype=prediction.dtype)
    estimate = (prediction.sum(-1)+rates)/scale
    actual = ((target.sum(-1)+rates)/scale).detach()
    probability = torch.sigmoid(-actual)
    kl = (F.binary_cross_entropy_with_logits(-estimate, probability, reduction='none')-
          F.binary_cross_entropy_with_logits(-actual, probability, reduction='none')).clamp_min(0.)
    return actual.abs()*kl


def positive_losses(prediction, target, rates, scales, q0_scale, *,
                    decision_weight=.25, q0_aux_weight=.25):
    i, j = _pairs(prediction.device)
    estimate, actual = prediction[:, j]-prediction[:, i], target[:, j]-target[:, i]
    component_scale = torch.as_tensor(scales['components'], device=prediction.device, dtype=prediction.dtype)
    positive = ((estimate-actual)/component_scale).square().mean((-2, -1))
    direction = direction_kl(estimate, actual, rates[:, j]-rates[:, i], scales['rd']).mean(-1)
    common = ((prediction-target).mean(-2)/q0_scale).square().mean(-1)
    return dict(positive_regression=positive, positive_direction=direction, q0_auxiliary=common,
                total=positive+decision_weight*direction+q0_aux_weight*common)


class PositivePairLearner(CorrectionLearner):
    """Offline fork; original normalizers and output parameterization stay fixed."""

    def configure_objective(self, scales, q0_aux_weight=.25):
        if not math.isfinite(q0_aux_weight) or q0_aux_weight < 0:
            raise ValueError('finite nonnegative q0 auxiliary weight required')
        self.positive_scales, self.q0_aux_weight = scales, float(q0_aux_weight)

    def _batch_losses(self, groups):
        sizes = [len(g['views']) for g in groups]
        if not sizes or min(sizes) < 1:
            raise ValueError('nonempty view groups required')
        # Preserve the original learner's full-state/gap call order so paired
        # trials also use identical per-record dropout masks.
        split = [None]*len(groups)
        for is_gap in (False, True):
            indices = [i for i, g in enumerate(groups) if (g.get('supervision') == 'action_gap') == is_gap]
            if not indices:
                continue
            prediction = self.network(*pack_inputs([v['inputs'] for i in indices
                for v in groups[i]['views']], self.device)).double()
            for i, p in zip(indices, prediction.split([sizes[i] for i in indices])):
                split[i] = p
        output = {k: [] for k in ('positive_regression', 'positive_direction', 'q0_auxiliary', 'total')}
        for g, p in zip(groups, split):
            if g.get('supervision') == 'action_gap':
                value = p.mean(0)
                estimate = value[g['new_tier']-1]-value[g['old_tier']-1]
                actual = g['delta_values'].to(estimate)
                component_scale = self.positive_scales['components'].to(estimate)
                regression = ((estimate-actual)/component_scale).square().mean()
                direction = direction_kl(estimate, actual, estimate.new_tensor(g['rate_delta']),
                                         self.positive_scales['rd'])
                item = dict(positive_regression=regression, positive_direction=direction,
                            q0_auxiliary=regression*0,
                            total=regression+self.decision_weight*direction)
            else:
                target = torch.stack([v['values'] for v in g['views']]).to(p)
                rates = torch.as_tensor(g['rates'], device=p.device, dtype=p.dtype)
                i, j = _pairs(p.device)
                component_scale = self.positive_scales['components'].to(p)
                regression = (((p[:, j]-p[:, i])-(target[:, j]-target[:, i]))/
                              component_scale).square().mean()
                # Match the allocator's average across the same camera window.
                direction = direction_kl((p[:, j]-p[:, i]).mean(0),
                    (target[:, j]-target[:, i]).mean(0), rates[j]-rates[i],
                    self.positive_scales['rd']).mean()
                common = ((p-target).mean(1)/self.scales['rd'].to(p)).square().mean()
                item = dict(positive_regression=regression, positive_direction=direction,
                    q0_auxiliary=common,
                    total=regression+self.decision_weight*direction+self.q0_aux_weight*common)
            for key in output:
                output[key].append(item[key])
        return {key: torch.stack(value) for key, value in output.items()}

    def state_dict(self):
        return dict(super().state_dict(), offline_objective=OBJECTIVE,
                    positive_scales=self.positive_scales, q0_aux_weight=self.q0_aux_weight)
