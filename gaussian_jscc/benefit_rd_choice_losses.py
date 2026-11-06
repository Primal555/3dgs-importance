"""Shared-unit RD regression with numeric and joint-choice supervision."""
import math

import torch
from torch.nn import functional as F

from .neural_correction_learning import CorrectionLearner
from .spatial_benefit import pack_inputs


OBJECTIVE = 'centered_rd_joint_choice_v1'


def fit_choice_targets(records):
    gaps, ranges = [], []
    for g in records:
        if g.get('validation', False):
            raise ValueError('validation labels cannot fit choice targets')
        if g.get('supervision') == 'action_gap':
            gap = torch.as_tensor(g['delta_values']).double().sum()+float(g['rate_delta'])
            gaps.append(gap.abs().reshape(1).cpu())
        else:
            truth = torch.stack([v['values'].double().cpu() for v in g['views']]).mean(0)
            cost = truth.sum(-1)+torch.as_tensor(g['rates']).double().cpu()
            i, j = torch.triu_indices(3, 3, offset=1)
            gaps.append((cost[j]-cost[i]).abs())
            ranges.append(cost.max()-cost.min())
    if not gaps or not ranges:
        raise ValueError('complete training teachers required to fit choice targets')
    gaps = torch.cat(gaps)
    ranges = torch.stack(ranges)
    if not torch.isfinite(gaps).all() or not torch.isfinite(ranges).all():
        raise ValueError('finite training teacher costs required')
    nonzero = gaps[gaps > 0]
    temperature = max(float(nonzero.median()) if len(nonzero) else 0., 1e-8)
    return dict(temperature=temperature, large_gap_threshold=float(torch.quantile(ranges, .9)),
        fit_scope='fixed training records only', temperature_statistic='nonzero pair-gap median; floor 1e-8',
        threshold_statistic='complete-teacher RD range Q90; record uniform',
        comparisons=len(gaps), complete_records=len(ranges))


def joint_kl(estimate, actual, temperature):
    logp = F.log_softmax(-estimate/temperature, dim=-1)
    logq = F.log_softmax(-actual.detach()/temperature, dim=-1)
    return (logq.exp()*(logq-logp)).sum(-1).clamp_min(0.)


def choice_losses(prediction, target, rates, scale, temperature, *,
                  component_weight=.25, decision_weight=.25):
    """[views, tiers, 2] inputs. Average cameras only for the allocation tasks."""
    rates = torch.as_tensor(rates, device=prediction.device, dtype=prediction.dtype)
    estimate = prediction.mean(0).sum(-1)+rates
    actual = target.mean(0).sum(-1)+rates
    rd = (((estimate-estimate.mean())-(actual-actual.mean()))/scale).square().mean()
    components = ((prediction-target)/scale).square().mean()
    # Keep choice curvature in the same RD/s units. A small typical-gap
    # temperature must not amplify classification gradients by 1/tau^2.
    joint = joint_kl(estimate, actual, temperature)*(temperature/scale)**2
    return dict(rd_regression=rd, component_regression=components, joint_choice=joint,
                total=rd+component_weight*components+decision_weight*joint)


class RDChoiceLearner(CorrectionLearner):
    def configure_objective(self, targets, component_weight=.25):
        if (not math.isfinite(component_weight) or component_weight <= 0 or
                not math.isfinite(targets['temperature']) or targets['temperature'] <= 0):
            raise ValueError('positive finite component weight and temperature required')
        self.choice_targets, self.component_weight = dict(targets), float(component_weight)

    def _batch_losses(self, groups):
        sizes = [len(g['views']) for g in groups]
        if not sizes or min(sizes) < 1:
            raise ValueError('nonempty view groups required')
        predictions = [None]*len(groups)
        # Match the original full-state/action-gap forward order and dropout.
        for is_gap in (False, True):
            indices = [i for i, g in enumerate(groups) if (g.get('supervision') == 'action_gap') == is_gap]
            if indices:
                output = self.network(*pack_inputs([v['inputs'] for i in indices
                    for v in groups[i]['views']], self.device)).double()
                for i, value in zip(indices, output.split([sizes[i] for i in indices])):
                    predictions[i] = value
        losses = {k: [] for k in ('rd_regression', 'component_regression', 'joint_choice', 'total')}
        scale = self.scales['rd'].to(self.device)
        for g, p in zip(groups, predictions):
            if g.get('supervision') == 'action_gap':
                difference = p.mean(0)[g['new_tier']-1]-p.mean(0)[g['old_tier']-1]
                truth = g['delta_values'].to(p)
                rd = ((difference.sum()-truth.sum())/scale).square()
                component = ((difference-truth)/scale).square().mean()
                estimated_cost = difference.sum()+g['rate_delta']
                true_cost = truth.sum()+g['rate_delta']
                temperature = self.choice_targets['temperature']
                joint = joint_kl(torch.stack((p.new_zeros(()), estimated_cost)),
                    torch.stack((p.new_zeros(()), true_cost)), temperature)*(temperature/scale)**2
                item = dict(rd_regression=rd, component_regression=component, joint_choice=joint,
                    total=rd+self.component_weight*component+self.decision_weight*joint)
            else:
                truth = torch.stack([v['values'] for v in g['views']]).to(p)
                item = choice_losses(p, truth, g['rates'], scale, self.choice_targets['temperature'],
                    component_weight=self.component_weight, decision_weight=self.decision_weight)
            for key in losses:
                losses[key].append(item[key])
        return {k: torch.stack(v) for k, v in losses.items()}

    def state_dict(self):
        return dict(super().state_dict(), offline_objective=OBJECTIVE,
                    choice_objective=self.choice_targets, component_weight=self.component_weight)
