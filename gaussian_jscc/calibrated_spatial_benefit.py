"""RD-gap-calibrated direction targets for the unchanged spatial value model."""
import torch
from torch.nn import functional as F

from .spatial_benefit import SpatialLearner

DIRECTION_OBJECTIVE = 'rd_gap_bernoulli_kl_v1'


def calibrated_direction_loss(prediction, truth, rates, scale, tie_fraction=.02):
    """Six RD comparisons with a stationary optimum at the true numeric values.

    q0 is the zero anchor. The known communication cost enters each state
    before pair formation. Scale, true-gap weights and ties retain the prior
    physical RD units; only the hard sign target becomes sigmoid(-true_gap).
    """
    scale = torch.as_tensor(scale, device=prediction.device, dtype=prediction.dtype).clamp_min(1e-8)
    predicted_cost = torch.cat((prediction.new_zeros(*prediction.shape[:-2], 1),
                                prediction.sum(-1)+rates), -1)/scale
    true_cost = torch.cat((truth.new_zeros(*truth.shape[:-2], 1), truth.sum(-1)+rates), -1)/scale
    i, j = torch.triu_indices(4, 4, offset=1, device=prediction.device)
    actual = (true_cost[..., j]-true_cost[..., i]).detach()
    estimate = predicted_cost[..., j]-predicted_cost[..., i]
    weight = torch.where(actual.abs() > tie_fraction, actual.abs(), 0.)
    target = torch.sigmoid(-actual)
    cross_entropy = F.binary_cross_entropy_with_logits(-estimate, target, reduction='none')
    entropy = F.binary_cross_entropy_with_logits(-actual, target, reduction='none')
    # BCE minus target entropy is Bernoulli KL. Both terms use the stable
    # logit implementation, including saturated targets. Clamp roundoff only.
    divergence = (cross_entropy-entropy).clamp_min(0.)
    return (weight*divergence).mean(-1)


class CalibratedSpatialLearner(SpatialLearner):
    direction_loss = staticmethod(calibrated_direction_loss)

    def state_dict(self):
        return dict(super().state_dict(), direction_objective=DIRECTION_OBJECTIVE)

    def load_state_dict(self, state):
        if state.get('direction_objective') != DIRECTION_OBJECTIVE:
            raise ValueError('calibrated spatial direction objective mismatch')
        super().load_state_dict(state)

    def validate(self):
        report = super().validate()
        report['direction_objective'] = DIRECTION_OBJECTIVE
        return report
