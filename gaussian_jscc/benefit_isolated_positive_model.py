"""Frozen common values and an independently trained positive-tier representation."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .benefit_footprint_features import CHANNELS
from .benefit_footprint_model import FootprintLearner
from .spatial_benefit import SpatialBenefitNetwork


class IsolatedPositiveNetwork(nn.Module):
    def __init__(self, anchor, relative):
        super().__init__()
        self.anchor = anchor.requires_grad_(False).eval()
        self.relative = relative
        width = relative.state_head[0].in_features
        self.profile_encoder = nn.Sequential(nn.Conv2d(CHANNELS, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.SiLU(), nn.Flatten(),
            nn.Linear(24*5*5, 32), nn.SiLU())
        self.profile_projection = nn.Linear(64, width, bias=False)
        nn.init.zeros_(self.profile_projection.weight)

    @classmethod
    def from_spatial(cls, state, dropout=.05):
        anchor = SpatialBenefitNetwork(state['points.0.weight'].shape[0], dropout)
        anchor.load_state_dict(state, strict=True)
        return cls(anchor, copy.deepcopy(anchor).requires_grad_(True))

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(self, features, context, states, weights, valid, footprints):
        with torch.no_grad():
            common = self.anchor(features, context, states, weights, valid).mean(1, keepdim=True)
        x = self.relative.state_embeddings(features, context, states, weights, valid)
        profile = self.profile_encoder(torch.asinh(footprints).reshape(-1, CHANNELS, 9, 9))
        x = x+self.profile_projection(profile.reshape(len(x), 4, 64))
        scores = self.relative.state_head(x[:, 1:])*self.relative.output_scale
        identical = ((features[:, 1:] == features[:, :1]).all(-1) &
            (states[:, 1:] == states[:, :1]).flatten(-2).all(-1) &
            (footprints[:, 1:] == footprints[:, :1]).flatten(2).all(-1))
        active = (~identical)[..., None]
        count = active.sum(1, keepdim=True).clamp_min(1)
        center = (scores*active).sum(1, keepdim=True)/count
        # Known zero candidates remain exactly zero. Redistribute the frozen
        # common sum only over active states, preserving its mean and gradients.
        return ((scores-center)+common*3/count).masked_fill(~active, 0.)


def positive_losses(prediction, truth, rates, scale, *, pair_weight=1., decision_weight=.25,
                    tie_fraction=.02):
    scale = torch.as_tensor(scale, device=prediction.device, dtype=prediction.dtype).clamp_min(1e-8)
    error = prediction-truth
    centered = ((error-error.mean(-2, keepdim=True))/scale).square().mean((-2, -1))
    i, j = torch.triu_indices(3, 3, offset=1, device=prediction.device)
    estimate = prediction[..., j, :]-prediction[..., i, :]
    actual = truth[..., j, :]-truth[..., i, :]
    pair = ((estimate-actual)/scale).square().mean((-2, -1))
    p = (estimate.sum(-1)+rates[..., j]-rates[..., i])/scale
    t = ((actual.sum(-1)+rates[..., j]-rates[..., i])/scale).detach()
    target = torch.sigmoid(-t)
    kl = (F.binary_cross_entropy_with_logits(-p, target, reduction='none')-
          F.binary_cross_entropy_with_logits(-t, target, reduction='none')).clamp_min(0.)
    direction = (kl*torch.where(t.abs() > tie_fraction, t.abs(), 0.)).mean(-1)
    return dict(centered=centered, pair=pair, decision=direction,
                total=centered+pair_weight*pair+decision_weight*direction)


class IsolatedPositiveLearner(FootprintLearner):
    def _batch_losses(self, groups):
        if any(g.get('supervision') == 'action_gap' for g in groups):
            raise ValueError('fixed candidate dataset requires complete three-tier teachers')
        sizes = [len(g['views']) for g in groups]
        if not sizes or min(sizes) < 1:
            raise ValueError('nonempty teacher view groups required')
        inputs = [v['inputs'] for g in groups for v in g['views']]
        p = self.network(*self.pack(inputs)).double()
        t = torch.stack([v['values'] for g in groups for v in g['views']]).to(p)
        rates = torch.stack([torch.as_tensor(g['rates']) for g in groups]).to(p)
        owner = torch.repeat_interleave(torch.arange(len(groups), device=p.device),
                                       torch.tensor(sizes, device=p.device))
        per_view = positive_losses(p, t, rates[owner], self.scales['rd'],
            pair_weight=self.pair_weight, decision_weight=self.decision_weight, tie_fraction=self.tie_fraction)
        result = {k: torch.stack([s.mean() for s in v.split(sizes)])
                  for k, v in per_view.items() if k != 'total'}
        mean_p = torch.stack([s.mean(0) for s in p.split(sizes)])
        mean_t = torch.stack([s.mean(0) for s in t.split(sizes)])
        # Decisions use the same camera-window mean as the teacher and deployment.
        result['decision'] = positive_losses(mean_p, mean_t, rates, self.scales['rd'],
            tie_fraction=self.tie_fraction)['decision']
        result['total'] = result['centered']+self.pair_weight*result['pair']+self.decision_weight*result['decision']
        return result


def make_learner(initial, config, device):
    learner = IsolatedPositiveLearner(None, device, hidden=config['hidden'], lr=config['benefit_lr'],
        dropout=config['dropout'], weight_decay=config['weight_decay'], decision_weight=config['decision_weight'],
        tie_fraction=config['tie_fraction'], common_weight=config['common_weight'], pair_weight=config['pair_weight'])
    learner.load_state_dict(copy.deepcopy(initial['learner']))
    learner.network = IsolatedPositiveNetwork.from_spatial(initial['learner']['network'], config['dropout']).to(device)
    learner.optimizer = torch.optim.AdamW([p for p in learner.network.parameters() if p.requires_grad],
        lr=config['benefit_lr'], weight_decay=config['weight_decay'])
    return learner
