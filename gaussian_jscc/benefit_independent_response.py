"""Independent physical response heads and record-stratum gradient budgets."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .benefit_footprint_model import FootprintSplitNetwork
from .benefit_response_model import TASKS, combine_terms
from .benefit_three_task import TaskAccumulator

VERSION = 'independent_response_ordinary_budget_v1'


def positive_energy_mean(groups):
    total = count = 0.
    for group in groups:
        if group.get('validation'):
            raise ValueError('validation cannot initialize energy')
        weight = float(group.get('importance_weight', 1.))
        if not 0 < weight < float('inf'):
            raise ValueError('invalid importance weight')
        energy = torch.stack([v['response_terms'].double()[:, 1] for v in group['views']])
        if not torch.isfinite(energy).all() or (energy < 0).any():
            raise ValueError('invalid energy labels')
        positive = energy > 0
        total += float(energy[positive].sum()) * weight
        count += int(positive.sum()) * weight
    return total/count if count else 0.


class IndependentResponseNetwork(FootprintSplitNetwork):
    """Two signed common/relative channels for A/S plus an independent B head."""
    def __init__(self, hidden=256, dropout=.05):
        super().__init__(hidden, dropout)
        self.energy_head = nn.Sequential(nn.Linear(self.relative_head[0].in_features, hidden),
                                         nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer('term_scales', torch.ones(3))

    @classmethod
    def from_footprint(cls, state, dropout, term_scales, initial_energy):
        scales = torch.as_tensor(term_scales, dtype=torch.float32)
        if scales.shape != (3,) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('three positive response scales required')
        if not 0 <= initial_energy < float('inf'):
            raise ValueError('finite nonnegative initial energy required')
        network = cls(state['points.0.weight'].shape[0], dropout)
        values = network.state_dict()
        for name, value in state.items():
            if not name.startswith(('common_head.', 'relative_head.')):
                values[name] = value.clone()
        network.load_state_dict(values, strict=True)
        network.term_scales.copy_(scales)
        # A zero final weight makes B begin at the training positive-energy mean.
        # The positive fallback handles a dataset whose entire energy is zero.
        ratio = max(float(initial_energy/scales[1]), 1e-3)
        bias = ratio + math.log(-math.expm1(-ratio))
        nn.init.zeros_(network.energy_head[-1].weight)
        nn.init.constant_(network.energy_head[-1].bias, bias)
        return network

    def forward_terms(self, features, context, states, weights, valid, footprints):
        x = self.state_embeddings(features, context, states, weights, valid)
        profile = self.profile_encoder(torch.asinh(footprints).reshape(-1, 15, 9, 9))
        x = x + self.profile_projection(profile.reshape(len(x), 4, 64))
        signed_scales = self.term_scales[[0, 2]]
        scores = self.common_head(x) * signed_scales
        common = scores[:, 1:].mean(1, keepdim=True) - scores[:, :1]
        relative = self.relative_head(x[:, 1:]) * signed_scales
        signed = (common + relative - relative.mean(1, keepdim=True)).double()
        energy = F.softplus(self.energy_head(x[:, 1:]).squeeze(-1)).double() * self.term_scales[1].double()
        terms = torch.stack((signed[..., 0], energy, signed[..., 1]), -1)
        identical = ((features[:, 1:] == features[:, :1]).all(-1) &
                     (states[:, 1:] == states[:, :1]).flatten(-2).all(-1) &
                     (footprints[:, 1:] == footprints[:, :1]).flatten(2).all(-1))
        return terms.masked_fill(identical[..., None], 0.)

    def forward(self, *args):
        return combine_terms(self.forward_terms(*args))


class GroupTaskAccumulator(TaskAccumulator):
    """Accumulate full-parameter group/task gradients from each shared forward."""
    def __init__(self, parameters, shared_indices, names=TASKS):
        super().__init__(parameters, shared_indices, names)
        self.group_gradients = {s: [[torch.zeros_like(p) for p in self.parameters]
                                    for _ in self.names] for s in (0, 1)}
        self.tail_factor = 1.

    def add(self, per_record, weights, labels, *, diagnostics=False):
        if per_record.ndim != 2 or per_record.shape[1] != len(self.names):
            raise ValueError('one loss per record and task required')
        weights = weights.to(per_record)
        labels = torch.as_tensor(labels, device=per_record.device)
        if labels.shape != (len(per_record),) or not ((labels == 0) | (labels == 1)).all():
            raise ValueError('binary record strata required')
        active = [s for s in (0, 1) if (labels == s).any()]
        remaining = len(active)*len(self.names)
        for s in active:
            mask = labels == s
            contribution = (per_record[mask]*weights[mask, None]).sum(0)
            self.losses += contribution.detach()
            record = self.strata[s]
            record['records'] += int(mask.sum())
            record['losses'] += contribution.detach()
            for k, loss in enumerate(contribution):
                remaining -= 1
                gradients = torch.autograd.grad(loss, self.parameters, retain_graph=remaining > 0,
                                                allow_unused=True)
                for i, gradient in enumerate(gradients):
                    if gradient is not None:
                        gradient = gradient.detach()
                        self.group_gradients[s][k][i].add_(gradient)
                        self.gradients[k][i].add_(gradient)
                for j, i in enumerate(self.shared_indices):
                    if gradients[i] is not None:
                        record['gradients'][k][j].add_(gradients[i].detach())

    def stratum_report(self, weights):
        # The report uses the actual group-specific numeric factors and global rank factor.
        report = {}
        vectors = []
        for s, name in ((0, 'ordinary'), (1, 'large_gap')):
            effective = weights.clone()
            effective[:4] *= 1. if s == 0 else self.tail_factor
            record = self.strata[s]
            combined = [sum(effective[k].to(g)*record['gradients'][k][j] for k in range(len(self.names)))
                        for j, g in enumerate(record['gradients'][0])]
            vectors.append(combined)
            report[name] = dict(records=record['records'],
                weighted_loss_contributions=record['losses'].cpu().tolist(),
                task_shared_gradient_norms=[float(sum(g.double().square().sum() for g in group).sqrt())
                                           for group in record['gradients']],
                effective_task_weights=effective.cpu().tolist(),
                combined_shared_gradient_norm=float(sum(g.double().square().sum() for g in combined).sqrt()))
        denominator = report['ordinary']['combined_shared_gradient_norm']*report['large_gap']['combined_shared_gradient_norm']
        report['gradient_cosine'] = float(sum((a.double()*b.double()).sum() for a, b in zip(*vectors)))/denominator if denominator else None
        return report

    def budgeted_shared_norms(self):
        return torch.stack([sum((self.group_gradients[0][k][i].double() +
            self.tail_factor*self.group_gradients[1][k][i].double()).square().sum()
            for i in self.shared_indices).sqrt() for k in range(4)])

    def applied_loss(self, weights):
        factors = weights.new_ones(5)
        factors[:4] = self.tail_factor
        return float(((self.strata[0]['losses'] + factors*self.strata[1]['losses'])*weights).sum())


class OrdinaryGradientBudget:
    def __init__(self, decay=.95):
        if not 0 <= decay < 1:
            raise ValueError('EMA decay must be in [0,1)')
        self.decay = float(decay)
        self.reference = None
        self.updates = 0
        self.limited = 0

    def state_dict(self):
        return dict(decay=self.decay, reference=self.reference, updates=self.updates, limited=self.limited)

    def load_state_dict(self, state):
        if state['decay'] != self.decay or (state['reference'] is not None and
                not 0 < state['reference'] < float('inf')):
            raise ValueError('invalid gradient budget state')
        self.reference = state['reference']
        self.updates, self.limited = state['updates'], state['limited']

    @torch.no_grad()
    def assign(self, accumulator, requested):
        if requested.shape != (5,) or not torch.isfinite(requested).all() or (requested < 0).any():
            raise ValueError('five finite nonnegative task weights required')
        numeric = {s: [sum(requested[k]*accumulator.group_gradients[s][k][i].double() for k in range(4))
                       for i in range(len(accumulator.parameters))] for s in (0, 1)}
        rank = [sum(requested[4]*accumulator.group_gradients[s][4][i].double() for s in (0, 1))
                for i in range(len(accumulator.parameters))]
        norm = lambda vector: float(sum(v.square().sum() for v in vector).sqrt())
        ordinary, tail, rank_norm = norm(numeric[0]), norm(numeric[1]), norm(rank)
        if not all(math.isfinite(v) for v in (ordinary, tail, rank_norm)):
            raise ValueError('nonfinite grouped gradients')
        if accumulator.strata[0]['records'] and ordinary > 0:
            self.reference = ordinary if self.reference is None else self.decay*self.reference+(1-self.decay)*ordinary
        factor = min(1., self.reference/tail) if self.reference is not None and tail > 0 else 1.
        combined = [a+factor*b for a, b in zip(numeric[0], numeric[1])]
        numeric_norm = norm(combined)
        if not math.isfinite(numeric_norm):
            raise ValueError('nonfinite combined gradient')
        rank_factor = min(1., numeric_norm/rank_norm) if rank_norm > 0 else 1.
        for p, n, r in zip(accumulator.parameters, combined, rank):
            p.grad = (n+rank_factor*r).to(p)
        accumulator.tail_factor = factor
        applied = requested.clone()
        applied[4] *= rank_factor
        self.updates += 1
        self.limited += int(factor < 1.)
        return applied, dict(ordinary_numeric_gradient_norm=ordinary, raw_tail_numeric_gradient_norm=tail,
            ordinary_gradient_reference=self.reference, ordinary_reference_fallback=self.reference is None,
            tail_gradient_factor=factor, applied_tail_numeric_gradient_norm=tail*factor,
            numeric_gradient_norm=numeric_norm, requested_rank_gradient_norm=rank_norm,
            applied_rank_gradient_norm=rank_norm*rank_factor, rank_gradient_factor=rank_factor,
            rank_gradient_limited=rank_factor < 1., applied_task_weights=applied.cpu().tolist(),
            group_budget_scope='all model parameters; numeric tasks only; before rank budget and global clipping')
