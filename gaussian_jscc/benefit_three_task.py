"""Separate value tasks and accumulate gradients at an effective batch boundary."""
import copy

import torch
from torch.nn import functional as F

from .benefit_footprint_model import FootprintLearner, FootprintSplitNetwork
from .benefit_footprint_rank_gradnorm import UniformCycles
from .benefit_rank_gradnorm import StratifiedCycles
from .benefit_stage_one import make_learner as make_original
from .physical_benefit import PhysicalLearner

OBJECTIVE = 'footprint_common_relative_positive_rank_v1'
TASKS = ('common', 'relative', 'ranking')


def three_losses(prediction, truth, rates, scale, temperature, common_weight, pair_weight):
    error = (prediction-truth)/scale
    common = error.mean(-2, keepdim=True)
    shared = (1+common_weight+pair_weight/2)*common.square().mean((-2, -1))
    relative = (1+2*pair_weight)*(error-common).square().mean((-2, -1))
    p = prediction.sum(-1)+rates
    t = truth.sum(-1)+rates
    i, j = torch.triu_indices(3, 3, offset=1, device=p.device)
    logits = -(p[..., j]-p[..., i])/temperature
    teacher = (-(t[..., j]-t[..., i])/temperature).detach()
    probability = teacher.sigmoid()
    rank = (F.binary_cross_entropy_with_logits(logits, probability, reduction='none')-
            F.binary_cross_entropy_with_logits(teacher, probability, reduction='none')).clamp_min(0.).mean(-1)
    return dict(common=shared, relative=relative, ranking=rank)


class ThreeTaskLearner(FootprintLearner):
    objective = OBJECTIVE
    loss_function = staticmethod(three_losses)

    def _batch_losses(self, groups):
        if not groups or any(g.get('supervision') == 'action_gap' or not g['views'] for g in groups):
            raise ValueError('nonempty complete teacher groups required')
        sizes = [len(g['views']) for g in groups]
        prediction = self.network(*self.pack([v['inputs'] for g in groups for v in g['views']])).double()
        truth = torch.stack([v['values'] for g in groups for v in g['views']]).to(prediction)
        rates = torch.stack([torch.as_tensor(g['rates']) for g in groups]).to(prediction)
        owner = torch.repeat_interleave(torch.arange(len(groups), device=prediction.device),
                                        torch.tensor(sizes, device=prediction.device))
        per_view = self.loss_function(prediction, truth, rates[owner], self.scales['rd'], self.rank_temperature,
                                self.common_weight, self.pair_weight)
        result = {k: torch.stack([v.mean() for v in per_view[k].split(sizes)]) for k in ('common', 'relative')}
        mean_p = torch.stack([v.mean(0) for v in prediction.split(sizes)])
        mean_t = torch.stack([v.mean(0) for v in truth.split(sizes)])
        result['ranking'] = self.loss_function(mean_p, mean_t, rates, self.scales['rd'], self.rank_temperature,
                                         self.common_weight, self.pair_weight)['ranking']
        return result

    def state_dict(self):
        return dict(PhysicalLearner.state_dict(self), direction_objective=self.objective,
                    rank_temperature=self.rank_temperature, common_weight=self.common_weight,
                    pair_weight=self.pair_weight)

    def load_state_dict(self, state):
        if hasattr(self, 'rank_temperature'):
            if (state.get('direction_objective') != self.objective or
                    any(state[k] != getattr(self, k) for k in ('rank_temperature', 'common_weight', 'pair_weight'))):
                raise ValueError('three-task resume objective mismatch')
            PhysicalLearner.load_state_dict(self, state)
        else:
            super().load_state_dict(state)


def make_learner(initial, config, device, temperature, objective=OBJECTIVE):
    if objective == OBJECTIVE:
        learner_type = ThreeTaskLearner
    else:
        from .benefit_view_gap import ViewGapLearner, OBJECTIVE as VIEW_GAP
        if objective != VIEW_GAP: raise ValueError('unknown benefit objective')
        learner_type = ViewGapLearner
    # Same construction RNG and initial function as the existing footprint arms.
    make_original(dict(settings=config, learner=initial['learner']), device)
    learner = learner_type(None, device, hidden=config['hidden'], lr=config['benefit_lr'],
        dropout=config['dropout'], weight_decay=config['weight_decay'], decision_weight=config['decision_weight'],
        tie_fraction=config['tie_fraction'], common_weight=config['common_weight'], pair_weight=config['pair_weight'])
    learner.load_state_dict(copy.deepcopy(initial['learner']))
    learner.network = FootprintSplitNetwork.from_spatial(initial['learner']['network'], config['dropout']).to(device)
    learner.optimizer = torch.optim.AdamW(learner.network.parameters(), lr=config['benefit_lr'],
                                         weight_decay=config['weight_decay'])
    learner.rank_temperature = float(temperature)
    if not 0 < learner.rank_temperature < float('inf'):
        raise ValueError('positive finite temperature required')
    return learner


def make_sampler(labels, batch, seed, *, stratified):
    return StratifiedCycles(labels, batch, seed) if stratified else UniformCycles(len(labels), batch, seed)


def sampling_weights(labels, sampler, chosen):
    if not isinstance(sampler, StratifiedCycles):
        return torch.ones(len(chosen), dtype=torch.double)
    size = sum(sampler.quotas.values())
    factors = {s: (len(sampler.members[s])/len(labels))/(sampler.quotas[s]/size) for s in sampler.strata}
    return torch.tensor([factors[labels[i]] for i in chosen], dtype=torch.double)


class TaskAccumulator:
    """Store sums of task gradients, freeing each microbatch graph immediately."""
    def __init__(self, parameters, shared_indices, names=TASKS):
        self.parameters = list(parameters)
        self.shared_indices = list(shared_indices)
        self.names = tuple(names)
        self.gradients = [[torch.zeros_like(p) for p in self.parameters] for _ in self.names]
        self.losses = torch.zeros(len(self.names), dtype=torch.double, device=self.parameters[0].device)
        self.strata = {s: dict(records=0, losses=torch.zeros_like(self.losses),
            gradients=[[torch.zeros_like(self.parameters[i]) for i in self.shared_indices] for _ in self.names])
            for s in (0, 1)}

    def add(self, per_record, weights, labels, *, diagnostics=False):
        if per_record.ndim != 2 or per_record.shape[1] != len(self.names):
            raise ValueError('one loss per record and task required')
        weights = weights.to(per_record); labels = torch.as_tensor(labels, device=per_record.device)
        tasks = (per_record*weights[:, None]).sum(0)
        self.losses += tasks.detach()
        if diagnostics:
            shared = [self.parameters[i] for i in self.shared_indices]
            for s, record in self.strata.items():
                mask = labels == s
                if not mask.any(): continue
                contribution = (per_record[mask]*weights[mask, None]).sum(0)
                record['records'] += int(mask.sum())
                record['losses'] += contribution.detach()
                for k, value in enumerate(contribution):
                    grads = torch.autograd.grad(value, shared, retain_graph=True, allow_unused=True)
                    for buffer, grad in zip(record['gradients'][k], grads):
                        if grad is not None: buffer.add_(grad.detach())
        for k, value in enumerate(tasks):
            grads = torch.autograd.grad(value, self.parameters, retain_graph=k+1 < len(self.names), allow_unused=True)
            for buffer, grad in zip(self.gradients[k], grads):
                if grad is not None: buffer.add_(grad.detach())

    def shared_norms(self):
        return torch.stack([sum(g[i].double().square().sum() for i in self.shared_indices).sqrt()
                            for g in self.gradients])

    def assign(self, weights):
        for i, p in enumerate(self.parameters):
            p.grad = sum(weights[k].to(p)*self.gradients[k][i] for k in range(len(self.names)))

    def stratum_report(self, weights):
        report = {}
        vectors = []
        for s, name in ((0, 'ordinary'), (1, 'large_gap')):
            record = self.strata[s]
            combined = [sum(weights[k].to(g)*record['gradients'][k][j] for k in range(len(self.names)))
                        for j, g in enumerate(record['gradients'][0])]
            vectors.append(combined)
            report[name] = dict(records=record['records'], weighted_loss_contributions=record['losses'].cpu().tolist(),
                task_shared_gradient_norms=[float(sum(g.double().square().sum() for g in group).sqrt())
                                           for group in record['gradients']],
                combined_shared_gradient_norm=float(sum(g.double().square().sum() for g in combined).sqrt()))
        dot = sum((a.double()*b.double()).sum() for a, b in zip(*vectors))
        denom = report['ordinary']['combined_shared_gradient_norm']*report['large_gap']['combined_shared_gradient_norm']
        report['gradient_cosine'] = float(dot)/denom if denom else None
        return report
