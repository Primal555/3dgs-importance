"""Original shared footprint/two-head architecture with RD ranking supervision."""
import copy
import math

import torch
from torch.nn import functional as F

from .benefit_footprint_model import FootprintLearner, FootprintSplitNetwork
from .benefit_stage_one import make_learner as make_original
from .physical_benefit import PhysicalLearner

OBJECTIVE = 'shared_footprint_rank_gradnorm_v1'


def split_losses(prediction, truth, rates, scale, temperature, common_weight, pair_weight):
    error = (prediction-truth)/scale
    common = error.mean(-2, keepdim=True)
    # Exact decomposition of absolute + cw*common + pw*six_pair_mse.
    magnitude = ((1+2*pair_weight)*(error-common).square().mean((-2, -1))+
                 (1+common_weight+pair_weight/2)*common.square().mean((-2, -1)))
    p = torch.cat((prediction.new_zeros(*prediction.shape[:-2], 1), prediction.sum(-1)+rates), -1)
    t = torch.cat((truth.new_zeros(*truth.shape[:-2], 1), truth.sum(-1)+rates), -1)
    i, j = torch.triu_indices(4, 4, offset=1, device=p.device)
    logits = -(p[..., j]-p[..., i])/temperature
    teacher = (-(t[..., j]-t[..., i])/temperature).detach()
    probability = teacher.sigmoid()
    ranking = (F.binary_cross_entropy_with_logits(logits, probability, reduction='none')-
               F.binary_cross_entropy_with_logits(teacher, probability, reduction='none')).clamp_min(0.).mean(-1)
    return dict(magnitude=magnitude, ranking=ranking)


class FootprintRankLearner(FootprintLearner):
    def configure(self, temperature):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('positive finite RankNet temperature required')
        self.rank_temperature = float(temperature)

    def _batch_losses(self, groups):
        if not groups or any(g.get('supervision') == 'action_gap' or not g['views'] for g in groups):
            raise ValueError('complete nonempty teacher groups required')
        sizes = [len(g['views']) for g in groups]
        inputs = [v['inputs'] for g in groups for v in g['views']]
        p = self.network(*self.pack(inputs)).double()
        t = torch.stack([v['values'] for g in groups for v in g['views']]).to(p)
        rates = torch.stack([torch.as_tensor(g['rates']) for g in groups]).to(p)
        owner = torch.repeat_interleave(torch.arange(len(groups), device=p.device),
                                        torch.tensor(sizes, device=p.device))
        losses = split_losses(p, t, rates[owner], self.scales['rd'], self.rank_temperature,
                              self.common_weight, self.pair_weight)
        magnitude = torch.stack([v.mean() for v in losses['magnitude'].split(sizes)])
        mean_p = torch.stack([v.mean(0) for v in p.split(sizes)])
        mean_t = torch.stack([v.mean(0) for v in t.split(sizes)])
        ranking = split_losses(mean_p, mean_t, rates, self.scales['rd'], self.rank_temperature,
                               self.common_weight, self.pair_weight)['ranking']
        return dict(magnitude=magnitude, ranking=ranking)

    def state_dict(self):
        return dict(PhysicalLearner.state_dict(self), direction_objective=OBJECTIVE,
                    rank_temperature=self.rank_temperature, common_weight=self.common_weight,
                    pair_weight=self.pair_weight)

    def load_state_dict(self, state):
        if hasattr(self, 'rank_temperature'):
            if (state.get('direction_objective') != OBJECTIVE or
                    state['rank_temperature'] != self.rank_temperature or
                    state['common_weight'] != self.common_weight or state['pair_weight'] != self.pair_weight):
                raise ValueError('shared footprint objective resume mismatch')
            PhysicalLearner.load_state_dict(self, state)
        else:
            super().load_state_dict(state)


def make_learner(initial, config, device, temperature):
    # Match the old paired experiment's RNG consumption before constructing
    # the upgraded arm. Only data/initial.pt enters this factory.
    make_original(dict(settings=config, learner=initial['learner']), device)
    learner = FootprintRankLearner(None, device, hidden=config['hidden'], lr=config['benefit_lr'],
        dropout=config['dropout'], weight_decay=config['weight_decay'], decision_weight=config['decision_weight'],
        tie_fraction=config['tie_fraction'], common_weight=config['common_weight'], pair_weight=config['pair_weight'])
    learner.load_state_dict(copy.deepcopy(initial['learner']))
    learner.network = FootprintSplitNetwork.from_spatial(initial['learner']['network'], config['dropout']).to(device)
    learner.optimizer = torch.optim.AdamW(learner.network.parameters(), lr=config['benefit_lr'],
                                         weight_decay=config['weight_decay'])
    learner.configure(temperature)
    return learner


class UniformCycles:
    """The original footprint control's uniform shuffled record stream."""
    def __init__(self, count, batch_size, seed):
        if min(count, batch_size) < 1:
            raise ValueError('positive dataset and batch size required')
        self.count, self.batch_size = count, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.cursor, self.visits = [], 0, [0]*count

    def next(self):
        chosen = []
        while len(chosen) < self.batch_size:
            if self.cursor == len(self.order):
                self.order = torch.randperm(self.count, generator=self.generator).tolist()
                self.cursor = 0
            take = min(self.batch_size-len(chosen), len(self.order)-self.cursor)
            chosen.extend(self.order[self.cursor:self.cursor+take]); self.cursor += take
        for i in chosen:
            self.visits[i] += 1
        return chosen

    def state_dict(self):
        return copy.deepcopy(dict(count=self.count, batch_size=self.batch_size, order=self.order,
            cursor=self.cursor, visits=self.visits, generator=self.generator.get_state()))

    def load_state_dict(self, state):
        if state['count'] != self.count or state['batch_size'] != self.batch_size:
            raise ValueError('uniform sampler configuration differs')
        state = copy.deepcopy(state)
        self.order, self.cursor, self.visits = state['order'], state['cursor'], state['visits']
        self.generator.set_state(state['generator'])
