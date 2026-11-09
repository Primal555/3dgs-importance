"""Fixed primary value task with separately supervised response components."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .benefit_footprint_model import FootprintSplitNetwork, footprint_pack
from .benefit_view_gap import view_gap_losses

VERSION = 'footprint_response_terms_v1'
TASKS = ('value','cross','energy','structure','ranking')


def combine_terms(terms):
    return torch.stack((terms[...,0]+terms[...,1],terms[...,2]),-1)


class ResponseTermsNetwork(FootprintSplitNetwork):
    def __init__(self, hidden=256, dropout=.05):
        super().__init__(hidden,dropout)
        self.energy_head = copy.deepcopy(self.relative_head)
        self.energy_head[-1] = nn.Linear(self.relative_head[-1].in_features,1)
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.constant_(self.energy_head[-1].bias, math.log(math.expm1(.01)))
        self.register_buffer('energy_scale',torch.ones(()))

    @classmethod
    def from_spatial(cls,state,dropout=.05,energy_scale=1.):
        if not math.isfinite(float(energy_scale)) or energy_scale <= 0:
            raise ValueError('positive energy scale required')
        network = super().from_spatial(state,dropout)
        for target,source in zip(network.energy_head[:-1],network.relative_head[:-1]):
            target.load_state_dict(source.state_dict())
        network.energy_scale.fill_(energy_scale)
        return network

    def forward_terms(self,features,context,states,weights,valid,footprints):
        x = self.state_embeddings(features,context,states,weights,valid)
        profile = self.profile_encoder(torch.asinh(footprints).reshape(-1,15,9,9)).reshape(len(x),4,64)
        x = x+self.profile_projection(profile)
        common_scores = self.common_head(x)*self.output_scale
        common = common_scores[:,1:].mean(1,keepdim=True)-common_scores[:,:1]
        scores = self.relative_head(x[:,1:])*self.output_scale
        value = (common+(scores-scores.mean(1,keepdim=True))).double()
        energy = F.softplus(self.energy_head(x[:,1:]).squeeze(-1)).double()*self.energy_scale.double()
        terms = torch.stack((value[...,0]-energy,energy,value[...,1]),-1)
        identical = ((features[:,1:] == features[:,:1]).all(-1) &
            (states[:,1:] == states[:,:1]).flatten(-2).all(-1) &
            (footprints[:,1:] == footprints[:,:1]).flatten(2).all(-1))
        return terms.masked_fill(identical[...,None],0.)

    def forward(self,*args):
        return combine_terms(self.forward_terms(*args))


def response_losses(predicted_terms, truth, rates, rd_scale, term_scales, temperature, common_weight, pair_weight):
    values, targets = combine_terms(predicted_terms),combine_terms(truth)
    core = view_gap_losses(values,targets,rates,rd_scale,temperature,common_weight,pair_weight)
    component = ((predicted_terms-truth)/term_scales).square().mean(-2)
    return dict(value=core['common']+core['relative'],cross=component[...,0],
                energy=component[...,1],structure=component[...,2],ranking=core['ranking'])


def task_weights(auxiliary_weights):
    if (auxiliary_weights.shape != (3,) or not torch.isfinite(auxiliary_weights).all() or
            (auxiliary_weights<0).any() or auxiliary_weights.sum()<=0):
        raise ValueError('three finite nonnegative auxiliary weights required')
    return torch.cat((auxiliary_weights.new_ones(1),
                      auxiliary_weights/auxiliary_weights.sum().clamp_min(1e-30),auxiliary_weights.new_ones(1)))


@torch.no_grad()
def assign_gradients(accumulator,requested):
    numeric_square = requested.new_zeros((),dtype=torch.double)
    rank_square = numeric_square.clone()
    for i in range(len(accumulator.parameters)):
        numeric = sum(requested[k]*accumulator.gradients[k][i].double() for k in range(4))
        rank = requested[4]*accumulator.gradients[4][i].double()
        numeric_square += numeric.square().sum(); rank_square += rank.square().sum()
    n,r = numeric_square.sqrt(),rank_square.sqrt()
    if not torch.isfinite(n+r): raise ValueError('nonfinite accumulated gradients')
    factor = min(1.,float(n)/float(r)) if r>0 else 1.
    applied = requested.clone(); applied[-1] *= factor
    accumulator.assign(applied)
    return applied,dict(numeric_gradient_norm=float(n),requested_rank_gradient_norm=float(r),
        applied_rank_gradient_norm=float(r)*factor,rank_gradient_factor=factor,
        rank_gradient_limited=factor<1.,applied_task_weights=applied.cpu().tolist())
