"""Independent common/relative heads with a zero-initialized region branch."""
import copy

import torch
from torch import nn

from .benefit_footprint_features import CHANNELS
from .neural_correction_learning import CorrectionLearner
from .spatial_benefit import SpatialBenefitNetwork, pack_inputs


def footprint_pack(inputs, device):
    return (*pack_inputs(inputs, device), torch.stack([x['footprints'] for x in inputs]).to(device, torch.float32))


class FootprintSplitNetwork(SpatialBenefitNetwork):
    def __init__(self, hidden=256, dropout=.05):
        super().__init__(hidden, dropout)
        width = self.state_head[0].in_features
        self.common_head = copy.deepcopy(self.state_head)
        self.relative_head = copy.deepcopy(self.state_head)
        del self.state_head
        self.profile_encoder = nn.Sequential(nn.Conv2d(CHANNELS, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.SiLU(), nn.Flatten(), nn.Linear(24*5*5, 32), nn.SiLU())
        self.profile_projection = nn.Linear(64, width, bias=False)
        nn.init.zeros_(self.profile_projection.weight)

    @classmethod
    def from_spatial(cls, state, dropout=.05):
        network = cls(state['points.0.weight'].shape[0], dropout)
        values = network.state_dict()
        for name, value in state.items():
            if name.startswith('state_head.'):
                tail = name[len('state_head.'):]
                values['common_head.'+tail] = value.clone()
                values['relative_head.'+tail] = value.clone()
            else:
                values[name] = value.clone()
        network.load_state_dict(values, strict=True)
        return network

    def forward(self, features, context, states, weights, valid, footprints):
        x = self.state_embeddings(features, context, states, weights, valid)
        b = len(x)
        profile = self.profile_encoder(torch.asinh(footprints).reshape(-1, CHANNELS, 9, 9))
        profile = profile.reshape(b, 4, 64)
        x = x+self.profile_projection(profile)
        common_scores = self.common_head(x)*self.output_scale
        common = common_scores[:, 1:].mean(1, keepdim=True)-common_scores[:, :1]
        scores = self.relative_head(x[:, 1:])*self.output_scale
        relative = scores-scores.mean(1, keepdim=True)
        result = common+relative
        identical = ((features[:, 1:] == features[:, :1]).all(-1) &
            (states[:, 1:] == states[:, :1]).flatten(-2).all(-1) &
            (footprints[:, 1:] == footprints[:, :1]).flatten(2).all(-1))
        return result.masked_fill(identical[..., None], 0.)


class FootprintLearner(CorrectionLearner):
    def pack(self, inputs):
        return footprint_pack(inputs, self.device)
