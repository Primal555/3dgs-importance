"""Sender-only conditional probability scores for a frozen latent representation."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def logistic_log_mass(value, location, scale, width):
    """log(sigmoid(b)-sigmoid(a)) without subtracting saturated CDFs.

    Keep interval width explicit: forming b-a from large endpoints loses it.
    Double precision here preserves tiny tail mass and its parameter gradients.
    """
    value, location, scale = value.double(), location.double(), scale.double()
    a = (value-location-width/2)/scale
    w = width/scale
    return F.logsigmoid(a+w)+F.logsigmoid(-a)+torch.log(-torch.expm1(-w))


def split_blocks(block_ids, fraction=.2, seed=42):
    if block_ids.ndim != 1 or (block_ids < 0).any() or not 0 < fraction < 1:
        raise ValueError('valid original block IDs and validation fraction required')
    unique = block_ids.unique(sorted=True)
    if len(unique) < 2: raise ValueError('at least two codec blocks required for isolated validation')
    count = min(len(unique)-1, max(1, round(len(unique)*fraction)))
    order = torch.randperm(len(unique), generator=torch.Generator().manual_seed(seed))
    valid = torch.isin(block_ids, unique[order[:count]])
    return (~valid).nonzero().flatten(), valid.nonzero().flatten()


def fit_statistics(y, train, chunk=8192):
    if y.ndim != 2 or train.ndim != 1 or not len(train):
        raise ValueError('nonempty training rows required')
    count = 0; mean = torch.zeros(y.shape[1], dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    for ids in train.split(chunk):
        x = y[ids].double().cpu()
        if not torch.isfinite(x).all(): raise ValueError('nonfinite training latent')
        n = len(x); center = x.mean(0); delta = center-mean
        m2 += (x-center).square().sum(0)+delta.square()*(count*n/(count+n))
        mean += delta*(n/(count+n)); count += n
    return mean.float(), (m2/count).clamp_min(0).sqrt().clamp_min(1e-6).float()


class GroupedEntropyModel(nn.Module):
    """Three-component logistic mixtures with causal channel-group conditions.

    Quantization belongs only to this statistical scoring branch. Actual codec
    inputs and its 48-dimensional continuous representation are never changed.
    """
    def __init__(self, groups=(16, 16, 16), hidden=128, mixtures=3, bin_width=.25):
        super().__init__()
        if (not groups or any(type(n) is not int or n < 1 for n in groups)
                or hidden < 1 or mixtures < 1 or not math.isfinite(bin_width) or bin_width <= 0):
            raise ValueError('invalid conditional entropy configuration')
        self.groups = tuple(groups); self.dim = sum(groups)
        self.hidden, self.mixtures, self.bin_width = hidden, mixtures, float(bin_width)
        self.register_buffer('mean', torch.zeros(self.dim))
        self.register_buffer('std', torch.ones(self.dim))
        self.conditioners = nn.ModuleList()
        self.first = nn.Parameter(self._initial(groups[0]))
        prefix = groups[0]
        for size in groups[1:]:
            net = nn.Sequential(nn.Linear(prefix, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, size*3*mixtures))
            with torch.no_grad():
                nn.init.normal_(net[-1].weight, std=.01)
                net[-1].bias.copy_(self._initial(size).flatten())
            self.conditioners.append(net); prefix += size

    def _initial(self, size):
        result = torch.zeros(size, 3, self.mixtures)
        result[:, 1] = torch.linspace(-1, 1, self.mixtures)
        result[:, 2] = math.log(math.expm1(.7))
        return result

    def configuration(self):
        return dict(groups=list(self.groups), hidden=self.hidden, mixtures=self.mixtures,
                    bin_width=self.bin_width, family='grouped_logistic_mass_v1', scale_floor=.01)

    def parameters_for_group(self, standardized, group):
        if group == 0: raw = self.first.expand(len(standardized), -1, -1, -1)
        else:
            prefix = sum(self.groups[:group])
            raw = self.conditioners[group-1](standardized[:, :prefix]).reshape(
                len(standardized), self.groups[group], 3, self.mixtures)
        logits, locations, rho = raw.unbind(-2)
        return logits, locations, F.softplus(rho)+.01

    def values(self, latent):
        u = (latent.detach().to(self.mean)-self.mean)/self.std
        return torch.round(u/self.bin_width)*self.bin_width

    def component_bits(self, latent):
        if latent.ndim != 2 or latent.shape[1] != self.dim:
            raise ValueError('latent dimensionality differs from entropy model')
        x = self.values(latent); offset = 0; pieces = []
        for g, size in enumerate(self.groups):
            logits, location, scale = self.parameters_for_group(x, g)
            mass = logistic_log_mass(x[:, offset:offset+size, None], location, scale, self.bin_width)
            logp = torch.logsumexp(F.log_softmax(logits.double(), -1)+mass, -1)
            pieces.append(-logp/math.log(2)); offset += size
        return torch.cat(pieces, -1)

    def bits(self, latent): return self.component_bits(latent).sum(-1)


def tier_map(scores, eta, rates):
    if (scores.ndim != 1 or not torch.isfinite(scores).all() or (scores < 0).any()
            or not math.isfinite(eta) or eta <= 0 or len(rates) < 2 or rates[0] != 0
            or any(type(v) is not int for v in rates) or any(a >= b for a, b in zip(rates, rates[1:]))):
        raise ValueError('finite nonnegative information scores, positive eta and increasing rates required')
    boundaries = torch.tensor(rates[1:], device=scores.device, dtype=torch.float64)
    return torch.searchsorted(boundaries, scores.double()*eta).clamp_max(len(rates)-2)+1


def eta_grid(scores, rates, count=9):
    positive = scores[scores > 0].double()
    if count < 2 or not len(positive): raise ValueError('nonzero scores and at least two eta candidates required')
    low, high = torch.quantile(positive, torch.tensor([.05, .95], dtype=torch.float64))
    start = .8*rates[1]/float(high)
    end = 1.2*rates[-2]/float(low)
    return torch.logspace(math.log10(start), math.log10(end), count, dtype=torch.float64).tolist()
