"""Opt-in amplitude-preserving power controls; no receiver scale side channel."""
import math

import torch
from torch import nn


class SegmentPowerNormalizer(nn.Module):
    """Normalize segments using population EMA or ORIGINAL spatial blocks.

    Each row of [blocks, points, reals] is one original codec block. Statistics
    are transmitter state. The receiver gets only the transmitted symbols;
    it never receives the current point/block energy or an inverse gain.
    """
    def __init__(self, rates, mode, *, floor=.01, momentum=.01):
        super().__init__()
        if (mode not in ('statistical', 'block') or not math.isfinite(floor) or floor <= 0
                or not math.isfinite(momentum) or not 0 < momentum <= 1):
            raise ValueError('invalid segment power normalization')
        self.rates = tuple(rates)
        self.mode, self.floor, self.momentum = mode, float(floor), float(momentum)
        if mode == 'statistical':
            self.register_buffer('running_energy', torch.ones(len(rates)-1))

    @torch.no_grad()
    def set_population_energy(self, values):
        if self.mode != 'statistical':
            raise ValueError('population calibration requires statistical power mode')
        values = torch.as_tensor(values, device=self.running_energy.device, dtype=self.running_energy.dtype)
        if values.shape != self.running_energy.shape or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError('finite nonnegative segment energies required')
        self.running_energy.copy_(values)

    def forward(self, raw, active):
        if raw.ndim != 3 or active.shape != raw.shape[:2] or active.dtype != torch.bool:
            raise ValueError('power normalization requires original [blocks,points,reals] and bool mask')
        raw = raw.masked_fill(~active[..., None], 0)
        outputs = []
        for index, (start, end) in enumerate(zip(self.rates[:-1], self.rates[1:])):
            segment = raw[..., 2*start:2*end]
            energy = segment.square().sum(-1, keepdim=True)/(end-start)
            if self.mode == 'block':
                denominator = energy.sum(1, keepdim=True)/active.sum(1, keepdim=True).clamp_min(1)[..., None]
            else:
                if self.training:
                    with torch.no_grad():
                        count = active.sum()
                        if count:
                            self.running_energy[index].lerp_(energy.sum()/count, self.momentum)
                # Clone the detached scalar: future EMA updates must not mutate
                # a denominator still needed by an outstanding backward pass.
                denominator = self.running_energy[index].detach().clone()
            outputs.append(segment/(denominator+self.floor).sqrt())
        return torch.cat(outputs, -1).masked_fill(~active[..., None], 0)
