"""Versioned block-relative geometry JSCC, with no clean receiver coordinates.

Seven systematic real values describe center(3), log radius(1), offset(3).
They occupy existing geometry slots, NOT metadata. Extra slots repeat offsets
and carry learned corrections. Slot 7 completes a constant-energy sphere, so
the receiver knows the amplitude scale from the tier alone. This deliberately
trades some coding freedom for a working, full-range noiseless initialization.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class BlockGeometry(nn.Module):
    radius_floor = 1e-4
    correction_bound = .25

    def __init__(self, rates, hidden):
        super().__init__()
        if min(rates[1:]) < 4:
            raise ValueError("block_relative_v4 needs at least 4 complex geometry symbols per retained row")
        self.register_buffer("rates", torch.tensor(rates), persistent=False)
        width = 2 * rates[-1]
        self.encoder = nn.Sequential(nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, width))
        # Received symbols + masks + permutation-invariant block feature + SNR/tier.
        self.decoder = nn.Sequential(nn.Linear(3 * width + 2, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        for head in (self.encoder[-1], self.decoder[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        axis = torch.full((width,), -1, dtype=torch.long)
        axis[4:7] = torch.arange(3)
        if width > 8:
            axis[8:] = torch.arange(width - 8) % 3
        self.register_buffer("offset_axis", axis, persistent=False)

    @staticmethod
    def average(values, active):
        return (values * active[..., None]).sum(-2, keepdim=True) / active.sum(-1, keepdim=True)[..., None].clamp_min(1)

    @classmethod
    def reference(cls, xyz, active):
        center = cls.average(xyz, active)
        radius = ((xyz - center).abs() * active[..., None]).amax(dim=(-2, -1), keepdim=True)
        return center, radius.clamp_min(cls.radius_floor)

    def layout(self, q, dtype):
        length = 2 * self.rates[q]
        mask = torch.arange(len(self.offset_axis), device=q.device) < length[..., None]
        data_mask = mask.clone()
        data_mask[..., 7] = False
        # |systematic + correction| <= 1.25, leaving >=19% energy in filler.
        gain = .9 * (self.rates[q] / (length - 1).clamp_min(1)).sqrt() / (1 + self.correction_bound)
        return mask.to(dtype), data_mask.to(dtype), gain[..., None]

    def encode(self, xyz, q, snr):
        active = (q > 0).to(xyz.dtype)
        center, radius = self.reference(xyz, active)
        offset = (xyz - center) / radius
        shared = torch.cat((center * 2 - 1,
                            2 * radius.log() / -math.log(self.radius_floor) + 1), -1)
        shared = shared.expand(*xyz.shape[:-1], 4)
        systematic = xyz.new_zeros((*xyz.shape[:-1], len(self.offset_axis)))
        systematic[..., :4] = shared
        for axis in range(3):
            systematic[..., self.offset_axis == axis] = offset[..., axis:axis+1]
        condition = torch.stack((q.to(xyz.dtype) / 3, torch.full_like(q, float(snr), dtype=xyz.dtype) / 20), -1)
        correction = self.correction_bound * self.encoder(torch.cat((shared, offset, condition), -1)).tanh()
        # Shared reference is an identifiable repeated signal, not learned
        # per-row coordinates whose pooled meaning could drift.
        correction = correction * (self.offset_axis >= 0).to(xyz.dtype)
        _, data_mask, gain = self.layout(q, xyz.dtype)
        data = (systematic + correction) * gain * data_mask
        remaining = self.rates[q].to(xyz.dtype) - data.square().sum(-1)
        filler = remaining.clamp_min(1e-8).sqrt() * active
        selector = F.one_hot(torch.tensor(7, device=q.device), data.shape[-1]).to(data)
        return data + filler[..., None] * selector

    def decode(self, received, q, snr):
        active = (q > 0).to(received.dtype)
        _, data_mask, gain = self.layout(q, received.dtype)
        data = received * data_mask / gain.clamp_min(1e-8)
        pooled = self.average(data, active)
        center = (pooled[..., :3] + 1) / 2
        radius = ((pooled[..., 3:4].clamp(-1, 1) - 1) * (-math.log(self.radius_floor)) / 2).exp()
        offset = []
        for axis in range(3):
            selected = (self.offset_axis == axis).to(received.dtype) * data_mask
            offset.append((data * selected).sum(-1) / selected.sum(-1).clamp_min(1))
        offset = torch.stack(offset, -1)
        condition = torch.stack((q.to(received.dtype) / 3,
                                 torch.full_like(q, float(snr), dtype=received.dtype) / 20), -1)
        correction = self.decoder(torch.cat((data, data_mask, pooled.expand_as(data), condition), -1))
        if getattr(self, 'individual', False):
            correction = self.correction_bound * correction.tanh()
        # Residual in LOCAL units; no normalized-affine bottleneck on absolute XYZ.
        return center + radius * (offset + correction)


class PilotBlockGeometry(BlockGeometry):
    """v5: useful pilot + block power normalization, deterministic local groups.

    Group boundaries depend ONLY on retained row order, never clean coordinates.
    Their size is codec configuration, not an uncounted coordinate side channel.
    Average complex energy is one per group (not necessarily per Gaussian).
    """
    def __init__(self, rates, hidden, group_size=256, pilot=.5):
        super().__init__(rates, hidden)
        if group_size < 1 or pilot <= 0:
            raise ValueError("positive geometry group size and pilot required")
        self.group_size = group_size
        self.pilot = pilot

    def _encode_groups(self, xyz, q, snr):
        # Reuse bounded systematic/residual construction, discard v4's filler.
        old = super().encode(xyz, q, snr)
        _, mask, gain = self.layout(q, xyz.dtype)
        data = old * mask / gain.clamp_min(1e-8)
        active = (q > 0).to(xyz.dtype)
        pilot_slot = F.one_hot(torch.tensor(7, device=q.device), data.shape[-1]).to(data)
        data = data + self.pilot * active[..., None] * pilot_slot
        energy = data.square().sum((-2, -1), keepdim=True)
        budget = self.rates[q].sum(-1, keepdim=True)[..., None].to(data)
        amplitude = (budget / energy.clamp_min(1e-12)).sqrt()
        return data * amplitude

    def _decode_groups(self, received, q, snr):
        active = (q > 0).to(received.dtype)
        # Only noisy pilots and known tier lengths are used by the receiver.
        estimated = self.average(received[..., 7:8], active) / self.pilot
        count = active.sum(-1, keepdim=True)[..., None].clamp_min(1)
        budget = self.rates[q].sum(-1, keepdim=True)[..., None].to(received)
        max_energy = (((2*self.rates[q]-1).clamp_min(0).to(received)
                       * (1+self.correction_bound)**2 + self.pilot**2) * active).sum(-1, keepdim=True)[..., None]
        # Physically valid amplitude bounds prevent ratio blow-ups for short,
        # noisy packets. No source statistics or sender gain cross this boundary.
        lower = (budget / max_energy.clamp_min(1e-12)).sqrt().clamp_min(1e-3)
        upper = (budget / (count*self.pilot**2)).sqrt().clamp_min(1e-3)
        estimated = estimated.maximum(lower).minimum(upper)
        _, _, legacy_gain = self.layout(q, received.dtype)
        return super().decode(received / estimated * legacy_gain, q, snr)

    def _grouped(self, values, q, snr, operation, output_dim):
        if q.ndim == 2:
            return torch.stack([self._grouped(v, t, snr, operation, output_dim)
                                for v, t in zip(values, q)])
        keep = torch.nonzero(q > 0, as_tuple=False).flatten()
        result = values.new_zeros((len(q), output_dim))
        if not len(keep):
            return result + values.sum()*0
        v, tiers = values[keep], q[keep]
        size = min(self.group_size, len(keep))
        # Avoid a one-row tail: balance group sizes (e.g. 257 -> 129+128).
        groups = (len(keep) + size - 1) // size
        size = (len(keep) + groups - 1) // groups
        padding = groups*size - len(keep)
        v = F.pad(v, (0, 0, 0, padding)).reshape(groups, size, -1)
        tiers = F.pad(tiers, (0, padding)).reshape(groups, size)
        decoded = operation(v, tiers, snr).reshape(-1, output_dim)[:len(keep)]
        return result.index_copy(0, keep, decoded)

    def encode(self, xyz, q, snr):
        return self._grouped(xyz, q, snr, self._encode_groups, len(self.offset_axis))

    def decode(self, received, q, snr):
        return self._grouped(received, q, snr, self._decode_groups, 3)


def position_objective(pred, target, active=None):
    """Equal block weighting; local precision plus global placement.

    Source radius is supervision only, never decoder input. Its floor bounds
    sensitivity. Beta=.01 is in local-radius units, not a claimed optimum.
    """
    if active is None:
        active = torch.ones_like(target[..., 0])
    with torch.no_grad():
        _, radius = BlockGeometry.reference(target, active)
    delta = pred - target.detach()
    local = F.smooth_l1_loss(delta / radius, torch.zeros_like(delta), beta=.01, reduction="none").mean(-1)
    global_term = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=.001, reduction="none").mean(-1)
    def reduce(value):
        return ((value * active).sum(-1) / active.sum(-1).clamp_min(1)).mean()
    local, global_term = reduce(local), reduce(global_term)
    return local + global_term, {"local_position_loss": float(local.detach()),
                                 "global_position_loss": float(global_term.detach())}
