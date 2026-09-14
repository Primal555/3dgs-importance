"""Scene-specific four-way MaskGaussian-style categorical resource masks.

The table is indexed by ORIGINAL PLY rows, never by an attention weight. Optional
per-row SNR slopes let deployment decisions depend on the operating SNR.
"""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


def scene_fingerprint(raw):
    digest = hashlib.sha256(str(tuple(raw.shape)).encode())
    for block in raw.split(4096):
        digest.update(block.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GaussianTierMask(nn.Module):
    def __init__(self, count, existence_prior=None, snr_conditioned=False):
        super().__init__()
        if count < 1:
            raise ValueError("a scene must contain at least one Gaussian")
        p = torch.full((count,), .99) if existence_prior is None else torch.as_tensor(existence_prior).float()
        if p.shape != (count,) or not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("existence prior must be a finite probability array [N]")
        p = p.clamp(1e-5, 1 - 1e-5)
        positive = torch.tensor([.1, .3, .6])
        probabilities = torch.cat(((1 - p)[:, None], p[:, None] * positive), -1)
        self.logits = nn.Parameter(probabilities.log())
        self.snr_slopes = nn.Parameter(torch.zeros_like(self.logits)) if snr_conditioned else None

    def scores(self, indices, snr):
        values = self.logits[indices]
        if self.snr_slopes is not None:
            values = values + self.snr_slopes[indices] * ((float(snr) - 10.) / 10.)
        return values

    def probabilities(self, indices, snr):
        return self.scores(indices, snr).softmax(-1)

    def choose(self, indices, snr, temperature=1., sample=True):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        scores = self.scores(indices, snr)
        if sample:
            return F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1)
        return F.one_hot(scores.argmax(-1), 4).to(scores.dtype)


def expected_rate(probabilities, rates):
    """Expected payload COMPLEX symbols; differentiable in probabilities."""
    return (probabilities * probabilities.new_tensor(rates)).sum(-1)


def reconstruction_auxiliary(pred, seed, target, active, seed_weight=.2):
    """Historical group-average objective; new training uses losses.reconstruction_loss."""
    # No incentive to drop a point just to suppress its auxiliary penalty.
    # Render loss, channel masks and expected rate provide allocation gradients.
    groups = ((0, 3), (3, 4), (4, 7), (7, 11), (11, 14), (14, target.shape[1]))
    per_row = torch.stack([F.smooth_l1_loss(pred[:, a:b], target[:, a:b], reduction="none").mean(-1)
                           for a, b in groups if b > a]).mean(0)
    per_row = per_row + seed_weight * F.smooth_l1_loss(seed, target[:, :3], reduction="none").mean(-1)
    return (per_row * active.detach()).sum()
