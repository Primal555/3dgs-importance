"""Scene-specific four-way MaskGaussian-style categorical resource masks.

The table is indexed by ORIGINAL PLY rows, never by an attention weight. Optional
per-row SNR slopes let deployment decisions depend on the operating SNR.
"""

import hashlib

import torch
from torch import nn


def scene_fingerprint(raw):
    digest = hashlib.sha256(str(tuple(raw.shape)).encode())
    for block in raw.split(4096):
        digest.update(block.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class GaussianTierMask(nn.Module):
    def __init__(self, count, existence_prior=None, snr_conditioned=False, tier_count=4):
        super().__init__()
        if count < 1:
            raise ValueError("a scene must contain at least one Gaussian")
        p = torch.full((count,), .99) if existence_prior is None else torch.as_tensor(existence_prior).float()
        if p.shape != (count,) or not torch.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("existence prior must be a finite probability array [N]")
        p = p.clamp(1e-5, 1 - 1e-5)
        if not 2 <= tier_count <= 256:
            raise ValueError('tier_count must be 2..256')
        positive = (torch.tensor([.1, .3, .6]) if tier_count == 4 else
                    torch.full((tier_count-1,), 1/(tier_count-1)))
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

def expected_rate(probabilities, rates):
    """Expected payload COMPLEX symbols; differentiable in probabilities."""
    return (probabilities * probabilities.new_tensor(rates)).sum(-1)
