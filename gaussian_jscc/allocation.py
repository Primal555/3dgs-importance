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
        # Keep the conservative q0 prior, but do not privilege the largest
        # positive prefix before any per-Gaussian evidence has been learned.
        positive = torch.full((tier_count-1,), 1/(tier_count-1))
        self.keep_logits = nn.Parameter(torch.stack((1-p, p), -1).log())
        self.logits = nn.Parameter(positive.log().expand(count, -1).clone())
        self.snr_slopes = nn.Parameter(torch.zeros_like(self.logits)) if snr_conditioned else None
        self.keep_snr_slopes = nn.Parameter(torch.zeros_like(self.keep_logits)) if snr_conditioned else None

    def scores(self, indices, snr):
        keep, tier = self.branch_scores(indices, snr)
        keep = keep.log_softmax(-1)
        return torch.cat((keep[..., :1], keep[..., 1:] + tier.log_softmax(-1)), -1)

    def branch_scores(self, indices, snr):
        values = self.logits[indices]
        keep = self.keep_logits[indices]
        if self.snr_slopes is not None:
            values = values + self.snr_slopes[indices] * ((float(snr) - 10.) / 10.)
            keep = keep + self.keep_snr_slopes[indices] * ((float(snr) - 10.) / 10.)
        return keep, values

    def sample(self, indices, snr, temperature=1., noise=None):
        """Hard forward, stochastic relaxed backward; replay preserves RNG."""
        keep, tier = self.branch_scores(indices, snr)
        def draw(scores, perturbation):
            if perturbation is None:
                return torch.nn.functional.gumbel_softmax(scores, tau=temperature, hard=True)
            soft = ((scores + perturbation) / temperature).softmax(-1)
            hard = torch.nn.functional.one_hot(soft.argmax(-1), scores.shape[-1]).to(soft)
            return hard + (soft - soft.detach())
        presence = draw(keep, None if noise is None else noise[..., :2])[..., 1]
        choices = draw(tier, None if noise is None else noise[..., 2:])
        positive = choices.detach().argmax(-1) + 1
        q = torch.where(presence.detach().bool(), positive, 0)
        return q, presence, choices, positive

    def probabilities(self, indices, snr):
        return self.scores(indices, snr).softmax(-1)

def expected_rate(probabilities, rates):
    """Expected payload COMPLEX symbols; differentiable in probabilities."""
    return (probabilities * probabilities.new_tensor(rates)).sum(-1)


DEPLOYMENT_DRAWS = 10


def _tiers_from_draws(draws, probabilities, tie_random):
    """Drop only after ten q0 draws; otherwise use the positive modal tier.

    Tied positive counts are resolved by their learned probabilities, then by
    seeded random numbers so equal initial tiers remain symmetric.
    """
    counts = torch.stack([(draws == tier).sum(-1)
                          for tier in range(probabilities.shape[1])], dim=-1)
    positive_counts = counts[:, 1:]
    tied = positive_counts == positive_counts.max(-1, keepdim=True).values
    tied_probabilities = probabilities[:, 1:].masked_fill(~tied, -1)
    tied = tied & (tied_probabilities == tied_probabilities.max(-1, keepdim=True).values)
    positive = tie_random.masked_fill(~tied, -1).argmax(-1) + 1
    return torch.where(counts[:, 0] == DEPLOYMENT_DRAWS, 0, positive)


@torch.no_grad()
def deployment_tiers(probabilities, seed=42):
    """Sample one reproducible ten-draw tier map in original source-row order.

    CPU sampling makes validation, export and transmission agree across devices.
    The resulting hard map must be sent; receivers do not resample it.
    """
    probabilities = probabilities.detach().cpu()
    if (probabilities.ndim != 2 or probabilities.shape[1] < 2 or
            not torch.isfinite(probabilities).all() or
            (probabilities < 0).any() or
            not torch.allclose(probabilities.sum(-1), torch.ones(len(probabilities)), atol=1e-5)):
        raise ValueError('deployment requires finite categorical probabilities [N,Q]')
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    parts = []
    for start in range(0, len(probabilities), 65536):
        p = probabilities[start:start+65536]
        draws = torch.rand((len(p), DEPLOYMENT_DRAWS), generator=generator)
        cdf = p.cumsum(-1).contiguous()
        cdf[:, -1] = 1.
        sampled = torch.searchsorted(cdf, draws).clamp_max(p.shape[1]-1)
        tie_random = torch.rand((len(p), p.shape[1]-1), generator=generator)
        parts.append(_tiers_from_draws(sampled, p, tie_random))
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
