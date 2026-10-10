"""Scene-specific four-way MaskGaussian-style categorical resource masks.

The table is indexed by ORIGINAL PLY rows, never by an attention weight. Optional
per-row SNR slopes let deployment decisions depend on the operating SNR.
"""

import hashlib
import math
from functools import lru_cache

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

    def sample_deployment(self, indices, snr, seed, temperature=1.):
        """Ten independent draws; hard deployment rule, relaxed local backward.

        Only the seed is cached by replay. Multiplying HARD-ST drop gates would
        incorrectly zero existence gradients whenever several draws retain a
        point, so the complement product uses the genuinely soft gates instead.
        """
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        keep, tier = self.branch_scores(indices, snr)
        generator = torch.Generator(device=keep.device).manual_seed(int(seed))
        noise = -torch.empty((*indices.shape, DEPLOYMENT_DRAWS, tier.shape[-1]+2),
                             device=keep.device, dtype=keep.dtype).exponential_(generator=generator).log()
        soft_keep = ((keep[..., None, :] + noise[..., :2])/temperature).softmax(-1)
        soft_tier = ((tier[..., None, :] + noise[..., 2:])/temperature).softmax(-1)
        draw_q = torch.where(soft_keep.detach().argmax(-1).bool(),
                             soft_tier.detach().argmax(-1)+1, 0)
        p = self.probabilities(indices, snr)
        tie_random = torch.rand(tier.shape, device=tier.device, dtype=tier.dtype, generator=generator)
        shape = indices.shape
        flat_draws, flat_p = draw_q.reshape(-1, DEPLOYMENT_DRAWS), p.reshape(-1, p.shape[-1])
        q = _tiers_from_draws(flat_draws, flat_p.detach(), tie_random.reshape(-1, tier.shape[-1])).reshape(shape)
        # All-q0 rows still need a positive counterfactual attribute prediction.
        tied = p[..., 1:] == p[..., 1:].max(-1, keepdim=True).values
        fallback = tie_random.masked_fill(~tied, -1).argmax(-1)+1
        positive = torch.where(q > 0, q, fallback)
        soft_presence = 1-soft_keep[..., 0].prod(-1)
        presence = (q > 0).to(soft_presence) + (soft_presence-soft_presence.detach())
        votes = (soft_keep[..., 1, None]*soft_tier).sum(-2)
        choices_soft = votes/votes.sum(-1, keepdim=True).clamp_min(torch.finfo(votes.dtype).tiny)
        choices_hard = torch.nn.functional.one_hot(positive-1, tier.shape[-1]).to(votes)
        choices = choices_hard + (choices_soft-choices_soft.detach())
        return q, presence, choices, positive

def expected_rate(probabilities, rates):
    """Expected payload COMPLEX symbols; differentiable in probabilities."""
    return (probabilities * probabilities.new_tensor(rates)).sum(-1)


DEPLOYMENT_DRAWS = 10


@lru_cache(maxsize=8)
def _count_patterns(draws):
    patterns = [(a, b, c, draws-a-b-c) for a in range(draws+1)
                for b in range(draws-a+1) for c in range(draws-a-b+1)]
    factors = [math.factorial(draws)/math.prod(math.factorial(n) for n in row) for row in patterns]
    return torch.tensor(patterns), torch.tensor(factors, dtype=torch.float64)


def deployment_probabilities(probabilities, draws=DEPLOYMENT_DRAWS):
    """Exact four-tier positive-mode distribution (286 count patterns at ten).

    Caller chunks rows and backpropagates each chunk immediately. Tie priority
    is piecewise constant in learned probabilities; an exact remaining tie is
    shared uniformly, matching seeded random tie breaking in expectation.
    Integer powers keep boundary derivatives finite without log(0) tricks.
    """
    if probabilities.ndim != 2 or probabilities.shape[-1] != 4 or not 1 <= draws <= 10:
        raise ValueError('expected [N,4] probabilities and 1..10 draws')
    counts, coefficients = _count_patterns(draws)
    counts = counts.to(probabilities.device)
    coefficients = coefficients.to(probabilities)
    mass = probabilities[:, None, :].pow(counts[None]).prod(-1)*coefficients
    tied = counts[:, 1:] == counts[:, 1:].max(-1, keepdim=True).values
    priority = probabilities.detach()[:, None, 1:].expand(-1, len(counts), -1).masked_fill(~tied, -1)
    winners = tied & (priority == priority.max(-1, keepdim=True).values)
    weights = winners.to(probabilities)/winners.sum(-1, keepdim=True)
    weights = weights * (counts[:, 0] != draws)[None, :, None]
    positive = (mass[..., None]*weights).sum(1)
    return torch.cat((probabilities[:, :1].pow(draws), positive), -1)


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
