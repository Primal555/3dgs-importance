"""Shared non-allocation helpers for learned Gaussian JSCC training."""
import torch

from .training import codec_batch


@torch.no_grad()
def decode_batches(model, feature_batches, q_batches, snr, kind, geometry, paired_noise=False):
    device = next(model.parameters()).device
    return torch.cat([codec_batch(model, f.to(device), q.to(device), snr, kind, geometry, 0,
                                 compute_auxiliary=False, paired_noise=paired_noise)[0]
                      for f, q in zip(feature_batches, q_batches)])


def layout_schedule(rates):
    """One update per positive prefix, then one per-point mixed update."""
    return tuple(range(1, len(rates))) + (None,)


def hard_layout(ids, uniform=None, drop=.05, tier_count=4):
    if tier_count < 2 or (uniform is not None and not 1 <= uniform < tier_count):
        raise ValueError('uniform tier must be positive and present in the rate table')
    valid = ids >= 0
    q = torch.full_like(ids, uniform) if uniform is not None else torch.randint(1, tier_count, ids.shape, device=ids.device)
    if uniform is None and drop:
        q[torch.rand(ids.shape, device=ids.device) < drop] = 0
    return torch.where(valid, q, 0)
