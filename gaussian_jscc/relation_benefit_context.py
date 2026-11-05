"""Portable, original-row background identities for conditional relation values."""
import hashlib

import torch


CONTEXT_NAMES = ('q1', 'q2', 'q3', 'mixed')
MIXED_PROBABILITIES = (.1, .3, .3, .3)


def background_configurations(count, seed=42):
    """Build fixed backgrounds without consuming application or CUDA RNG state.

    The mixed assignment is independent of codec packing. Prefixes are stable
    when a larger scene is requested with the same seed. Every point draws one
    of ten equiprobable buckets, mapped to the declared 1/3/3/3 mixture.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('positive original-row scene size required')
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    buckets = torch.randint(10, (count,), generator=generator)
    mixed = torch.bucketize(buckets, torch.tensor([1, 4, 7]), right=True)
    tiers = [torch.full((count,), tier, dtype=torch.long) for tier in (1, 2, 3)]
    return [dict(name=name, tiers=value) for name, value in zip(CONTEXT_NAMES, [*tiers, mixed])]


def context_identity(configurations, seed=42):
    """Serializable tier identities shared by feature caches and teachers."""
    return dict(version='original_row_contexts_v1', seed=int(seed),
        names=[context['name'] for context in configurations],
        mixed_probabilities=list(MIXED_PROBABILITIES),
        tier_sha256=[hashlib.sha256(context['tiers'].contiguous().numpy().tobytes()).hexdigest()
                     for context in configurations])
