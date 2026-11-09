"""Independent information thresholds and calibration-only RD selection."""
import math

import torch

from .budgeted_allocation_evaluation import tier_hash


def threshold_map(scores, low, high):
    if (scores.ndim != 1 or not len(scores) or not torch.isfinite(scores).all()
            or not math.isfinite(low) or not math.isfinite(high) or low > high):
        raise ValueError('finite scores and ordered finite thresholds required')
    # Float64 keeps the below-minimum boundary distinct even for float32 inputs.
    scores = scores.detach().cpu().double()
    return 1 + (scores > low).long() + (scores > high).long()


def candidate_maps(scores, quantiles, cost):
    """Yield unique actual maps; cost includes all transmitted side streams."""
    scores = scores.detach().cpu().double()
    ordered = scores.sort().values
    if (not len(ordered) or not torch.isfinite(ordered).all() or
            sorted(set(quantiles)) != list(quantiles) or quantiles[0] != 0 or quantiles[-1] != 1):
        raise ValueError('sorted unique quantiles must start at 0 and end at 1')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in quantiles):
        raise ValueError('invalid threshold quantile')
    n = len(scores)
    def boundary(count):
        return math.nextafter(float(ordered[0]), -math.inf) if count <= 0 else float(ordered[min(count,n)-1])
    seen = set()
    def record(low, high, kind, **extra):
        q = threshold_map(scores, low, high)
        key = tier_hash(q)
        if key in seen: return None
        seen.add(key)
        return dict(kind=kind, low=low, high=high, tier_sha256=key,
                    tier_counts=torch.bincount(q, minlength=4).tolist(), **extra), q
    for i, p in enumerate(quantiles):
        for r in quantiles[i:]:
            result = record(boundary(int(p*n)), boundary(int(r*n)), 'quantile_grid',
                            requested_quantiles=[p,r])
            if result is not None: yield result
    budget = cost(torch.full((n,), 2, dtype=torch.long))['total_channel_uses']
    for p in quantiles[1:-1]:
        low = boundary(int(p*n)); n1 = int((scores <= low).sum())
        n3 = min(n-n1, (16*n1)//24)
        # Real metadata can put a payload-balanced map above the budget.
        # Shrink the high tier, and remeasure full cost after every change.
        for _ in range(32):
            high = max(low, boundary(n-n3)); q = threshold_map(scores,low,high)
            excess = cost(q)['total_channel_uses'] - budget
            if excess <= 0:
                result = record(low, high, 'q2_budget', requested_low_quantile=p)
                if result is not None: yield result
                break
            actual3 = int((q == 3).sum())
            if actual3 == 0: break
            n3 = max(0, actual3-max(1, math.ceil(excess/24)))


def select_maps(rows, betas, q2_normalized_cost):
    selected = {}
    for beta in betas:
        if not math.isfinite(beta) or beta < 0: raise ValueError('finite nonnegative beta required')
        best = min(rows, key=lambda r:(r['normalized_distortion']+beta*r['normalized_cost'],
                                      r['normalized_cost'], r['tier_sha256']))
        selected['beta_'+format(beta,'.12g').replace('.','p')] = dict(best, beta=beta,
            rd_score=best['normalized_distortion']+beta*best['normalized_cost'])
    feasible = [r for r in rows if r['normalized_cost'] <= q2_normalized_cost]
    if not feasible: raise ValueError('q2 budget set must include uniform q2')
    selected['q2_budget'] = dict(min(feasible,key=lambda r:(r['normalized_distortion'],
        r['normalized_cost'],r['tier_sha256'])), beta=None)
    return selected
