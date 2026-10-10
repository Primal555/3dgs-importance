"""Allocation priors, measured rate proxy and original-row deployment records.

Existence is a prior, not a measured marginal rate-distortion benefit. This
module never substitutes opacity for a missing MaskGaussian existence mask.
"""
import json
import math
import zlib
import time
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from .allocation import DEPLOYMENT_DRAWS, deployment_tiers
from .position_delivery import training_position_cost
from .transport import pack_tiers, tier_id_bits


def load_existence_prior(spec, ply, count):
    info = {'requested': spec, 'source': 'none', 'row_order': 'original input PLY'}
    if not spec or spec == 'none':
        return None, info
    if spec in ('auto', 'ply'):
        vertex = PlyData.read(str(ply))['vertex'].data
        if not {'masks_0', 'masks_1'} <= set(vertex.dtype.names):
            if spec == 'ply':
                raise ValueError('PLY does not contain masks_0/masks_1 existence logits')
            info['reason'] = 'PLY has no mask logits; default high-retention initialization, NOT historical importance'
            return None, info
        scores = torch.from_numpy(np.stack([vertex['masks_0'], vertex['masks_1']], -1).copy()).float()
        if not torch.isfinite(scores).all():
            raise ValueError('nonfinite MaskGaussian logits')
        # scene/gaussian_model.py: LogSoftmax -> gumbel_softmax, column 0 = keep.
        prior = scores.softmax(-1)[:, 0]
        info['source'] = 'PLY masks_0/masks_1 softmax; column 0 is existence'
    else:
        prior = torch.as_tensor(np.load(spec, allow_pickle=False)).float()
        info['source'] = str(Path(spec).resolve())
        info['alignment_note'] = 'external NPY must match input PLY rows; length alone cannot prove identity'
    if prior.shape != (count,) or not torch.isfinite(prior).all() or ((prior < 0) | (prior > 1)).any():
        raise ValueError('existence prior must be finite [N] probabilities in input PLY order')
    info['quantiles'] = torch.quantile(prior, torch.tensor([0., .1, .5, .9, 1.])).tolist()
    return prior, info


class AllocationCostMeter:
    """Measured payload + XYZ stream + zlib-packed tier map, per SOURCE point.

    XYZ includes its framing; the tier proxy excludes packet JSON/bbox/model-ID
    framing. Final transmit measures the complete packet. Shared model weights
    and reliable-side-stream FEC/retransmissions are not modeled here.
    """
    def __init__(self, cfg, position_meter, count, bits_per_use):
        if count < 1 or not math.isfinite(bits_per_use) or bits_per_use <= 0:
            raise ValueError('invalid allocation rate dimensions')
        self.cfg, self.positions, self.count, self.bits = cfg, position_meter, count, bits_per_use
        full = self.details(torch.full((count,), len(cfg.rates)-1))
        self.normalizer = full['allocation_uses_per_source_gaussian']
        self.position_unit = full['position_channel_uses_estimate'] / count

    def details(self, q):
        started = time.perf_counter()
        q = q.detach().cpu().reshape(-1)
        if len(q) != self.count or q.dtype != torch.long or ((q < 0) | (q >= len(self.cfg.rates))).any():
            raise ValueError('allocation cost needs valid packet-order tiers for every source row')
        payload = int(torch.tensor(self.cfg.rates)[q].sum())
        result = training_position_cost(self.cfg, int((q > 0).sum()), self.count, payload,
                                        self.bits, self.positions.stream_bytes(q))
        position_seconds = self.positions.last_seconds
        tier_started = time.perf_counter()
        tier_bytes = len(zlib.compress(pack_tiers(q.numpy(), tier_id_bits(len(self.cfg.rates))), level=9))
        tier_seconds = time.perf_counter()-tier_started
        side = result['position_channel_uses_estimate'] + math.ceil(tier_bytes*8/self.bits)
        result.update(tier_map_proxy_bytes=tier_bytes, allocation_side_uses_per_source_gaussian=side/self.count,
                      position_compression_seconds=position_seconds, position_cost_cache_hit=self.positions.last_cache_hit,
                      tier_map_compression_seconds=tier_seconds, rate_accounting_seconds=time.perf_counter()-started,
                      allocation_uses_per_source_gaussian=(payload+side)/self.count,
                      allocation_rate_scope='payload + framed XYZ + zlib tier-map proxy; excludes packet JSON/header and shared weights')
        return result


def prior_ranked_tiers(prior, learned_tiers):
    """Same tier counts/payload; assign larger budgets to greater existence.

    Deterministic ties use original row order. This is a diagnostic heuristic,
    not an optimum or an equal TOTAL-cost comparison (XYZ compression changes).
    """
    ranking = torch.argsort(prior.cpu(), stable=True)
    result = torch.empty_like(learned_tiers.cpu())
    result[ranking] = torch.sort(learned_tiers.cpu()).values
    return result


@torch.no_grad()
def record_allocation(out, step, mask, snr, rates, prior=None, seed=42, save_probabilities=True,
                      deployment_expectation=False):
    """Snapshot and append deployment counts; arrays always ORIGINAL PLY order."""
    from .render_validation import append_json
    probs = torch.cat([mask.probabilities(torch.arange(i, min(i+65536, len(mask.logits)),
                         device=mask.logits.device), snr).cpu() for i in range(0,len(mask.logits),65536)])
    q = deployment_tiers(probs, seed)
    count = len(q)
    counts = torch.bincount(q, minlength=len(rates))
    previous_path = Path(out)/'allocation_latest'/'tiers.npy'
    previous = torch.from_numpy(np.load(previous_path).astype(np.int64)) if previous_path.exists() else None
    info = {'step': step, 'snr': snr, 'rates': list(rates), 'source_gaussians': count,
            'hard_tier_counts': counts.tolist(), 'hard_tier_shares': (counts.double()/count).tolist(),
            'expected_tier_counts': probs.double().sum(0).tolist(),
            'hard_payload_symbols': int(torch.tensor(rates)[q].sum()),
            'mean_entropy_nats': float(-(probs*probs.clamp_min(1e-30).log()).sum(-1).mean()),
            'mean_max_probability': float(probs.max(-1).values.mean()),
            'mean_q0_probability': float(probs[:, 0].mean()),
            'expected_q0_after_ten_draws': float(probs[:, 0].double().pow(DEPLOYMENT_DRAWS).sum()),
            'allocation_seed': seed, 'allocation_draws': DEPLOYMENT_DRAWS,
            'changed_since_previous': int((q != previous).sum()) if previous is not None else None,
            'decision': '10 draws; q0 iff all q0; otherwise positive mode; ties use learned probability then seeded random'}
    info['expected_tier_counts_definition'] = 'single-draw categorical probabilities, NOT ten-draw deployment'
    info['single_draw_expected_tier_counts'] = info['expected_tier_counts']
    if deployment_expectation:
        from .allocation import deployment_probabilities
        expected = torch.zeros(len(rates), device=mask.logits.device, dtype=torch.float64)
        for part in probs.split(1024):
            expected += deployment_probabilities(part.to(mask.logits.device)).double().sum(0)
        info['deployment_expected_tier_counts'] = expected.cpu().tolist()
        info['deployment_expected_payload_symbols'] = float((expected*expected.new_tensor(rates)).sum())
    if prior is not None:
        bins = (prior.cpu()*10).long().clamp(0,9)
        table = torch.bincount(bins*len(rates)+q, minlength=10*len(rates)).reshape(10,len(rates))
        info['existence_decile_tier_counts'] = table.tolist()
        info['mean_original_existence_by_tier'] = [float(prior[q == i].mean()) if (q == i).any() else None
                                                  for i in range(len(rates))]
    folder = Path(out)/'allocation_latest'
    folder.mkdir(exist_ok=True)
    np.save(folder/'tiers.npy', q.numpy().astype(np.uint8), allow_pickle=False)
    if save_probabilities:
        np.save(folder/'probabilities.npy', probs.numpy(), allow_pickle=False)
    if prior is not None and save_probabilities:
        np.save(folder/'prior_ranked_tiers.npy', prior_ranked_tiers(prior,q).numpy().astype(np.uint8))
    (folder/'allocation.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    append_json(Path(out)/'allocation_history.jsonl', info)
    print('Deployment allocation: '+', '.join(f'q{i}={n} ({n/count:.1%})' for i,n in enumerate(counts.tolist())), flush=True)
    return q, info
