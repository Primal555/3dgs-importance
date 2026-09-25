"""Allocation priors, measured rate proxy and original-row deployment records.

Existence is a prior, not a measured marginal rate-distortion benefit. This
module never substitutes opacity for a missing MaskGaussian existence mask.
"""
import json
import math
import zlib
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
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
        self.normalizer = self.details(torch.full((count,), len(cfg.rates)-1))['allocation_uses_per_source_gaussian']

    def details(self, q):
        q = q.detach().cpu().reshape(-1)
        if len(q) != self.count or q.dtype != torch.long or ((q < 0) | (q >= len(self.cfg.rates))).any():
            raise ValueError('allocation cost needs valid packet-order tiers for every source row')
        payload = int(torch.tensor(self.cfg.rates)[q].sum())
        result = training_position_cost(self.cfg, int((q > 0).sum()), self.count, payload,
                                        self.bits, self.positions.stream_bytes(q))
        tier_bytes = len(zlib.compress(pack_tiers(q.numpy(), tier_id_bits(len(self.cfg.rates))), level=9))
        side = result['position_channel_uses_estimate'] + math.ceil(tier_bytes*8/self.bits)
        result.update(tier_map_proxy_bytes=tier_bytes, allocation_side_uses_per_source_gaussian=side/self.count,
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
def record_allocation(out, step, mask, snr, rates, prior=None):
    """Snapshot and append deployment counts; arrays always ORIGINAL PLY order."""
    from .render_validation import append_json
    probs = torch.cat([mask.probabilities(torch.arange(i, min(i+65536, len(mask.logits)),
                         device=mask.logits.device), snr).cpu() for i in range(0,len(mask.logits),65536)])
    q = probs.argmax(-1)
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
            'changed_since_previous': int((q != previous).sum()) if previous is not None else None,
            'decision': 'argmax; source-specific learned logits; no hard budget guarantee'}
    if prior is not None:
        bins = (prior.cpu()*10).long().clamp(0,9)
        table = torch.bincount(bins*len(rates)+q, minlength=10*len(rates)).reshape(10,len(rates))
        info['existence_decile_tier_counts'] = table.tolist()
        info['mean_original_existence_by_tier'] = [float(prior[q == i].mean()) if (q == i).any() else None
                                                  for i in range(len(rates))]
    folder = Path(out)/'allocation_latest'
    folder.mkdir(exist_ok=True)
    np.save(folder/'tiers.npy', q.numpy().astype(np.uint8), allow_pickle=False)
    np.save(folder/'probabilities.npy', probs.numpy(), allow_pickle=False)
    if prior is not None:
        np.save(folder/'prior_ranked_tiers.npy', prior_ranked_tiers(prior,q).numpy().astype(np.uint8))
    (folder/'allocation.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    append_json(Path(out)/'allocation_history.jsonl', info)
    print('Deployment allocation: '+', '.join(f'q{i}={n} ({n/count:.1%})' for i,n in enumerate(counts.tolist())), flush=True)
    return q, info
