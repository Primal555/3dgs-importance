"""Per-Gaussian image feedback for progressive four-tier allocation.

The deployed forward is hard: q0 contributes no splat/payload/XYZ. During
training, a q1 *counterfactual* is rendered behind a zero MaskGaussian gate
for q0 rows, so those rows can receive an image derivative. Positive-tier
gradients use straight-through nested prefix gates. These are biased local
surrogates, not derivatives of zlib bytes or exact finite differences.
"""
import time

import torch
from torch.nn import functional as F

from .data import to_raw
from .training import full_scene_step


def local_mask_step(model, mask, feature_batches, id_batches, geometry, snr, kind,
                    distortion_fn, beta, rate_meter, train_codec=True, mode='replay'):
    if model.cfg.prefix_mode != 'progressive':
        raise ValueError('local mask gradient requires progressive prefixes')
    if rate_meter is None:
        raise ValueError('local mask gradient requires measured side-rate accounting')
    device = next(model.parameters()).device
    started = time.perf_counter()
    batches, flat_tiers = [], []
    shadow_count = 0
    q_by_source = torch.empty(len(mask.logits), device=device, dtype=torch.long)
    with torch.no_grad():
        for ids in id_batches:
            ids = ids.to(device)
            valid = ids >= 0
            logits = mask.scores(ids.clamp_min(0), snr)
            q = torch.distributions.Categorical(logits=logits).sample()
            q = torch.where(valid, q, 0)
            absent = valid & (q == 0)
            shadow_selected = absent
            shadow_count += int(shadow_selected.sum())
            batches.append(torch.stack((q, ids, shadow_selected.long()), -1))
            flat_tiers.append(q[valid].cpu())
            q_by_source[ids[valid]] = q[valid]
    measured = rate_meter.details(torch.cat(flat_tiers))
    rate_seconds = time.perf_counter() - started
    counts = torch.bincount(torch.cat(flat_tiers), minlength=len(model.cfg.rates))

    def forward(features, packed):
        q, ids, shadow_selected = packed.unbind(-1)
        valid = ids >= 0
        probabilities = mask.scores(ids.clamp_min(0), snr).softmax(-1)
        pred = model.forward_st_prefix_batches(features, features[..., :3], q,
                                               probabilities, snr, kind)
        keep = valid & (q > 0)
        absent = shadow_selected.bool()
        rows = [to_raw(pred[keep], geometry, model)]
        if absent.any():
            # This prediction is hypothetical only. No q0 payload/XYZ is sent.
            # Its only derivative comes from the zero-valued existence gate.
            shadow_q = torch.where(absent, 1, q)
            with torch.no_grad():
                shadow_choices = F.one_hot(shadow_q, len(model.cfg.rates)).to(features)
                shadow, _, _ = model.forward_tier_batches(
                    features, features[..., :3], shadow_choices, snr, kind,
                    paired_noise=True)
                rows.append(to_raw(shadow[absent], geometry, model))
        presence = 1 - probabilities[..., 0]
        hard_presence = (q > 0).to(presence)
        st_presence = hard_presence + presence - presence.detach()
        gates = [st_presence[keep], st_presence[absent]] if absent.any() else [st_presence[keep]]
        scene = torch.cat((torch.cat(rows), torch.cat(gates)[:, None]), -1)
        return scene, scene.sum() * 0, {}

    original_flags = [p.requires_grad for p in model.parameters()]
    if not train_codec:
        model.requires_grad_(False)
    try:
        # Every sampled q0 source row has a hard-zero counterfactual splat.
        scene_loss, stats = full_scene_step(
            model, list(zip(feature_batches, batches)), geometry, snr, kind,
            distortion_fn, attr_weight=0., mode=mode, batch_forward=forward)
    finally:
        if not train_codec:
            for parameter, flag in zip(model.parameters(), original_flags):
                parameter.requires_grad_(flag)
    if mask.logits.grad is None:
        raise RuntimeError('masked render did not supply gradients to allocation logits')
    image_gradient = mask.logits.grad.detach().clone()
    image_grad_q0 = image_gradient[q_by_source == 0]
    image_grad_positive = image_gradient[q_by_source > 0]

    # Exact differentiable expectation for payload. Compressed XYZ is not
    # separable by point, so use its measured per-retained-row average as a
    # detached local proxy; always report/validate actual compressed bytes.
    all_ids = torch.arange(len(mask.logits), device=device)
    probabilities = mask.probabilities(all_ids, snr)
    rates = probabilities.new_tensor(model.cfg.rates)
    expected_payload = (probabilities * rates).sum(-1).mean()
    retained = int(counts[1:].sum())
    xyz_unit = (measured['position_channel_uses_estimate'] / retained
                if retained else rate_meter.position_unit)
    expected_xyz = xyz_unit * (1 - probabilities[:, 0]).mean()
    tier_uses = measured['tier_map_proxy_bytes'] * 8 / rate_meter.bits / rate_meter.count
    proxy_total = expected_payload + expected_xyz + tier_uses
    rate_loss = beta * proxy_total / rate_meter.normalizer
    rate_loss.backward()
    rate_gradient = mask.logits.grad.detach() - image_gradient
    stats.update(mask_estimator='local masked-render gradient + straight-through progressive prefixes',
                 mask_gradient_caveat='biased local surrogate; compressed XYZ/tier costs measured but not differentiated exactly',
                 sampled_tier_counts=counts.tolist(),
                 expected_symbols_per_gaussian=float(expected_payload.detach()),
                 expected_xyz_proxy_uses_per_gaussian=float(expected_xyz.detach()),
                 measured_allocation_uses_per_gaussian=measured['allocation_uses_per_source_gaussian'],
                 measured_side_uses_per_gaussian=measured['allocation_side_uses_per_source_gaussian'],
                 measured_rate_penalty=beta*measured['allocation_uses_per_source_gaussian']/rate_meter.normalizer,
                 rate_loss=float(rate_loss.detach()), rate_accounting_seconds=rate_seconds,
                 mask_image_grad_norm=float(image_gradient.norm()),
                 mask_image_grad_q0_norm=float(image_grad_q0.norm()),
                 mask_image_grad_positive_norm=float(image_grad_positive.norm()),
                 mask_image_grad_nonzero_rows=int((image_gradient.norm(dim=-1) > 0).sum()),
                 mask_rate_grad_norm=float(rate_gradient.norm()),
                 position_compression_seconds=measured.get('position_compression_seconds', 0.),
                 tier_map_compression_seconds=measured.get('tier_map_compression_seconds', 0.),
                 sampled_retained_gaussians=retained, codec_updated=train_codec)
    stats['sampled_q0_shadow_candidates'] = shadow_count
    return scene_loss + rate_loss.detach(), stats
