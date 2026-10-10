"""Per-Gaussian image feedback for progressive four-tier allocation.

The deployed forward is hard: q0 contributes no splat/payload/XYZ. During
training, a conditional positive-tier *counterfactual* is rendered behind a zero MaskGaussian gate
for q0 rows, so those rows can receive an image derivative. Positive-tier
gradients use straight-through nested prefix gates. These are biased local
surrogates, not derivatives of zlib bytes or exact finite differences.
"""
import time

import torch

from .data import to_raw
from .training import full_scene_step
from .allocation import deployment_probabilities


def local_mask_step(model, mask, feature_batches, id_batches, geometry, snr, kind,
                    distortion_fn, beta, rate_meter, train_codec=True, mode='replay',
                    sampling='single', rate_chunk_size=1024):
    if model.cfg.prefix_mode != 'progressive':
        raise ValueError('local mask gradient requires progressive prefixes')
    if rate_meter is None:
        raise ValueError('local mask gradient requires measured side-rate accounting')
    if sampling not in ('single', 'deployment') or rate_chunk_size < 1:
        raise ValueError('invalid allocator sampling or rate chunk size')
    if sampling == 'deployment' and len(model.cfg.rates) != 4:
        raise ValueError('exact ten-draw training currently supports q0 plus three positive tiers')
    device = next(model.parameters()).device
    started = time.perf_counter()
    batches, flat_tiers = [], []
    shadow_count = 0
    q_by_source = torch.empty(len(mask.logits), device=device, dtype=torch.long)
    with torch.no_grad():
        for ids in id_batches:
            ids = ids.to(device)
            valid = ids >= 0
            if sampling == 'deployment':
                seed = int(torch.randint(2**31, ()).item())
                q, _, _, _ = mask.sample_deployment(ids.clamp_min(0), snr, seed=seed)
                replay = torch.full((*ids.shape, 1), seed, device=device, dtype=torch.float64)
            else:
                noise = -torch.empty((*ids.shape, len(model.cfg.rates)+1), device=device).exponential_().log()
                q, _, _, _ = mask.sample(ids.clamp_min(0), snr, noise=noise)
                replay = noise.double()
            q = torch.where(valid, q, 0)
            absent = valid & (q == 0)
            shadow_selected = absent
            shadow_count += int(shadow_selected.sum())
            batches.append(torch.cat((torch.stack((q, ids), -1).double(), replay), -1))
            flat_tiers.append(q[valid].cpu())
            q_by_source[ids[valid]] = q[valid]
    measured = rate_meter.details(torch.cat(flat_tiers))
    rate_seconds = time.perf_counter() - started
    counts = torch.bincount(torch.cat(flat_tiers), minlength=len(model.cfg.rates))

    def forward(features, packed):
        q, ids = packed[..., 0].long(), packed[..., 1].long()
        valid = ids >= 0
        if sampling == 'deployment':
            _, presence, conditional, positive = mask.sample_deployment(
                ids.clamp_min(0), snr, seed=int(packed.reshape(-1, 3)[0, 2].item()))
        else:
            _, presence, conditional, positive = mask.sample(ids.clamp_min(0), snr,
                                                             noise=packed[..., 2:].to(features))
        keep = valid & (q > 0)
        absent = valid & (q == 0)
        # Do not let hypothetical q0 points alter the live codec's attention
        # context. Its forward must see the actual delivered map.
        probabilities = torch.cat(((q == 0).to(conditional)[..., None],
                                   conditional * (q > 0)[..., None]), -1)
        pred = model.forward_st_prefix_batches(features, features[..., :3], q,
                                               probabilities, snr, kind)
        hard_presence = (q > 0).to(presence)
        st_presence = hard_presence + (presence - presence.detach())
        rows, gates = [to_raw(pred[keep], geometry, model)], [st_presence[keep]]
        if absent.any():
            shadow_q = torch.where(absent, positive, q)
            with torch.no_grad():
                choices = torch.nn.functional.one_hot(shadow_q, len(model.cfg.rates)).to(features)
                shadow, _, _ = model.forward_tier_batches(features, features[..., :3], choices,
                                                          snr, kind, paired_noise=True)
            rows.append(to_raw(shadow[absent], geometry, model))
            gates.append(st_presence[absent])
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
    if mask.keep_logits.grad is None:
        raise RuntimeError('masked render did not supply gradients to allocation logits')
    def gradients():
        return torch.cat([p.grad.detach() if p.grad is not None else torch.zeros_like(p)
                          for p in (mask.keep_logits, mask.logits)], -1)
    image_gradient = gradients().clone()
    image_grad_q0 = image_gradient[q_by_source == 0]
    image_grad_positive = image_gradient[q_by_source > 0]

    # Exact differentiable expectation for payload. Compressed XYZ is not
    # separable by point, so use its measured per-retained-row average as a
    # detached local proxy; always report/validate actual compressed bytes.
    retained = int(counts[1:].sum())
    xyz_unit = (measured['position_channel_uses_estimate'] / retained
                if retained else rate_meter.position_unit)
    tier_uses = measured['tier_map_proxy_bytes'] * 8 / rate_meter.bits / rate_meter.count
    expected_payload = torch.zeros((), device=device)
    expected_xyz = torch.zeros_like(expected_payload)
    mean_keep, deployment_keep = torch.zeros_like(expected_payload), torch.zeros_like(expected_payload)
    conditional_mean = torch.zeros(len(model.cfg.rates)-1, device=device)
    rate_started = time.perf_counter()
    # Release each count-pattern graph immediately: never retain N x 286 x Q
    # for an entire multi-million-Gaussian scene.
    for start in range(0, len(mask.logits), rate_chunk_size):
        ids = torch.arange(start, min(start+rate_chunk_size, len(mask.logits)), device=device)
        # Detach the gathered branch scores into small local leaves. A normal
        # indexed parameter backward allocates a dense N-row gradient for EVERY
        # chunk, making a supposedly bounded calculation quadratic in N/chunk.
        keep, tier = mask.branch_scores(ids, snr)
        keep = keep.detach().requires_grad_(True)
        tier = tier.detach().requires_grad_(True)
        keep_log = keep.log_softmax(-1)
        p = torch.cat((keep_log[:, :1], keep_log[:, 1:]+tier.log_softmax(-1)), -1).softmax(-1)
        deployed = deployment_probabilities(p) if sampling == 'deployment' else p
        payload_part = (deployed * p.new_tensor(model.cfg.rates)).sum()/len(mask.logits)
        xyz_part = xyz_unit*(1-deployed[:, 0]).sum()/len(mask.logits)
        score_gradients = torch.autograd.grad(beta*(payload_part+xyz_part)/rate_meter.normalizer, (keep, tier))
        for parameter, gradient in zip((mask.keep_logits, mask.logits), score_gradients):
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad[start:start+len(ids)].add_(gradient)
        if mask.snr_slopes is not None:
            for parameter, gradient in zip((mask.keep_snr_slopes, mask.snr_slopes), score_gradients):
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad[start:start+len(ids)].add_(gradient*((float(snr)-10.)/10.))
        expected_payload += payload_part.detach()
        expected_xyz += xyz_part.detach()
        mean_keep += (1-p[:, 0]).detach().sum()/len(mask.logits)
        deployment_keep += (1-deployed[:, 0]).detach().sum()/len(mask.logits)
        conditional_mean += (p[:, 1:]/(1-p[:, :1]).clamp_min(1e-12)).detach().sum(0)/len(mask.logits)
    rate_loss = beta*(expected_payload+expected_xyz+tier_uses)/rate_meter.normalizer
    rate_gradient = gradients() - image_gradient
    stats.update(mask_estimator='hierarchical hard Gumbel keep/tier + local masked-render gradient + progressive prefixes',
                 mask_gradient_caveat='biased local surrogate; compressed XYZ/tier costs measured but not differentiated exactly',
                 sampled_tier_counts=counts.tolist(),
                 allocation_sampling=sampling, allocation_draws=10 if sampling=='deployment' else 1,
                 allocation_rate_normalizer=rate_meter.normalizer,
                 allocation_rate_normalizer_definition='full q3 payload + measured XYZ + tier proxy per source Gaussian',
                 deployment_rate_expectation='exact multinomial positive mode; piecewise probability tie priority' if sampling=='deployment' else 'single draw',
                 mean_deployment_keep_probability=float(deployment_keep),
                 relaxation_temperature=1., rate_chunk_size=rate_chunk_size,
                 deployment_rate_gradient_seconds=time.perf_counter()-rate_started,
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
                 mask_keep_image_grad_norm=float(image_gradient[:, :2].norm()),
                 mask_tier_image_grad_norm=float(image_gradient[:, 2:].norm()),
                 mask_keep_rate_grad_norm=float(rate_gradient[:, :2].norm()),
                 mask_tier_rate_grad_norm=float(rate_gradient[:, 2:].norm()),
                 mean_keep_probability=float(mean_keep),
                 conditional_tier_probabilities_mean=conditional_mean.tolist(),
                 position_compression_seconds=measured.get('position_compression_seconds', 0.),
                 tier_map_compression_seconds=measured.get('tier_map_compression_seconds', 0.),
                 sampled_retained_gaussians=retained, codec_updated=train_codec)
    stats['sampled_q0_shadow_candidates'] = shadow_count
    return scene_loss + rate_loss.detach(), stats
