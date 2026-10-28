"""Frozen-codec allocation credit in the actual mixed-tier neighborhood.

Only scene-sized actual/shadow attributes and actual semantic latents live for
one update. Two candidate rounds change ONE point per block, not all neighbors.
No compression or differentiable codec graph is needed for this VJP estimator.
"""
from contextlib import nullcontext
import math
import torch
from torch.nn import functional as F
from .data import to_raw
from .optimization import preserved_rng
from .successive_channel import present_scales


def timing(timings, name):
    return timings.measure(name) if timings is not None else nullcontext()


def rng_state(device):
    return torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state()


def set_rng(state, device):
    if device.type == 'cuda': torch.cuda.set_rng_state(state, device)
    else: torch.set_rng_state(state)


def raw_batches(pred, geometry, model):
    return to_raw(pred.reshape(-1, pred.shape[-1]), geometry, model).reshape(*pred.shape[:2], -1)


@torch.no_grad()
def mixed_point_candidates(model, features, q, target_columns, snr, kind, geometry, *,
                           semantic_y, actual, actual_rng, timings):
    columns = target_columns.to(q.device)
    active = columns >= 0; rows = torch.arange(len(q), device=q.device)
    current = q[rows, columns.clamp_min(0)]
    baseline = raw_batches(actual, geometry, model)
    # Heterogeneous alternatives batch into exactly TWO calls. Each target's
    # current action reuses baseline; other block members keep their real tier.
    for shift in (1, 2):
        tiers = (current-1+shift) % 3+1
        alternative = q.clone()
        alternative[rows[active], columns[active]] = tiers[active]
        with preserved_rng(features.device), timing(timings, 'candidate_decode_seconds'):
            set_rng(actual_rng, features.device)
            pred = model.forward_tier_batches(features, features[..., :3], F.one_hot(alternative, 4).to(features),
                snr, kind, paired_noise=True, _semantic_y=semantic_y)[0]
        delta = (raw_batches(pred, geometry, model)-baseline)*(q > 0)[..., None]*active[:, None, None]
        yield tiers.masked_fill(~active, -1), delta


def scene_gradient(task, scene, timings):
    """Match the direct arithmetic mean; free each rasterizer graph immediately."""
    if hasattr(task, 'component_gradient'):
        return task.component_gradient(scene, timings)
    if not hasattr(task, '_terms'):
        loss = task(scene)
        gradient, = torch.autograd.grad(loss, scene)
        return float(loss.detach()), gradient
    gradient = torch.zeros_like(scene); values = []; terms = iter(task._terms(scene))
    for index in range(len(task.cameras)):
        with timing(timings, f'view_{index}_render_seconds'):
            term = next(terms)
        if not torch.isfinite(term): raise RuntimeError('nonfinite render distortion')
        with timing(timings, f'view_{index}_backward_seconds'):
            part, = torch.autograd.grad(term + scene.sum()*0, scene)
            gradient.add_(part, alpha=1/len(task.cameras))
        values.append(float(term.detach()))
    task.stats = dict(view_mse=values, views_per_step=len(values))
    return sum(values)/len(values), gradient


@torch.no_grad()
def audit_interventions(model, features, q, positive, ids, columns, geometry, scene, upstream,
                        offset, actual_rng, task, snr, kind, base_loss, limit):
    """Independent finite rerenders; keep-set changes NEVER reuse actual y."""
    records = []
    valid = ids >= 0
    # Additional audit is intentionally small. It does not limit the full q0 gate path.
    targets = [tuple(v) for v in (valid & (q == 0)).nonzero().tolist()[:1]]
    targets += [(row, int(col)) for row, col in enumerate(columns.tolist()) if col >= 0]
    for row, column in targets[:limit]:
        old = int(q[row, column]); new = 0 if old else int(positive[row, column])
        for tier in ((new, (old % 3)+1, ((old+1) % 3)+1) if old else (new,)):
            if len(records) >= limit: break
            alternative = q.clone(); alternative[row, column] = tier
            with preserved_rng(features.device):
                set_rng(actual_rng, features.device)
                pred = model.forward_tier_batches(features, features[..., :3], F.one_hot(alternative, 4).to(features),
                                                  snr, kind, paired_noise=True)[0]
            changed = scene.detach().clone()
            raw = raw_batches(pred, geometry, model)
            block_scene = torch.cat((raw[valid], (alternative[valid] > 0).to(raw)[:, None]), -1)
            changed[offset:offset+int(valid.sum())] = block_scene
            loss = float(task(changed))
            tangent = float(((changed-scene.detach())*upstream).sum())
            if old==0 or tier==0:
                # The training keep estimator varies the gate, not raw
                # attributes AND the gate simultaneously. Fresh context is
                # still used above for the true finite intervention.
                flat_index = int(valid.flatten()[:row*q.shape[1]+column].sum())
                linear = float(upstream[offset+flat_index,-1])*(1 if old==0 else -1)
            else:
                linear = tangent
            records.append(dict(source_row=int(ids[row, column]), old_tier=old, new_tier=tier,
                measured_delta_mse=loss-base_loss, first_order_delta_mse=linear,
                full_scene_tangent_delta_mse=tangent,
                sign_agreement=(loss-base_loss)*linear >= 0,
                absolute_error=abs(loss-base_loss-linear)))
    return records


def budgeted_mask_step(model, table, feature_batches, id_batches, geometry, task, scheduler,
                       snr, kind, *, d_ref, price, side_proxy, timings, audit_limit=0,
                       rate_normalizer=None, budget=None, inspector=None):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('budgeted allocation requires an eval-mode, fully frozen codec')
    if rate_normalizer is not None and budget is not None:
        raise ValueError('supply one rate normalization, not both rate_normalizer and legacy budget')
    rate_normalizer = budget if rate_normalizer is None else rate_normalizer
    if (rate_normalizer is None or not math.isfinite(rate_normalizer) or rate_normalizer <= 0
            or not math.isfinite(d_ref) or d_ref <= 0 or not math.isfinite(price) or price < 0):
        raise ValueError('invalid objective normalization or rate coefficient')
    device = next(model.parameters()).device
    count = len(table.logits); q_source = torch.empty(count, dtype=torch.long)
    structured = hasattr(scheduler, 'accumulate_components')
    if structured:
        keep_tiers = torch.empty(count, dtype=torch.long)
    cache, parts, cache_bytes, q0_rows = [], [], 0, 0
    with torch.no_grad():
        for features, source_ids in zip(feature_batches, id_batches):
            f, ids = features.to(device), source_ids.to(device)
            valid = ids >= 0
            with timing(timings, 'sampling_seconds'):
                q, _, _, positive = table.sample(ids.clamp_min(0), snr)
                q = q.masked_fill(~valid, 0)
            q_source[ids[valid].cpu()] = q[valid].cpu()
            if structured:
                keep_tiers[ids[valid].cpu()] = torch.where(q > 0, q, positive)[valid].cpu()
            with timing(timings, 'semantic_encode_seconds'):
                y = model.semantic_encoder(f, f[..., :3], q > 0)
            state = rng_state(device)
            with timing(timings, 'actual_decode_seconds'):
                actual = model.forward_tier_batches(f, f[..., :3], F.one_hot(q, 4).to(f), snr, kind,
                                                    paired_noise=True, _semantic_y=y)[0]
                raw = raw_batches(actual, geometry, model)
            absent = valid & (q == 0)
            if absent.any():
                # Hypothetical q0 attributes never enter live attention context.
                shadow_q = torch.where(absent, positive, q)
                with preserved_rng(device), timing(timings, 'q0_shadow_seconds'):
                    set_rng(state, device)
                    shadow = model.forward_tier_batches(f, f[..., :3], F.one_hot(shadow_q, 4).to(f),
                                                        snr, kind, paired_noise=True)[0]
                    raw[absent] = raw_batches(shadow, geometry, model)[absent]
                q0_rows += int(absent.sum())
            parts.append(torch.cat((raw[valid], (q[valid] > 0).to(raw)[:, None]), -1))
            cache.append((q, positive, y, actual, state))
            cache_bytes += sum(x.numel()*x.element_size() for x in (q, positive, y, actual))
    scene = torch.cat(parts).detach().requires_grad_(True); del parts
    loss, upstream = scene_gradient(task, scene, timings)
    actual_stats = dict(getattr(task, 'stats', {}))
    if not torch.isfinite(upstream).all(): raise RuntimeError('nonfinite allocation scene gradient')
    keep_image = torch.zeros(count, 2, dtype=torch.float64, device=device)
    if structured:
        keep_parts = torch.zeros(count, 2, dtype=torch.float64, device=device)
        component_costs = []
    costs, selected, all_audits = [], [], []
    image_costs, rate_costs, locations = [], [], []
    evidence = hasattr(scheduler, 'select_rounds')
    with timing(timings, 'candidate_schedule_seconds'):
        if evidence:
            influence = torch.zeros(count, dtype=torch.float64)
            offset = 0
            for ids in id_batches:
                valid = ids >= 0; n = int(valid.sum())
                value = (upstream[offset:offset+n, :-1]*scene[offset:offset+n, :-1]).abs().sum(-1)
                value = value+upstream[offset:offset+n, -1].abs()
                influence[ids[valid]] = value.detach().double().cpu(); offset += n
            probabilities = table.probabilities(torch.arange(count, device=table.logits.device), snr).detach()
            rounds = scheduler.select_rounds(q_source, influence, probabilities, side_proxy['step'])
        else:
            columns, _ = scheduler.select(q_source)
            rounds = [columns]
    offset = 0
    bits = side_proxy.get('bits_per_use', 2.)
    rates = torch.tensor(model.cfg.rates, device=device, dtype=torch.float64)
    with torch.no_grad():
        for batch_index, (features, source_ids, entry) in enumerate(zip(feature_batches, id_batches, cache)):
            f, ids = features.to(device), source_ids.to(device)
            q, positive, y, actual, state = entry
            valid = ids >= 0; n = int(valid.sum())
            g = torch.zeros((*q.shape, scene.shape[-1]), device=device, dtype=upstream.dtype)
            g[valid] = upstream[offset:offset+n]
            keep_image[ids[valid], 1] = g[..., -1][valid].double()/d_ref
            if structured:
                component_g = []
                for j, gradient in enumerate(task.component_upstream):
                    part = torch.zeros_like(g); part[valid] = gradient[offset:offset+n]
                    keep_parts[ids[valid], j] = part[..., -1][valid].double()/d_ref
                    component_g.append(part)
            for columns in rounds:
                cols = columns[batch_index]
                active = cols >= 0
                if active.any():
                    rows = torch.arange(len(q), device=device)[active.to(device)]
                    col = cols[active].to(device)
                    action_image = torch.zeros(len(rows), 3, dtype=torch.float64, device=device)
                    action_rate = torch.zeros_like(action_image)
                    if structured:
                        action_parts = torch.zeros(len(rows), 3, 2, dtype=torch.float64, device=device)
                    block_scale_base = present_scales(q).sum(-1)
                    for tiers, delta in mixed_point_candidates(model, f, q, cols, snr, kind, geometry,
                            semantic_y=y, actual=actual, actual_rng=state, timings=timings):
                        image = (g[..., :-1]*delta).sum((1, 2)).double()/d_ref
                        alt = q.clone(); alt[rows, col] = tiers[rows]
                        scale_delta = (present_scales(alt).sum(-1)-block_scale_base)*16/bits
                        rate_delta = rates[tiers[rows]]-rates[q[rows, col]]+scale_delta[rows]
                        action_image[torch.arange(len(rows), device=device), tiers[rows]-1] = image[rows]
                        if structured:
                            for j, part in enumerate(component_g):
                                value = (part[..., :-1]*delta).sum((1, 2)).double()/d_ref
                                action_parts[torch.arange(len(rows), device=device), tiers[rows]-1, j] = value[rows]
                        action_rate[torch.arange(len(rows), device=device), tiers[rows]-1] = price*rate_delta/rate_normalizer
                    costs.append(action_image+action_rate); selected.append(ids[rows, col].cpu())
                    if evidence:
                        image_costs.append(action_image); rate_costs.append(action_rate)
                        if structured: component_costs.append(action_parts)
                        locations.extend((batch_index, int(r), int(c)) for r,c in zip(rows.tolist(), col.tolist()))
            cols = rounds[0][batch_index]
            if len(all_audits) < audit_limit:
                with timing(timings, 'audit_seconds'):
                    all_audits.extend(audit_interventions(model, f, q, positive, ids, cols, geometry,
                        scene, upstream, offset, state, task, snr, kind, loss, audit_limit-len(all_audits)))
            offset += n
    tier_ids = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)
    feedback_stats = {}
    if evidence:
        images = torch.cat(image_costs) if image_costs else torch.empty(0, 3, device=device, dtype=torch.float64)
        rate_values = torch.cat(rate_costs) if rate_costs else torch.empty_like(images)
        if structured:
            from .structured_allocation_finite import correct_components
            components = torch.cat(component_costs) if component_costs else torch.empty(0, 3, 2, device=device, dtype=torch.float64)
            approximate, approximate_keep = components.clone(), keep_parts.clone()
            with timing(timings, 'finite_feedback_seconds'):
                records, feedback_stats, exact, keep_exact = correct_components(model, feature_batches,
                    id_batches, cache, geometry, task, scene, scheduler, tier_ids, locations, components,
                    rate_values, keep_parts, keep_tiers, table, d_ref, price, rate_normalizer, side_proxy, snr, kind)
            costs = [scheduler.accumulate_components(tier_ids, components, rate_values, side_proxy['step'],
                task.distortion.lambda_s, approximate=approximate, exact=exact)]
            keep_image[:, 1] = scheduler.accumulate_keep(keep_tiers, keep_parts,
                table.logits.softmax(-1), side_proxy['step'], task.distortion.lambda_s,
                approximate=approximate_keep, exact=keep_exact)
            table.keep_trust = scheduler.keep_trust
            table.tier_trust = scheduler.tier_trust[tier_ids]
        else:
            from .allocation_finite import correct_actions
            with timing(timings, 'finite_feedback_seconds'):
                records, feedback_stats = correct_actions(model, feature_batches, id_batches, cache, geometry,
                    task, scene, upstream, scheduler, tier_ids, locations, images, rate_values, keep_image,
                    d_ref, snr, kind)
            costs = [scheduler.accumulate(tier_ids, images, rate_values, side_proxy['step'])]
        all_audits.extend(records)
    p = table.probabilities(torch.arange(count, device=table.logits.device), snr).detach()
    conditional = table.logits.softmax(-1).detach()
    expected_keep_payload = (conditional*rates[1:]).sum(-1)
    unit_side = side_proxy['xyz_per_retained'] + side_proxy.get('scale_per_retained', 0.)
    keep_costs = keep_image.clone()
    keep_costs[:, 1] += price*(expected_keep_payload+unit_side)/rate_normalizer
    if hasattr(scheduler, 'calibrate_keep_update'):
        scheduler.calibrate_keep_update(keep_costs)
        table.keep_trust = scheduler.keep_trust
    if inspector is not None:
        if not structured or scheduler.finite_samples:
            raise ValueError('update inspector requires calibrated history and zero training finite replacements')
        inspector(model=model,features=feature_batches,id_batches=id_batches,cache=cache,
            geometry=geometry,task=task,scene=scene,scheduler=scheduler,tier_ids=tier_ids,
            locations=locations,components=approximate,rates=rate_values,keep_parts=approximate_keep,
            keep_tiers=keep_tiers,table=table,tier_costs=costs[0],keep_costs=keep_costs,
            d_ref=d_ref,price=price,c_ref=rate_normalizer,side_proxy=side_proxy,snr=snr,kind=kind)
    expected = float((p*rates).sum() + (1-p[:, 0]).sum()*unit_side + side_proxy['fixed_uses'])
    sampled = float(rates[q_source.to(device)].sum()) + int((q_source>0).sum())*unit_side + side_proxy['fixed_uses']
    scheduler.record(tier_ids, side_proxy.get('step', 1))
    if evidence: feedback_stats.update(scheduler.diagnostics(q_source, step=side_proxy['step']))
    return dict(render_loss=loss, keep_image_costs=keep_image, keep_costs=keep_costs,
        tier_ids=tier_ids, tier_costs=torch.cat(costs) if costs else torch.empty(0, 3, device=device),
        q=q_source, sampled_tier_counts=torch.bincount(q_source, minlength=4).tolist(),
        q0_feedback_rows=q0_rows, expected_cost_proxy=expected, sampled_cost_proxy=sampled,
        candidate_codec_calls=2*sum(bool((c>=0).any()) for columns in rounds for c in columns), actual_codec_calls=len(cache),
        q0_codec_calls=sum(bool(((ids>=0)&(e[0].cpu()==0)).any()) for ids,e in zip(id_batches,cache)),
        cache_peak_mib=cache_bytes/2**20, audit=all_audits, **feedback_stats, **actual_stats)
