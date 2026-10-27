"""Independent paired counterfactuals selected by decision risk, not MSE alone."""
import torch
from torch.nn import functional as F
from .optimization import preserved_rng
from .successive_channel import present_scales
from .structured_evidence import risk_choices


@torch.no_grad()
def correct_components(model, features, id_batches, cache, geometry, task, scene, scheduler,
                       tier_ids, locations, components, rates, keep_parts, keep_tiers, table,
                       d_ref, price, c_ref, side_proxy, snr, kind):
    from .budgeted_allocation_kernel import raw_batches, set_rng
    n = len(scheduler.counts); device = scene.device; step = side_proxy['step']
    exact = torch.zeros(len(tier_ids), 3, dtype=torch.bool)
    keep_exact = torch.zeros(n, dtype=torch.bool)
    limit = scheduler.finite_samples
    stats = dict(finite_feedback=bool(limit), finite_corrections=0, finite_positive_corrections=0,
        finite_keep_corrections=0, finite_nonzero_pairs=0, finite_sign_disagreements=0,
        finite_zero_predictions=0, finite_absolute_error_mean=0., finite_decision_regret_mean=0.)
    if not limit: return [], stats, exact, keep_exact
    q_source = torch.zeros(n, dtype=torch.long); offsets, offset = [], 0
    for ids, entry in zip(id_batches, cache):
        valid = ids >= 0; q_source[ids[valid]] = entry[0][valid.to(device)].cpu()
        offsets.append(offset); offset += int(valid.sum())
    weight = task.distortion.lambda_s
    image = scheduler._combine(components, weight)
    # Reserve one quarter for keep/drop. Unused positive capacity goes to keep,
    # unused keep capacity to positive, respecting the total render budget.
    positive_limit = min(len(tier_ids), limit-limit//4)
    keep_limit = min(n, limit-positive_limit)
    positive_limit = min(len(tier_ids), limit-keep_limit)
    positive = risk_choices(tier_ids, image+rates, q_source, scheduler, positive_limit, step)
    interventions = [(locations[i], tier, i, 'positive', group) for i,tier,group in positive]
    # A keep comparison has one alternative, so use the same three-way compute
    # priorities directly on toggle costs. Every q0 already has gradient credit.
    raw_rates = torch.tensor(model.cfg.rates, dtype=torch.float64)
    bits = side_proxy.get('bits_per_use', 2.)
    proxy_cost = price*(raw_rates[keep_tiers]+side_proxy['xyz_per_retained']+
                        side_proxy.get('scale_per_retained', 0.))/c_ref
    keep_delta = scheduler._combine(keep_parts.cpu(), weight)+proxy_cost
    keep_delta *= torch.where(q_source > 0, -1., 1.)
    variance = (scheduler.keep_second-scheduler.keep_mean.square()).clamp_min(0).sum((1, 2)).sqrt()
    risk = variance+scheduler.error+scheduler.refresh.double()
    rotation = torch.roll(torch.arange(n), -(step % n))
    keep_action_checks = scheduler.keep_exact_count[torch.arange(n), keep_tiers-1]
    orders = [rotation[torch.argsort(keep_delta[rotation].abs(), stable=True)],
              rotation[torch.argsort(risk[rotation], descending=True, stable=True)],
              rotation[torch.argsort(keep_action_checks[rotation], stable=True)]]
    chosen = set()
    cursors = [0, 0, 0]
    for slot in range(keep_limit):
        group = (slot+step-1) % 3
        while int(orders[group][cursors[group]]) in chosen: cursors[group] += 1
        source = int(orders[group][cursors[group]]); chosen.add(source)
        for batch, ids in enumerate(id_batches):
            found = (ids == source).nonzero()
            if len(found):
                row, col = found[0].tolist(); old = int(q_source[source])
                interventions.append(((batch, row, col), 0 if old else int(keep_tiers[source]), None,
                    'keep', ('boundary', 'uncertainty', 'coverage')[group])); break
    records = []
    for (batch, row, col), new, position, branch, group in interventions:
        f = features[batch].to(device); ids = id_batches[batch].to(device); valid = ids >= 0
        q, _, y, _, rng = cache[batch]; old = int(q[row, col]); source = int(ids[row, col])
        alt = q.clone(); alt[row, col] = new
        with preserved_rng(device):
            set_rng(rng, device)
            pred = model.forward_tier_batches(f, f[..., :3], F.one_hot(alt, 4).to(f), snr, kind,
                paired_noise=True, **({'_semantic_y': y} if old > 0 and new > 0 else {}))[0]
        changed = scene.detach().clone(); raw = raw_batches(pred, geometry, model)
        start, count = offsets[batch], int(valid.sum())
        changed[start:start+count] = torch.cat((raw[valid], (alt[valid] > 0).to(raw)[:, None]), -1)
        measured = task.delta_components(changed)
        if branch == 'positive':
            predicted = components[position, new-1].clone()*d_ref
            rate_delta = float(rates[position, new-1])
            components[position, new-1] = measured/d_ref; exact[position, new-1] = True
        else:
            sign = 1 if old == 0 else -1
            predicted = keep_parts[source].clone()*d_ref*sign
            # Analytic marginal side cost; actual complete packet is measured at
            # control/selection time. No compression inside these counterfactuals.
            scales = float((present_scales(alt).sum()-present_scales(q).sum())*16/bits)
            rate_delta = price*(float(raw_rates[new]-raw_rates[old])+
                sign*side_proxy['xyz_per_retained']+scales)/c_ref
            keep_parts[source] = measured/d_ref*sign; keep_exact[source] = True
        measured_d = float(scheduler._combine(measured, weight))
        predicted_d = float(scheduler._combine(predicted, weight))
        real_rd, proxy_rd = measured_d/d_ref+rate_delta, predicted_d/d_ref+rate_delta
        informative = abs(real_rd) > 1e-14 and abs(proxy_rd) > 1e-14
        agrees = real_rd*proxy_rd > 0 if informative else None
        scheduler.finite_counts[source] += 1
        records.append(dict(source_row=source, old_tier=old, new_tier=new, feedback_kind=branch,
            selection_group=group, measured_delta_mse=float(measured[0]),
            measured_delta_structure=float(measured[1]), measured_delta_distortion=measured_d,
            first_order_delta_mse=float(predicted[0]), first_order_delta_structure=float(predicted[1]),
            first_order_delta_distortion=predicted_d, rate_delta_cost=rate_delta,
            measured_delta_rd=real_rd, first_order_delta_rd=proxy_rd,
            sign_agreement=agrees, absolute_error=abs(real_rd-proxy_rd),
            decision_regret=abs(real_rd) if agrees is False else 0., applied_to_training=True))
    stats.update(finite_corrections=len(records),
        finite_positive_corrections=sum(r['feedback_kind']=='positive' for r in records),
        finite_keep_corrections=sum(r['feedback_kind']=='keep' for r in records),
        finite_nonzero_pairs=sum(r['sign_agreement'] is not None for r in records),
        finite_sign_disagreements=sum(r['sign_agreement'] is False for r in records),
        finite_zero_predictions=sum(abs(r['first_order_delta_rd']) <= 1e-14 for r in records),
        finite_absolute_error_mean=sum(r['absolute_error'] for r in records)/max(1,len(records)),
        finite_decision_regret_mean=sum(r['decision_regret'] for r in records)/max(1,len(records)))
    for branch in ('positive', 'keep'):
        subset = [r for r in records if r['feedback_kind'] == branch]
        stats[branch+'_finite_rd_sign_errors'] = sum(r['sign_agreement'] is False for r in subset)
        stats[branch+'_finite_rd_pairs'] = sum(r['sign_agreement'] is not None for r in subset)
    return records, stats, exact, keep_exact
