"""Scale one Adam displacement using measured policy-output divergence."""
import copy
import math
import torch
from .student_rd_audit_tools import take_step


def output_divergence(before, after):
    result = {}
    for name, old, new in zip(('keep', 'tier'), before, after):
        old_log = old.detach().cpu().double().log_softmax(-1)
        new_log = new.detach().cpu().double().log_softmax(-1)
        values = (old_log.exp()*(old_log-new_log)).sum(-1).clamp_min(0)
        if not torch.isfinite(values).all(): raise RuntimeError('nonfinite policy output')
        result[name] = dict(mean=float(values.mean()), p95=float(values.quantile(.95)), max=float(values.max()))
    return result


def output_controlled_step(student, optimizer, losses, clip, *, before, heads, diagnostic,
                           keep_kl=2e-5, tier_kl=2e-5):
    if min(keep_kl, tier_kl) <= 0 or not all(math.isfinite(x) for x in (keep_kl, tier_kl)):
        raise ValueError('output KL budgets must be positive and finite')
    parameters = list(student.parameters()); old = [p.detach().clone() for p in parameters]
    # Adam advances exactly once. Extra forwards never update moments or its step counter.
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    try:
        if diagnostic:
            record = take_step(student, optimizer, losses, clip)
        else:
            optimizer.zero_grad(set_to_none=True); sum(losses.values()).backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
            optimizer.step(); record = dict(grad_norm_before_clip=float(norm))
        delta = [p.detach()-v for p, v in zip(parameters, old)]
        after = heads(); proposal = output_divergence(before, after)
        observed = proposal; scale = 1.; forwards = 1
        for attempt in range(17):
            ratio = max(observed['keep']['mean']/keep_kl, observed['tier']['mean']/tier_kl)
            if ratio <= 1+1e-9: break
            if attempt == 16:
                raise RuntimeError('policy output displacement failed to enter its numerical trust region')
            scale *= min(.8, .9/math.sqrt(ratio))
            with torch.no_grad():
                for p, start, direction in zip(parameters, old, delta): p.copy_(start+scale*direction)
            after = heads(); observed = output_divergence(before, after); forwards += 1
        if diagnostic:
            record['gradient_dot_parameter_step'] = {k:v*scale for k,v in record['gradient_dot_parameter_step'].items()}
            record['parameter_step_norm'] *= scale
            record['directional_product_scope'] = 'gradient dotted with uniformly scaled Adam displacement'
        old_p = before[1].double().softmax(-1)
        margin = old_p.topk(2, dim=-1).values; margin = margin[:, 0]-margin[:, 1]
        changed = before[1].argmax(-1) != after[1].argmax(-1)
        record['output_step'] = dict(applied_scale=scale, proposal=proposal, accepted=observed,
            output_check_forwards=forwards, backtrack_forwards=forwards-1,
            keep_kl_budget=keep_kl, tier_kl_budget=tier_kl,
            positive_winner_changes=int(changed.sum()),
            changed_positive_margin_mean=float(margin[changed].mean()) if changed.any() else None,
            decision_scope='all source points; output KL controls displacement, hard-map RD measured separately',
            optimizer_steps=1)
        return record
    except BaseException:
        with torch.no_grad():
            for p, value in zip(parameters, old): p.copy_(value)
        optimizer.load_state_dict(optimizer_before)
        raise
