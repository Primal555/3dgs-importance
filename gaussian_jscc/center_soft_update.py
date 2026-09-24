"""Continuous Adam with historical update-size protection, no loss approval.

History stores UNSCALED proposal/parameter norm ratios divided by scheduled LR.
This prevents protection from progressively shrinking its own reference size,
and makes scheduled LR decay independent of the amplitude calibration.
"""
import copy
import math
import statistics

import torch


def scheduled_center_lr(base, step, steps, start=.5, end_ratio=.2):
    """One-based center step; hold base through start, then cosine to end_ratio."""
    if not 0 <= start < 1 or not 0 < end_ratio <= 1:
        raise ValueError('invalid center LR schedule')
    fraction = (step-1)/max(steps-1, 1)
    t = max(0., min(1., (fraction-start)/(1-start)))
    return base*(end_ratio+(1-end_ratio)*.5*(1+math.cos(math.pi*t)))


def soft_adam_step(optimizer, history, *, window=100, warmup=100, multiplier=3.):
    """One Adam step, optional per-group rescaling; never tests training loss.

    Moments/counters advance exactly once even for rescaled steps. Nonfinite
    gradients/proposals/state raise, restoring pre-step weights and optimizer;
    they are NOT silently accepted or counted as successful iterations.
    """
    if not 1 <= warmup <= window or not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('require 1 <= warmup <= window and finite multiplier >= 1')
    groups = []
    for i, g in enumerate(optimizer.param_groups):
        ps = [p for p in g['params'] if p.grad is not None]
        if not ps:
            continue
        if not math.isfinite(g['lr']) or g['lr'] <= 0:
            raise ValueError('soft update requires positive finite LR')
        if not all(bool(torch.isfinite(p.grad).all()) for p in ps):
            raise FloatingPointError('nonfinite gradients; optimizer not stepped')
        groups.append((g.get('name', str(i)), g['lr'], ps, [p.detach().clone() for p in ps]))
    if not groups:
        raise ValueError('soft update requires active gradients')
    if len({name for name, *_ in groups}) != len(groups):
        raise ValueError('optimizer group names must be unique')
    saved = copy.deepcopy(optimizer.state_dict())
    result, next_history = {}, copy.deepcopy(history)
    try:
        optimizer.step()
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                    raise FloatingPointError('nonfinite Adam state')
        with torch.no_grad():
            for name, lr, ps, originals in groups:
                deltas = [p-v for p, v in zip(ps, originals)]
                if not all(bool(torch.isfinite(d).all()) for d in deltas):
                    raise FloatingPointError('nonfinite Adam proposal')
                norm = lambda values: float(torch.stack([v.double().square().sum() for v in values]).sum().sqrt())
                parameter_norm = norm(originals)
                proposal_norm = norm(deltas)
                denominator = max(parameter_norm, 1e-12)
                relative = proposal_norm/denominator
                rows = history.get(name, [])[-window:]
                calibrated = len(rows) >= warmup
                reference = statistics.median(rows) if rows else None
                cap = multiplier*reference*lr if calibrated else None
                # A zero reference cannot freeze a group that starts learning late.
                scale = min(1., cap/relative) if cap is not None and cap > 0 and relative > 0 else 1.
                if scale < 1:
                    for p, v, d in zip(ps, originals, deltas):
                        p.copy_(v+scale*d)
                actual_norm = norm([p-v for p, v in zip(ps, originals)])
                next_history[name] = (rows+[relative/lr])[-window:]
                result[name] = {'scale': scale, 'limited': scale < 1, 'calibrated': calibrated,
                    'lr': lr, 'parameter_norm': parameter_norm, 'proposal_norm': proposal_norm,
                    'actual_update_norm': actual_norm, 'relative_proposal': relative,
                    'relative_actual_update': actual_norm/denominator, 'relative_cap': cap,
                    'zero_update': actual_norm == 0}
    except BaseException:
        with torch.no_grad():
            for _, _, ps, originals in groups:
                for p, v in zip(ps, originals):
                    p.copy_(v)
        optimizer.load_state_dict(saved)
        raise
    history.clear()
    history.update(next_history)
    return {'mode': 'continuous_adam_soft_limit', 'groups': result,
            'loss_acceptance_test': False, 'momentum_restarted': False}
