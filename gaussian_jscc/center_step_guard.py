"""Same-training-batch Adam guards, with no additional loss terms.

Legacy joint guard fully rolls back rejected proposals. Directional branch v2
adds first-moment restart and independent common/local acceptance; its total
rejections retain the momentum reset. Neither guarantees heldout improvement.
"""
import copy
import math

import torch


def guarded_step(optimizer, closure, loss_before, max_backtracks=4):
    if not isinstance(max_backtracks, int) or max_backtracks < 0:
        raise ValueError('max_backtracks must be a nonnegative integer')
    baseline = float(loss_before)
    if not math.isfinite(baseline):
        raise FloatingPointError('nonfinite baseline loss')
    parameters = [p for g in optimizer.param_groups for p in g['params']]
    original = [p.detach().clone() for p in parameters]
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    devices = sorted({p.device.index for p in parameters if p.device.type == 'cuda'})
    lrs = {g.get('name', str(i)): g['lr'] for i, g in enumerate(optimizer.param_groups)}

    def restore():
        with torch.no_grad():
            for p, value in zip(parameters, original):
                p.copy_(value)
        optimizer.load_state_dict(saved_optimizer)

    trials = []
    try:
        optimizer.step()
        proposal = [p.detach().clone() for p in parameters]
        for attempt in range(max_backtracks+1):
            scale = 2.**(-attempt)
            with torch.no_grad():
                if attempt:
                    for p, initial, candidate in zip(parameters, original, proposal):
                        p.copy_(initial+scale*(candidate-initial))
                finite = all(bool(torch.isfinite(p).all()) for p in parameters)
                # Current center networks contain no dropout or running-stat
                # layers. Preserve RNG so diagnostics never alter later batches.
                with torch.random.fork_rng(devices=devices):
                    value = float(closure()) if finite else float('inf')
            trials.append({'scale': scale, 'loss': value if math.isfinite(value) else None})
            if math.isfinite(value) and value <= baseline:
                return {'accepted': True, 'scale': scale, 'trials': trials,
                        'loss_before': baseline, 'loss_after': value,
                        'effective_lrs': {k: v*scale for k, v in lrs.items()}}
        restore()
        return {'accepted': False, 'scale': 0., 'trials': trials,
                'loss_before': baseline, 'loss_after': baseline,
                'effective_lrs': {k: 0. for k in lrs}}
    except BaseException:
        restore()
        raise


def directional_guarded_step(optimizer, closure, loss_before, max_backtracks=8):
    """V2: direction check + first-moment restart; second moments are retained.

    Only parameters with gradients participate. This permits independent
    acceptance/counters in a shared optimizer. On total rejection, weights,
    variance estimates and counters revert, but the active first moments are
    cleared so the next batch is not forced to reuse a known failed momentum.
    Exceptions instead restore the entire original state, including momentum.
    """
    if not isinstance(max_backtracks, int) or max_backtracks < 0:
        raise ValueError('max_backtracks must be a nonnegative integer')
    baseline = float(loss_before)
    if not math.isfinite(baseline):
        raise FloatingPointError('nonfinite baseline loss')
    parameters = [p for g in optimizer.param_groups for p in g['params'] if p.grad is not None]
    if not parameters:
        raise ValueError('directional guard requires active gradients')
    original = [p.detach().clone() for p in parameters]
    gradients = [p.grad.detach().clone() for p in parameters]
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    devices = sorted({p.device.index for p in parameters if p.device.type == 'cuda'})
    lrs = {g.get('name', str(i)): g['lr'] for i, g in enumerate(optimizer.param_groups)
           if any(p.grad is not None for p in g['params'])}
    attempts = []

    def restore():
        with torch.no_grad():
            for p, value in zip(parameters, original):
                p.copy_(value)
        # load_state_dict can alias tensors on the same device: each candidate
        # needs a fresh copy, otherwise its step mutates the rollback snapshot.
        optimizer.load_state_dict(copy.deepcopy(saved_optimizer))

    def clear_momentum():
        for p in parameters:
            if 'exp_avg' in optimizer.state.get(p, {}):
                optimizer.state[p]['exp_avg'].zero_()

    try:
        for restart in (False, True):
            if restart:
                restore()
                clear_momentum()
            optimizer.step()
            proposal = [p.detach().clone() for p in parameters]
            delta = [b-a for a, b in zip(original, proposal)]
            directional = float(torch.stack([(g.double()*d.double()).sum()
                                             for g, d in zip(gradients, delta)]).sum())
            record = {'momentum_restarted': restart,
                      'directional_derivative': directional if math.isfinite(directional) else None,
                      'descent_direction': math.isfinite(directional) and directional < 0,
                      'trials': []}
            attempts.append(record)
            if not record['descent_direction']:
                # Shrinking an uphill direction does not turn it downhill.
                continue
            for backtrack in range(max_backtracks+1):
                scale = 2.**(-backtrack)
                with torch.no_grad():
                    if backtrack:
                        for p, initial, d in zip(parameters, original, delta):
                            p.copy_(initial+scale*d)
                    finite = bool(torch.stack([torch.isfinite(p).all() for p in parameters]).all())
                    with torch.random.fork_rng(devices=devices):
                        value = float(closure()) if finite else float('inf')
                record['trials'].append({'scale': scale, 'loss': value if math.isfinite(value) else None})
                # Strict improvement: float-rounding equality is not counted
                # as a useful update and must not hide stalled training.
                if math.isfinite(value) and value < baseline:
                    return {'accepted': True, 'scale': scale, 'attempts': attempts,
                            'momentum_restarted': restart, 'momentum_cleared_on_reject': False,
                            'loss_before': baseline, 'loss_after': value,
                            'effective_lrs': {k: v*scale for k, v in lrs.items()}}
        restore()
        clear_momentum()
        return {'accepted': False, 'scale': 0., 'attempts': attempts,
                'momentum_restarted': True, 'momentum_cleared_on_reject': True,
                'loss_before': baseline, 'loss_after': baseline,
                'effective_lrs': {k: 0. for k in lrs}}
    except BaseException:
        restore()
        raise


def branch_guarded_step(model, optimizer, closure, loss_before, max_backtracks=8, clip_norm=0.):
    """Common then local, fresh gradient at each branch's actual starting point.

    The first backward was already computed by the trainer. Recompute the
    second branch after common acceptance/rejection using the SAME original
    position loss. No heldout data, auxiliary objective, or stale local gradient.
    An exception rolls back the entire outer iteration, not a half-applied step.
    """
    modules = model.learned.module_parameters()
    branches = {branch: [p for side in ('encoder', 'decoder')
                         for p in modules[f'center_{side}.{branch}']] for branch in ('common', 'local')}
    parameters = [p for g in optimizer.param_groups for p in g['params']]
    if len({id(p) for ps in branches.values() for p in ps}) != sum(map(len, branches.values())):
        raise ValueError('common/local branches must have disjoint parameters')
    if {id(p) for p in parameters} != {id(p) for ps in branches.values() for p in ps}:
        raise ValueError('branch guard is restricted to the independent center optimizer')
    original = [p.detach().clone() for p in parameters]
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    results = {}
    try:
        for branch, active in branches.items():
            if branch == 'common':
                active_ids = {id(p) for p in active}
                for p in parameters:
                    if id(p) not in active_ids:
                        p.grad = None
                baseline = float(loss_before)
            else:
                optimizer.zero_grad(set_to_none=True)
                objective = closure()
                if not bool(torch.isfinite(objective)):
                    raise FloatingPointError('nonfinite loss after common branch')
                grads = torch.autograd.grad(objective, active, allow_unused=True)
                baseline = float(objective.detach())
                for p, grad in zip(active, grads):
                    p.grad = grad
                del objective, grads
            grad_norm = float(torch.stack([p.grad.double().square().sum() for p in active
                                          if p.grad is not None]).sum().sqrt())
            if not math.isfinite(grad_norm):
                raise FloatingPointError('nonfinite branch gradient')
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(active, clip_norm, error_if_nonfinite=True)
            results[branch] = directional_guarded_step(optimizer, closure, baseline, max_backtracks)
            results[branch]['grad_norm'] = grad_norm
            results[branch]['clip_factor'] = min(1., clip_norm/(grad_norm+1e-6)) if clip_norm else 1.
        return {'mode': 'branch_directional_v2', 'branches': results,
                'accepted': any(r['accepted'] for r in results.values()),
                'scale': sum(r['scale'] for r in results.values())/2,
                'scale_definition': 'arithmetic mean of branch scales, NOT one global effective LR',
                'loss_before': float(loss_before), 'loss_after': results['local']['loss_after']}
    except BaseException:
        with torch.no_grad():
            for p, value in zip(parameters, original):
                p.copy_(value)
        optimizer.load_state_dict(optimizer_state)
        raise


def update_rejection_streaks(progress, result, warn_after, stop_after):
    """Persist counters in trainer progress so exact resume cannot reset alerts."""
    streaks = progress.setdefault('guard_rejection_streaks', {'common': 0, 'local': 0})
    warnings, stopped = [], []
    for branch, row in result['branches'].items():
        streaks[branch] = 0 if row['accepted'] else streaks[branch]+1
        if streaks[branch] and streaks[branch] % warn_after == 0:
            warnings.append(branch)
        if streaks[branch] >= stop_after:
            stopped.append(branch)
    return warnings, stopped
