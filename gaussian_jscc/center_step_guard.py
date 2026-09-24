"""Joint pointwise-XYZ Adam backtracking; no architecture or loss changes.

Trials use only the current training batch. A rejected/uphill momentum proposal
may restart the first moment, retaining Adam variance and step history. This is
not a guarantee of validation or rendering improvement.
"""
import copy
import math

import torch


def directional_guarded_step(optimizer, closure, loss_before, max_backtracks=8):
    """Direction check + first-moment restart; second moments are retained.

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



def update_rejection_streak(progress, result, warn_after, stop_after):
    count = 0 if result['accepted'] else progress.get('guard_rejection_streak', 0)+1
    progress['guard_rejection_streak'] = count
    return bool(count and count % warn_after == 0), count >= stop_after


def guard_summary(rows):
    entries = [r['stats']['step_guard'] for r in rows if r.get('stats', {}).get('step_guard')]
    if not entries:
        return None
    return {'mode': 'pointwise_joint_directional', 'attempted_updates': len(entries),
            'accepted_updates': sum(r['accepted'] for r in entries),
            'skipped_updates': sum(not r['accepted'] for r in entries),
            'reduced_updates': sum(0 < r['scale'] < 1 for r in entries),
            'momentum_restarts': sum(r['momentum_restarted'] for r in entries),
            'mean_scale_including_skips': sum(r['scale'] for r in entries)/len(entries),
            'scope': 'same-training-batch strict decrease only; not validation/render improvement'}
