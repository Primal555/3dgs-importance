"""Same-training-batch backtracking of ONE Adam proposal, no extra loss.

Accepted partial steps keep the proposal's moments/step counter (one gradient
observation), equivalent to scaling that iteration's LR. Rejected steps restore
both parameters and optimizer state. This does not guarantee heldout improvement.
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
