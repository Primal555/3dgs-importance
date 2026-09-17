"""Explicit clipping scope and inexpensive training-gradient observability."""
from contextlib import contextmanager
import random
import numpy as np
import torch


def parameter_group(name):
    if name.startswith('learned.'):
        part = name.split('.')[1]
        if part == 'heads':
            return 'xyz_head' if name.split('.')[2] == 'xyz' else 'attribute_heads'
        if part.startswith('dec'):
            return 'shared_decoder'
        if part in ('tier', 'snr'):
            return 'conditioning'
        return 'shared_encoder'
    return 'other'


def update_stats(model, before):
    groups = {}
    for name, p in model.named_parameters():
        if name not in before:
            continue
        groups.setdefault(parameter_group(name), []).append((before[name], p.detach()-before[name]))
    return {k: {'update_norm': float(torch.cat([d.reshape(-1) for _, d in v]).norm()),
                'relative_update': float(torch.cat([d.reshape(-1) for _, d in v]).norm()/
                                         torch.cat([p.reshape(-1) for p, _ in v]).norm().clamp_min(1e-12))}
            for k, v in groups.items()}


def clip_codec_gradients(model, max_norm=1., mode='global'):
    """Branch mode caps disjoint groups, not the global norm. No hidden scaling."""
    if mode not in ('global','branch','none') or not np.isfinite(max_norm) or max_norm <= 0:
        raise ValueError('Invalid gradient clipping settings')
    groups={}
    for name,p in model.named_parameters():
        if p.grad is not None:
            groups.setdefault(parameter_group(name),[]).append(p)
    norms={k:torch.stack([p.grad.detach().norm() for p in ps]).norm() for k,ps in groups.items()}
    total=torch.stack(list(norms.values())).norm() if norms else next(model.parameters()).new_zeros(())
    if not torch.isfinite(total):
        bad=[k for k,v in norms.items() if not torch.isfinite(v)]
        raise RuntimeError(f'Nonfinite codec gradients in {bad}; no optimizer step performed')
    if mode=='global':
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm,error_if_nonfinite=True)
    elif mode == 'branch':
        for ps in groups.values():
            torch.nn.utils.clip_grad_norm_(ps,max_norm,error_if_nonfinite=True)
    stats={'clip_mode':mode,'clip_norm':max_norm,'gradient_groups':{}}
    for k,ps in groups.items():
        before=float(norms[k]); after=float(torch.stack([p.grad.detach().norm() for p in ps]).norm())
        factor=1. if mode=='none' else min(1.,max_norm/(float(total if mode=='global' else norms[k])+1e-6))
        stats['gradient_groups'][k]={'before':before,'after':after,'clip_factor':factor}
    stats['post_clip_total_norm']=float(torch.stack([p.grad.detach().norm() for ps in groups.values() for p in ps]).norm()) if groups else 0.
    return total,stats


@contextmanager
def preserved_rng(device):
    """Fixed evaluations must not change later training samples/noise."""
    python_state, numpy_state=random.getstate(),np.random.get_state()
    devices=list(range(torch.cuda.device_count())) if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
