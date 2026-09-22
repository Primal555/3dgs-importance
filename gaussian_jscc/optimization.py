"""Explicit clipping scope and inexpensive training-gradient observability."""
from contextlib import contextmanager
import random
import numpy as np
import torch


class ValidationLRSchedule:
    """Phase-local plateau schedule; patience counts bad validation checks.

    Defaults are explicit engineering choices, not loss-derived constants.
    No training-step loss or cross-phase metric enters this scheduler.
    """
    def __init__(self, optimizer, mode='plateau', factor=.5, patience=3,
                 threshold=.005, min_lr=1e-6):
        if mode not in ('plateau', 'constant'):
            raise ValueError('unknown learning-rate schedule')
        if not 0 < factor < 1 or patience < 1 or not 0 <= threshold < 1:
            raise ValueError('invalid LR factor/patience/relative threshold')
        if not np.isfinite(min_lr) or min_lr <= 0:
            raise ValueError('minimum LR must be positive and finite')
        if mode == 'plateau' and min_lr > min(g['lr'] for g in optimizer.param_groups):
            raise ValueError('minimum LR exceeds phase starting LR')
        self.optimizer = optimizer
        self.scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience-1,
            threshold=threshold, threshold_mode='rel', min_lr=min_lr, eps=0.)
            if mode == 'plateau' else None)

    def observe(self, score):
        score = float(score)
        if not np.isfinite(score):
            raise ValueError('nonfinite validation metric for LR schedule')
        before = [g['lr'] for g in self.optimizer.param_groups]
        if self.scheduler is not None:
            self.scheduler.step(score)
        after = [g['lr'] for g in self.optimizer.param_groups]
        return {'metric': score, 'lr_before': before, 'lr_after': after,
                'reduced': any(b > a for b, a in zip(before, after)),
                'bad_checks': self.scheduler.num_bad_epochs if self.scheduler else 0}


def parameter_group(name):
    if name.startswith('learned.'):
        part = name.split('.')[1]
        if part == 'dec_trunk':
            sub = name.split('.')[2]
            if sub == 'memory_reads':
                return 'decoder_memory_layer_' + name.split('.')[3]
            if sub in ('xyz_readout','xyz_norms'):
                return 'xyz_multidepth_head'
            if sub == 'heads':
                return 'covariance_head' if name.split('.')[3]=='logcov' else 'attribute_heads'
            if sub == 'blocks':
                return 'decoder_transformer_layer_'+name.split('.')[3]
            return 'decoder_trunk_input_output'
        if part == 'dec_xyz_center':
            return 'xyz_center_head'
        if part == 'dec_xyz_symbols':
            return 'xyz_symbol_head'
        if part == 'context_heads':
            head = name.split('.')[2]
            return ('xyz_context_head' if head == 'xyz' else
                    'covariance_context_head' if head == 'logcov' else 'attribute_context_heads')
        if part == 'heads':
            if name.split('.')[2] == 'logcov':
                return 'covariance_head'
            return 'xyz_head' if name.split('.')[2] == 'xyz' else 'attribute_heads'
        if part.startswith(('enc_geometry', 'enc_appearance', 'dec_geometry', 'dec_appearance')):
            prefix, stream = part.split('_')[:2]
            return stream + ('_encoder' if prefix == 'enc' else '_decoder')
        if part in ('enc_exchange', 'dec_exchange'):
            return 'encoder_exchange' if part == 'enc_exchange' else 'decoder_exchange'
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
