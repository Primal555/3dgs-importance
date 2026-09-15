"""Explicit clipping scope and inexpensive training-gradient observability."""
from contextlib import contextmanager
import random
import numpy as np
import torch


def parameter_group(name):
    prefix = name.split('.')[0]
    if prefix in ('geo_dec', 'position_seed'):
        return 'geometry_decoder'
    if prefix == 'enc_geometry':
        return 'geometry_encoder'
    if prefix in ('dec_in', 'dec_blocks', 'heads', 'position_updates'):
        return 'attribute_decoder'
    if prefix == 'enc_out':
        return 'attribute_encoder'
    if prefix in ('enc_in', 'enc_blocks'):
        return 'shared_encoder'
    return 'conditioning'


def all_tier_attribute_step(model, features, geometry, snr, channel):
    """Same block/SNR at all positive rates; equal mean, three sequential graphs.

    This enforces coverage, not a guarantee that every tier improves. Backward
    happens here once per tier with weight 1/3, then caller clips/steps once.
    """
    from .losses import reconstruction_loss
    total=features.new_zeros(())
    metrics={}; per_tier={}
    for tier in (1,2,3):
        q=torch.full((len(features),),tier,device=features.device,dtype=torch.long)
        pred=model(features,features[:,:3],q,snr,channel)
        loss,terms=reconstruction_loss(pred,features,geometry,model,return_terms=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite attribute loss at tier {tier}')
        (loss/3).backward()
        total=total+loss.detach()/3
        for key,value in terms.items():
            metrics[key+'_loss']=metrics.get(key+'_loss',0.)+float(value.detach().mean())/3
        with torch.no_grad():
            delta=(pred[:,:3].clamp(0,1)-features[:,:3])*geometry.span.to(features)
            per_tier[str(tier)]={'loss':float(loss.detach()),'geometry_loss':float(terms['geometry'].detach().mean()),
                                 'position_rmse':float(delta.square().mean().sqrt()),
                                 'position_distance_p95':float(torch.quantile(delta.norm(dim=-1),.95)),
                                 'out_of_bounds_fraction':float(((pred[:,:3]<0)|(pred[:,:3]>1)).float().mean())}
    return total,{**metrics,'tier_training':'all','per_tier':per_tier}


def clip_codec_gradients(model, max_norm=1., mode='global'):
    """Branch mode caps disjoint groups, not the global norm. No hidden scaling."""
    if mode not in ('global','branch') or not np.isfinite(max_norm) or max_norm <= 0:
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
    else:
        for ps in groups.values():
            torch.nn.utils.clip_grad_norm_(ps,max_norm,error_if_nonfinite=True)
    stats={'clip_mode':mode,'clip_norm':max_norm,'gradient_groups':{}}
    for k,ps in groups.items():
        before=float(norms[k]); after=float(torch.stack([p.grad.detach().norm() for p in ps]).norm())
        factor=min(1.,max_norm/(float(total if mode=='global' else norms[k])+1e-6))
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
