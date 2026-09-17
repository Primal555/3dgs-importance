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
    if name.startswith('block_geometry.reference_encoder') or name == 'block_geometry.reference_power_logit':
        return 'reference_encoder'
    if name.startswith('block_geometry.reference_decoder'):
        return 'reference_decoder'
    if name.startswith('block_geometry.'):
        return 'geometry_encoder' if name.startswith('block_geometry.encoder.') else 'geometry_decoder'
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


def training_layouts(model, features, include_drop=True):
    """Individual mode: 75% mixed layouts, 25% uniform coverage, equal weights."""
    if model.cfg.individual_tiers:
        n=len(features);device=features.device
        uniform=int(torch.randint(1,4,(),device=device))
        layouts=[(str(uniform),torch.full((n,),uniform,device=device,dtype=torch.long))]
        for i in range(3):
            # Random skew, not just a uniform q histogram on every block.
            probabilities=torch.softmax(torch.randn(4,device=device)*1.5,0)
            if not include_drop: probabilities[0]=0
            q=torch.multinomial(probabilities,n,replacement=True)
            q[0]=1
            layouts.append((f'mixed{i+1}',q))
        return layouts
    layouts=[(str(t),torch.full((len(features),),t,device=features.device,dtype=torch.long)) for t in (1,2,3)]
    if model.cfg.position_head=='reference_v6':
        mixed=torch.randint(0 if include_drop else 1,4,(len(features),),device=features.device)
        mixed[0]=1
        layouts.append(('mixed',mixed))
    return layouts


def all_tier_attribute_step(model, features, geometry, snr, channel):
    """Sequential layout graphs, equal mean, then caller clips/steps once.

    Historical coverage is uniform tiers plus v6 mixed. Individual mode uses
    three mixed layouts and one sampled uniform tier. No monotonicity guarantee.
    """
    from .losses import reconstruction_loss, position_training_inputs
    total=features.new_zeros(())
    metrics={}; per_tier={}
    layouts=training_layouts(model,features,include_drop=model.cfg.individual_tiers)
    for tier,q in layouts:
        pred=model(features,features[:,:3],q,snr,channel)
        loss,terms=reconstruction_loss(pred,features,geometry,model,return_terms=True,
                                       active=(q>0).to(features) if model.cfg.individual_tiers else None,
                                       **position_training_inputs(model,features,q,snr))
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite attribute loss at tier {tier}')
        (loss/len(layouts)).backward()
        total=total+loss.detach()/len(layouts)
        for key,value in terms.items():
            metrics[key+'_loss']=metrics.get(key+'_loss',0.)+float(value.detach().mean())/len(layouts)
        with torch.no_grad():
            delta=((pred[:,:3].clamp(0,1)-features[:,:3])*geometry.span.to(features))[q>0]
            per_tier[str(tier)]={'loss':float(loss.detach()),'geometry_loss':float(terms['geometry'].detach().mean()),
                                 'position_rmse':float(delta.square().mean().sqrt()),
                                 'position_distance_p95':float(torch.quantile(delta.norm(dim=-1),.95)),
                                 'out_of_bounds_fraction':float(((pred[:,:3]<0)|(pred[:,:3]>1)).float().mean())}
    return total,{**metrics,'tier_training':'all','per_tier':per_tier}


def all_tier_geometry_step(model, features, geometry, snr, channel, clean_weight=0.):
    from .block_geometry import position_objective
    total = features.new_zeros(())
    metrics, per_tier = {}, {}
    xyz = features[:, :3]
    layouts=training_layouts(model,features)
    v6=model.cfg.position_head=='reference_v6'
    for label,q in layouts:
        pred = model.geometry_forward(xyz, q, snr, channel)
        if v6:
            from .reference_geometry import reference_position_rows
            scale=model.block_geometry.supervision_scale(xyz,q)
            loss=(reference_position_rows(pred,features,geometry,model,scale)*(q>0)).mean()
            terms={}
            clean_weight=model.cfg.geometry_clean_weight
        else:
            loss, terms = position_objective(pred, xyz)
        terms['noisy_position_loss'] = float(loss.detach())
        if clean_weight:
            clean = model.geometry_forward(xyz, q, snr, 'none')
            if v6:
                clean_loss=(reference_position_rows(clean,features,geometry,model,scale)*(q>0)).mean()
            else:
                clean_loss, _ = position_objective(clean, xyz)
            terms['clean_position_loss'] = float(clean_loss.detach())
            loss = loss + clean_weight * clean_loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite geometry objective')
        if v6: loss=loss*model.cfg.geometry_weight
        (loss / len(layouts)).backward()
        total += loss.detach() / len(layouts)
        for key, value in terms.items():
            metrics[key] = metrics.get(key, 0.) + value / len(layouts)
        with torch.no_grad():
            delta = (pred[q>0] - xyz[q>0]) * geometry.span.to(xyz)
            per_tier[label] = {'loss': float(loss.detach()),
                                   'unclipped_position_rmse': float(delta.square().mean().sqrt()),
                                   'position_distance_p95': float(torch.quantile(delta.norm(dim=-1), .95))}
    return total, {**metrics, 'geometry_loss': float(total), 'per_tier': per_tier, 'tier_training': 'all'}


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
