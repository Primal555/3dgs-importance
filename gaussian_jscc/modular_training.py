"""Explicit latent reconstruction for the channel modules of modular JSCC."""
import math
import torch
from torch.nn import functional as F
from .data import to_raw
from .local_response import local_response_loss
from .training import full_scene_step
from .optimization import clip_codec_gradients, update_stats
from .position_delivery import delivered_positions

OBJECTIVE_VERSION = 'latent_reconstruction_v2'


def objective_settings(channel_latent_weight=1.):
    if not math.isfinite(channel_latent_weight) or channel_latent_weight <= 0:
        raise ValueError('channel latent weight must be positive and finite')
    return {'version':OBJECTIVE_VERSION,'channel_bootstrap':'latent_mse',
            'channel_render_latent_weight':float(channel_latent_weight)}

MODULES = ('semantic_encoder','channel_encoder','channel_decoder','semantic_decoder')
PHASE_MODULES = {
    'semantic_bootstrap': ('semantic_encoder','semantic_decoder'),
    'semantic_render': ('semantic_encoder','semantic_decoder'),
    'channel_bootstrap': ('channel_encoder','channel_decoder'),
    'channel_render': ('channel_encoder','channel_decoder','semantic_decoder'),
}


def configure_phase(model, phase, module_lrs):
    if phase not in PHASE_MODULES:
        raise ValueError('unknown modular training phase')
    groups=[]
    for name in MODULES:
        module=getattr(model,name)
        trainable=name in PHASE_MODULES[phase]
        module.requires_grad_(trainable)
        for p in module.parameters(): p.grad=None
        if trainable:
            lr=float(module_lrs.get(name,1e-4))
            if not math.isfinite(lr) or lr<=0: raise ValueError('learning rates must be positive and finite')
            groups.append({'params':list(module.parameters()),'lr':lr,'name':name})
    return torch.optim.Adam(groups)


def predict(model, features, q, snr, kind, phase, paired_noise=False, *, tx_condition_snr=None,rx_condition_snr=None):
    if phase not in PHASE_MODULES: raise ValueError('unknown phase')
    if phase.startswith('semantic'):
        return model.forward_clean_batches(features,features[...,:3],q>0)
    return model.forward_tier_batches(features,features[...,:3],F.one_hot(q,4).to(features),snr,kind,paired_noise,
                                     tx_condition_snr=tx_condition_snr,rx_condition_snr=rx_condition_snr)[0]


def latent_reconstruction_loss(y, y_hat, q):
    """Mean squared error per active point and latent dimension.

    The target is stop-gradient, not batch-normalized or rescaled. Statistics
    are sums divided by active count so full_scene_step can combine uneven
    padded batches without changing the objective or per-tier diagnostics.
    """
    if y.shape != y_hat.shape or y.shape[:-1] != q.shape or y.shape[-1] < 1:
        raise ValueError('latent target/prediction/tier shapes do not match')
    active=q>0
    target=y.detach()[active]
    received=y_hat[active]
    errors=(received-target).square().mean(-1)
    count=max(errors.numel(),1)
    loss=errors.sum()/count
    metrics={'latent_mse':loss.detach(),
             'latent_target_power':target.square().sum()/max(target.numel(),1)}
    for tier in (1,2,3):
        chosen=q[active]==tier
        metrics[f'latent_q{tier}_error']=errors.detach()[chosen].sum()/count
        metrics[f'latent_q{tier}_fraction']=chosen.sum().to(y.dtype)/count
    return loss,metrics


def latent_diagnostics(metrics):
    values={key:float(value) for key,value in metrics.items()}
    mse=values['latent_mse']; power=values['latent_target_power']
    return {'latent_mse':mse,'latent_target_power':power,
            'latent_nmse':mse/max(power,1e-12),
            'latent_mse_by_tier':{str(t):values[f'latent_q{t}_error']/values[f'latent_q{t}_fraction']
                if values[f'latent_q{t}_fraction']>0 else None for t in (1,2,3)}}


def bootstrap_step(model, features, q, geometry, snr, kind, phase, directions=None):
    model.set_scene_geometry(geometry)
    if phase == 'channel_bootstrap':
        if not (q>0).any(): raise ValueError('channel bootstrap requires active Gaussians')
        y,y_hat,_=model.forward_channel_batches(features,features[...,:3],F.one_hot(q,4).to(features),snr,kind)
        loss,metrics=latent_reconstruction_loss(y,y_hat,q)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite latent loss; no optimizer step performed')
        loss.backward()
        return loss.detach(),dict(latent_diagnostics(metrics),objective='latent_mse',
                                  weighted_latent_loss=float(loss.detach()))
    active=q>0
    latent={}
    hooks=[model.semantic_encoder.register_forward_hook(lambda m,a,o:latent.update(y=o.detach()))]
    if phase.startswith('channel'):
        hooks.append(model.channel_decoder.register_forward_hook(lambda m,a,o:latent.update(y_hat=o.detach())))
    try:
        pred=predict(model,features,q,snr,kind,phase)
    finally:
        for hook in hooks: hook.remove()
    loss,stats=local_response_loss(pred[active],features[active],geometry,model,directions=directions)
    if not torch.isfinite(loss): raise RuntimeError('nonfinite local response loss; no optimizer step performed')
    loss.backward()
    y=latent['y'][active]
    stats.update(objective='local_response_mse',latent_mean=float(y.mean()),latent_std=float(y.std(unbiased=False)),
                 latent_mse=float((latent.get('y_hat',latent['y'])[active]-y).square().mean()))
    return loss.detach(),stats


def render_step(model, batches, geometry, snr, kind, task, phase, mode='replay',channel_latent_weight=1.):
    model.set_scene_geometry(geometry)
    objective_settings(channel_latent_weight)
    communication=phase=='channel_render'
    def batch_forward(f,q):
        if communication:
            y,y_hat,_=model.forward_channel_batches(f,f[...,:3],F.one_hot(q,4).to(f),snr,kind)
            auxiliary,metrics=latent_reconstruction_loss(y,y_hat,q)
            pred=model.decode_representation(y_hat,q,delivered_positions(f[...,:3],q,model.cfg))
        else:
            pred=predict(model,f,q,snr,kind,phase)
            auxiliary,metrics=pred.sum()*0,{}
        raw=to_raw(pred[q>0],geometry,model)
        return raw,auxiliary,metrics
    loss,stats=full_scene_step(model,batches,geometry,snr,kind,task,
                               attr_weight=channel_latent_weight if communication else 0,mode=mode,
                               profile=True,batch_forward=batch_forward)
    stats.update(objective='source_ply_rgb_mse')
    if communication:
        metrics={key[:-5]:stats.pop(key) for key in list(stats) if key.startswith('latent_') and key.endswith('_loss')}
        stats.update(latent_diagnostics(metrics),objective='source_ply_rgb_mse+latent_mse',
                     channel_latent_weight=channel_latent_weight,
                     weighted_latent_loss=channel_latent_weight*stats['aux_loss'])
    return loss,stats


def finish_update(model, optimizer, before):
    norm,stats=clip_codec_gradients(model,mode='none')
    optimizer.step()
    changes=update_stats(model,before)
    for name in MODULES:
        changes.setdefault(name,{'update_norm':0.,'relative_update':0.})
        stats['gradient_groups'].setdefault(name,{'before':0.,'after':0.,'clip_factor':1.})
    stats.update(grad_norm=float(norm),updates=changes)
    for n,p in model.named_parameters():
        if not p.requires_grad and not torch.equal(p.detach(),before[n]):
            raise RuntimeError('frozen parameter changed: '+n)
    return stats
