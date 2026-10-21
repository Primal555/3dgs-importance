"""Explicit latent reconstruction for the channel modules of modular JSCC."""
import math
import torch
from torch.nn import functional as F
from .data import to_raw
from .local_response import local_response_loss, directional_color_loss, camera_directions
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


def configure_phase(model, phase, module_lrs, *, freeze_semantic=False,denoiser_only=False,neighbor_only=False):
    if phase not in PHASE_MODULES:
        raise ValueError('unknown modular training phase')
    if neighbor_only:
        if denoiser_only or phase!='channel_bootstrap' or not freeze_semantic or not hasattr(model.channel_decoder,'neighbor_refiner'):
            raise ValueError('neighbor-only optimization requires frozen-semantic channel bootstrap and a neighbor refiner')
        model.requires_grad_(False)
        for p in model.parameters():p.grad=None
        model.channel_decoder.neighbor_refiner.requires_grad_(True)
        lr=float(module_lrs.get('channel_neighbor',1e-4))
        if not math.isfinite(lr) or lr<=0:raise ValueError('learning rates must be positive and finite')
        return torch.optim.Adam([dict(params=list(model.channel_decoder.neighbor_refiner.parameters()),lr=lr,name='channel_neighbor')])
    if denoiser_only:
        if phase!='channel_bootstrap' or not freeze_semantic or not hasattr(model.channel_decoder,'denoisers'):
            raise ValueError('denoiser-only optimization requires frozen-semantic channel bootstrap with denoisers')
        model.requires_grad_(False)
        for p in model.parameters(): p.grad=None
        model.channel_decoder.denoisers.requires_grad_(True)
        lr=float(module_lrs.get('channel_denoiser',1e-4))
        if not math.isfinite(lr) or lr<=0: raise ValueError('learning rates must be positive and finite')
        return torch.optim.Adam([{'params':list(model.channel_decoder.denoisers.parameters()),'lr':lr,'name':'channel_denoiser'}])
    groups=[]
    for name in MODULES:
        module=getattr(model,name)
        trainable=name in PHASE_MODULES[phase] and not (freeze_semantic and name.startswith('semantic_'))
        module.requires_grad_(trainable)
        for p in module.parameters(): p.grad=None
        if trainable:
            lr=float(module_lrs.get(name,1e-4))
            if not math.isfinite(lr) or lr<=0: raise ValueError('learning rates must be positive and finite')
            groups.append({'params':list(module.parameters()),'lr':lr,'name':name})
    return torch.optim.Adam(groups)


def predict(model, features, q, snr, kind, phase, paired_noise=False, *, tx_condition_snr=None,rx_condition_snr=None,noise_reference_symbols=None):
    if phase not in PHASE_MODULES: raise ValueError('unknown phase')
    if phase.startswith('semantic'):
        return model.forward_clean_batches(features,features[...,:3],q>0)
    return model.forward_tier_batches(features,features[...,:3],F.one_hot(q,len(model.cfg.rates)).to(features),snr,kind,paired_noise,
                                     tx_condition_snr=tx_condition_snr,rx_condition_snr=rx_condition_snr,
                                     noise_reference_symbols=noise_reference_symbols)[0]


def latent_reconstruction_loss(y, y_hat, q, tier_count=4):
    """Mean squared error per active point and latent dimension.

    The target is stop-gradient, not batch-normalized or rescaled. Statistics
    are sums divided by active count so full_scene_step can combine uneven
    padded batches without changing the objective or per-tier diagnostics.
    """
    if y.shape != y_hat.shape or y.shape[:-1] != q.shape or y.shape[-1] < 1:
        raise ValueError('latent target/prediction/tier shapes do not match')
    if not 2<=tier_count<=256 or ((q<0)|(q>=tier_count)).any():
        raise ValueError('latent loss tiers outside configured rate table')
    active=q>0
    target=y.detach()[active]
    received=y_hat[active]
    errors=(received-target).square().mean(-1)
    count=max(errors.numel(),1)
    loss=errors.sum()/count
    metrics={'latent_mse':loss.detach(),
             'latent_target_power':target.square().sum()/max(target.numel(),1)}
    for tier in range(1,tier_count):
        chosen=q[active]==tier
        metrics[f'latent_q{tier}_error']=errors.detach()[chosen].sum()/count
        metrics[f'latent_q{tier}_fraction']=chosen.sum().to(y.dtype)/count
    return loss,metrics


def latent_diagnostics(metrics):
    values={key:float(value) for key,value in metrics.items()}
    mse=values['latent_mse']; power=values['latent_target_power']
    tiers=sorted(int(k[len('latent_q'):-len('_fraction')]) for k in values if k.startswith('latent_q') and k.endswith('_fraction'))
    return {'latent_mse':mse,'latent_target_power':power,
            'latent_nmse':mse/max(power,1e-12),
            'latent_mse_by_tier':{str(t):values[f'latent_q{t}_error']/values[f'latent_q{t}_fraction']
                if values[f'latent_q{t}_fraction']>0 else None for t in tiers}}


def bootstrap_step(model, features, q, geometry, snr, kind, phase, directions=None):
    model.set_scene_geometry(geometry)
    if phase == 'channel_bootstrap':
        if not (q>0).any(): raise ValueError('channel bootstrap requires active Gaussians')
        y,y_hat,_=model.forward_channel_batches(features,features[...,:3],F.one_hot(q,len(model.cfg.rates)).to(features),snr,kind)
        loss,metrics=latent_reconstruction_loss(y,y_hat,q,len(model.cfg.rates))
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


def color_bootstrap_step(model, features, q, geometry, training_camera_centers, views=8):
    """Clean semantic update; half camera directions, half uniform sphere.

    Other attribute heads receive no Adam/momentum update. Shared semantic
    features do change; consequently non-color predictions are not guaranteed
    to stay fixed. Both channel modules remain frozen by configure_phase.
    """
    if type(views) is not int or views<2 or views%2:
        raise ValueError('color directions must be an even integer >=2')
    model.set_scene_geometry(geometry)
    active=q>0
    if not active.any(): raise ValueError('color bootstrap requires active Gaussians')
    centers=torch.as_tensor(training_camera_centers,device=features.device,dtype=features.dtype)
    if centers.ndim!=2 or centers.shape[-1]!=3 or not len(centers) or not torch.isfinite(centers).all():
        raise ValueError('finite nonempty training camera centers required')
    camera_ids=torch.randint(len(centers),(views//2,),device=features.device)
    selected=centers[camera_ids]
    points=geometry.denormalize(features[active][:,:3])
    camera=camera_directions(points,selected)
    sphere=F.normalize(torch.randn(views//2,3,device=features.device,dtype=features.dtype),dim=-1)
    directions=torch.cat((camera,sphere[None].expand(len(points),-1,-1)),1)
    pred=predict(model,features,q,10.,'none','semantic_bootstrap')
    loss,stats=directional_color_loss(pred[active],features[active],model,directions)
    loss.backward()
    for name in ('opacity','scale','rotation'):
        for p in model.semantic_decoder.heads[name].parameters(): p.grad=None
    stats.update(color_camera_directions=views//2,color_uniform_directions=views//2,
                 color_training_camera_indices=camera_ids.cpu().tolist())
    return loss.detach(),stats


def render_step(model, batches, geometry, snr, kind, task, phase, mode='replay',channel_latent_weight=1.,replay_observer=None):
    model.set_scene_geometry(geometry)
    objective_settings(channel_latent_weight)
    communication=phase=='channel_render'
    def batch_forward(f,q):
        if communication:
            y,y_hat,_=model.forward_channel_batches(f,f[...,:3],F.one_hot(q,len(model.cfg.rates)).to(f),snr,kind)
            auxiliary,metrics=latent_reconstruction_loss(y,y_hat,q,len(model.cfg.rates))
            pred=model.decode_representation(y_hat,q,delivered_positions(f[...,:3],q,model.cfg))
        else:
            pred=predict(model,f,q,snr,kind,phase)
            auxiliary,metrics=pred.sum()*0,{}
        raw=to_raw(pred[q>0],geometry,model)
        return raw,auxiliary,metrics
    loss,stats=full_scene_step(model,batches,geometry,snr,kind,task,
                               attr_weight=channel_latent_weight if communication else 0,mode=mode,
                               profile=True,batch_forward=batch_forward,replay_observer=replay_observer)
    image_kind=getattr(getattr(task,'objective',None),'kind','mse')
    stats.update(objective='source_ply_rgb_'+image_kind)
    if communication:
        metrics={key[:-5]:stats.pop(key) for key in list(stats) if key.startswith('latent_') and key.endswith('_loss')}
        stats.update(latent_diagnostics(metrics),objective='source_ply_rgb_'+image_kind+'+latent_mse',
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
