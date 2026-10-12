"""Phase ownership and unchanged response/render objectives for modular JSCC."""
import math
import torch
from torch.nn import functional as F
from .data import to_raw
from .local_response import local_response_loss
from .training import full_scene_step
from .optimization import clip_codec_gradients, update_stats

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


def predict(model, features, q, snr, kind, phase, paired_noise=False):
    if phase not in PHASE_MODULES: raise ValueError('unknown phase')
    if phase.startswith('semantic'):
        return model.forward_clean_batches(features,features[...,:3],q>0)
    return model.forward_tier_batches(features,features[...,:3],F.one_hot(q,4).to(features),snr,kind,paired_noise)[0]


def bootstrap_step(model, features, q, geometry, snr, kind, phase, directions=None):
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


def render_step(model, batches, geometry, snr, kind, task, phase, mode='replay'):
    def batch_forward(f,q):
        pred=predict(model,f,q,snr,kind,phase)
        raw=to_raw(pred[q>0],geometry,model)
        return raw,pred.sum()*0,{}
    loss,stats=full_scene_step(model,batches,geometry,snr,kind,task,attr_weight=0,mode=mode,
                               profile=True,batch_forward=batch_forward)
    stats.update(objective='source_ply_rgb_mse')
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
