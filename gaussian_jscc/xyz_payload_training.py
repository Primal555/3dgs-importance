"""Explicit coordinate/attribute recovery; no reliable-position objective."""
import math
import torch
from torch.nn import functional as F
from .data import to_raw
from .local_response import isolated_attribute_response_loss
from .modular_training import MODULES,PHASE_MODULES,color_bootstrap_step
from .modular_training import configure_phase as _configure
from .optimization import clip_codec_gradients,update_stats
from .training import full_scene_step


def objective_settings(xyz_weight=1.,channel_latent_weight=1.):
    if any(not math.isfinite(v) or v<=0 for v in (xyz_weight,channel_latent_weight)):
        raise ValueError('XYZ and channel latent weights must be positive finite')
    return dict(version='xyz_payload_group_mse_v1',xyz_latent_weight=float(xyz_weight),
                channel_latent_weight=float(channel_latent_weight),
                definition='mean XYZ component MSE + mean attribute latent MSE; valid points only')


def configure_phase(model,phase,module_lrs,*,joint_semantic_encoder=False):
    optimizer=_configure(model,phase,module_lrs)
    if phase=='channel_render' and joint_semantic_encoder:
        model.semantic_encoder.requires_grad_(True)
        optimizer.add_param_group(dict(params=list(model.semantic_encoder.parameters()),
                                       lr=float(module_lrs.get('semantic_encoder',1e-4)),name='semantic_encoder'))
    return optimizer


def payload_loss(y,y_hat,q,xyz_weight=1.):
    objective_settings(xyz_weight)
    if y.shape!=y_hat.shape or y.shape[:-1]!=q.shape or y.shape[-1]<=3:
        raise ValueError('matching [batch,points,3+attribute dimensions] payload required')
    active=q>0; target=y.detach()[active]; received=y_hat[active]
    if not torch.isfinite(target).all() or not torch.isfinite(received).all():
        raise ValueError('nonfinite active payload')
    count=max(len(target),1); error=(received-target).square()
    xyz=error[:,:3].sum()/(3*count); attr=error[:,3:].sum()/((y.shape[-1]-3)*count)
    flat=error.sum()/(y.shape[-1]*count)
    power=target.square().sum()/max(target.numel(),1)
    stats={'payload_xyz_mse':xyz.detach(),'payload_attr_mse':attr.detach(),
           'payload_xyz_power':target[:,:3].square().sum()/(3*count),
           'payload_attr_power':target[:,3:].square().sum()/((y.shape[-1]-3)*count),
           'latent_mse':flat.detach(),'latent_target_power':power.detach()}
    for tier in range(1,4):
        selected=q[active]==tier; n=selected.sum().to(y)
        stats[f'q{tier}_fraction']=n/count
        stats[f'q{tier}_xyz_error']=error[selected,:3].sum()/(3*count)
        stats[f'q{tier}_attr_error']=error[selected,3:].sum()/((y.shape[-1]-3)*count)
    return xyz_weight*xyz+attr,stats


def diagnostics(stats):
    row={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in stats.items()}
    row['latent_nmse']=row['latent_mse']/max(row['latent_target_power'],1e-12)
    row['payload_xyz_nmse']=row['payload_xyz_mse']/max(row['payload_xyz_power'],1e-12)
    row['payload_attr_nmse']=row['payload_attr_mse']/max(row['payload_attr_power'],1e-12)
    for tier in range(1,4):
        fraction=row[f'q{tier}_fraction']
        for branch in ('xyz','attr'):
            row[f'q{tier}_{branch}_mse']=row[f'q{tier}_{branch}_error']/fraction if fraction else None
    return row


def bootstrap_step(model,features,q,geometry,snr,kind,phase,xyz_weight=1.):
    model.set_scene_geometry(geometry)
    if not (q>0).any(): raise ValueError('bootstrap requires active points')
    if phase=='channel_bootstrap':
        y,yh,_=model.forward_channel_batches(features,features[...,:3],F.one_hot(q,4).to(features),snr,kind)
        loss,stats=payload_loss(y,yh,q,xyz_weight)
        record=dict(diagnostics(stats),objective='group_balanced_payload_mse',payload_loss=float(loss.detach()))
    elif phase=='semantic_bootstrap':
        pred=model.forward_clean_batches(features,features[...,:3],q>0)
        loss,record=isolated_attribute_response_loss(pred[q>0],features[q>0],geometry,model)
        record.update(objective='local_response_mse')
    else: raise ValueError('not a bootstrap phase')
    if not torch.isfinite(loss): raise RuntimeError('nonfinite bootstrap loss; no optimizer update')
    loss.backward()
    return loss.detach(),record


def render_step(model,batches,geometry,snr,kind,task,phase,mode='replay',xyz_weight=1.,channel_latent_weight=1.):
    objective_settings(xyz_weight,channel_latent_weight)
    communication=phase=='channel_render'
    if phase not in ('semantic_render','channel_render'): raise ValueError('not a render phase')
    model.set_scene_geometry(geometry)
    def forward(f,q):
        if communication:
            y,yh,_=model.forward_channel_batches(f,f[...,:3],F.one_hot(q,4).to(f),snr,kind)
            aux,metrics=payload_loss(y,yh,q,xyz_weight)
            pred=model.decode_representation(yh,q)
        else:
            pred=model.forward_clean_batches(f,f[...,:3],q>0)
            aux=pred.sum()*0; metrics={}
        return to_raw(pred[q>0],geometry,model),aux,metrics
    loss,stats=full_scene_step(model,batches,geometry,snr,kind,task,
                              attr_weight=channel_latent_weight if communication else 0.,
                              mode=mode,profile=True,batch_forward=forward)
    stats.update(objective='source_ply_rgb_mse')
    if communication:
        keys=('payload_xyz_mse','payload_attr_mse','payload_xyz_power','payload_attr_power',
              'latent_mse','latent_target_power',*[f'q{t}_{k}' for t in range(1,4) for k in ('fraction','xyz_error','attr_error')])
        values={k:stats.pop(k+'_loss') for k in keys}
        stats.update(diagnostics(values),objective='source_ply_rgb_mse+group_balanced_payload_mse',
                     payload_loss=stats['aux_loss'],weighted_latent_loss=channel_latent_weight*stats['aux_loss'])
    return loss,stats


def finish_update(model,optimizer,before):
    norm,stats=clip_codec_gradients(model,mode='none')
    if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient; optimizer not updated')
    optimizer.step()
    for n,p in model.named_parameters():
        if not torch.isfinite(p).all(): raise RuntimeError('nonfinite parameter after optimizer update: '+n)
        if not p.requires_grad and not torch.equal(p.detach(),before[n]):
            raise RuntimeError('frozen parameter changed: '+n)
    changes=update_stats(model,before)
    for name in MODULES:
        changes.setdefault(name,{'update_norm':0.,'relative_update':0.})
        stats['gradient_groups'].setdefault(name,{'before':0.,'after':0.,'clip_factor':1.})
    stats.update(grad_norm=float(norm),updates=changes)
    return stats
