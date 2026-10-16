"""Isolated one-pair JSCC updates and exact training checkpoints."""
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from .codec import GaussianCodec, CodecConfig
from .local_response import local_response_loss, directional_color_loss, camera_directions
from .modular_checkpoint import capture_rng
from .optimization import clip_codec_gradients

LAYOUTS=('1','2','3','mixed')


def update_spec(phase, phase_step, color_every=4):
    if phase not in ('bootstrap','render') or phase_step<1 or color_every<0:
        raise ValueError('invalid phase, update index or color interval')
    if phase=='render': return 'render',LAYOUTS[(phase_step-1)%4]
    colors=phase_step//color_every if color_every else 0
    if color_every and phase_step%color_every==0:
        return 'direction_color',LAYOUTS[(colors-1+2)%4]
    return 'local_response',LAYOUTS[(phase_step-colors-1)%4]


def make_optimizer(model, lr):
    if not math.isfinite(lr) or lr<=0: raise ValueError('positive finite learning rate required')
    model.requires_grad_(True)
    model.learned.heads['xyz'].requires_grad_(False)
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)


def bootstrap_step(model, features, q, geometry, centers, *, color=False, color_directions=8):
    active=q>0
    if not active.any(): raise ValueError('bootstrap requires active Gaussians')
    pred=model.forward_tier_batches(features,features[...,:3],F.one_hot(q,4).to(features),10.,'awgn')[0]
    if color:
        if color_directions<2 or color_directions%2: raise ValueError('color directions must be even and >=2')
        centers=torch.as_tensor(centers,device=features.device,dtype=features.dtype)
        if centers.ndim!=2 or centers.shape[-1]!=3 or not len(centers) or not torch.isfinite(centers).all():
            raise ValueError('finite nonempty training camera centers required')
        indices=torch.randint(len(centers),(color_directions//2,),device=features.device)
        camera=camera_directions(geometry.denormalize(features[active][:,:3]),centers[indices])
        sphere=F.normalize(torch.randn(color_directions//2,3,device=features.device,dtype=features.dtype),dim=-1)
        directions=torch.cat((camera,sphere[None].expand(len(camera),-1,-1)),1)
        loss,stats=directional_color_loss(pred[active],features[active],model,directions)
        stats['color_training_camera_indices']=indices.cpu().tolist()
    else:
        loss,stats=local_response_loss(pred[active],features[active],geometry,model)
        stats['objective']='local_response_mse'
    if not torch.isfinite(loss): raise RuntimeError('nonfinite bootstrap loss; no update performed')
    loss.backward()
    if color:
        # Clear gradients even when a previous stage's non-color outputs fed a
        # color refinement. None also prevents old Adam moments moving heads.
        for heads in [model.learned.heads]+[r.heads for r in model.learned.refinements]:
            for name,head in heads.items():
                if name not in ('dc','sh'):
                    for p in head.parameters(): p.grad=None
    return loss.detach(),stats


def parameter_group(name):
    if name.startswith('learned.refinements.'):
        return 'enhancement_'+str(int(name.split('.')[2])+2)
    part=name.split('.')[1] if name.startswith('learned.') else ''
    if part=='heads':return 'xyz_head' if name.split('.')[2]=='xyz' else 'base_attribute_heads'
    if part.startswith('dec'):return 'decoder'
    if part in ('tier','snr'):return 'conditioning'
    return 'encoder'


def finish_update(model, optimizer, before):
    norm,_=clip_codec_gradients(model,mode='none')
    groups={}
    for n,p in model.named_parameters():groups.setdefault(parameter_group(n),[]).append((n,p))
    gradients={g:float(torch.stack([p.grad.norm() for _,p in ps if p.grad is not None]).norm())
               if any(p.grad is not None for _,p in ps) else 0. for g,ps in groups.items()}
    optimizer.step()
    updates={g:{'update_norm':float(torch.cat([(p.detach()-before[n]).flatten() for n,p in ps]).norm())}
             for g,ps in groups.items()}
    for n,p in model.named_parameters():
        if not p.requires_grad and not torch.equal(p,before[n]):raise RuntimeError('frozen parameter changed: '+n)
    return dict(grad_norm=float(norm),gradient_norms=gradients,updates=updates,clip_mode='none')


def save_state(path, model, optimizer, progress, settings, scene_record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    saved=dict(version=4,config=model.cfg.to_dict(),state_dict=model.state_dict(),step=progress['step'],
        training=dict(integrated_version=1,progress=progress,settings=settings,scene_record=scene_record,
                      optimizer=optimizer.state_dict(),rng=capture_rng()))
    temporary=path.with_name(path.name+'.tmp')
    try:
        torch.save(saved,temporary);temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()


def restore_state(path, device):
    saved=torch.load(path,map_location='cpu',weights_only=True);state=saved.get('training',{})
    if saved.get('version')!=4 or state.get('integrated_version')!=1:
        raise ValueError('expected integrated experiment training checkpoint, not weights-only/modular checkpoint')
    model=GaussianCodec(CodecConfig.from_dict(saved['config'])).to(device)
    model.load_state_dict(saved['state_dict'])
    optimizer=make_optimizer(model,state['settings']['lr']);optimizer.load_state_dict(state['optimizer'])
    return model,optimizer,state
