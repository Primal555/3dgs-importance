"""Three channel-only optimizer scopes and one persistent common objective."""
import random
import torch

from .codec import channel,prefix_mask
from .channel_supervision_controls import color_directions
from .successive_objective import common_terms,add_color

PHASES=('main','neighbor','cooperative')


def configure_optimizer(model,phase,rates):
    if phase not in PHASES:raise ValueError('unknown successive training phase')
    model.requires_grad_(False)
    for parameter in model.parameters():parameter.grad=None
    model.channel_decoder.neighbor_enabled.fill_(phase!='main')
    groups=[]
    if phase!='neighbor':
        lr=rates['cooperative'] if phase=='cooperative' else rates['main']
        for name,module in [('channel_encoder',model.channel_encoder),('channel_main',model.channel_decoder.stages)]:
            module.requires_grad_(True);groups.append(dict(params=list(module.parameters()),lr=lr,name=name))
    if phase!='main':
        model.channel_decoder.neighbor_refiner.requires_grad_(True)
        groups.append(dict(params=list(model.channel_decoder.neighbor_refiner.parameters()),lr=rates['neighbor'],name='channel_neighbor'))
    return torch.optim.Adam(groups)


def optimizer_group_updates(model,optimizer,before):
    names={id(parameter):name for name,parameter in model.named_parameters()}
    result={}
    for group in optimizer.param_groups:
        parameters=group['params']
        delta=torch.stack([(p.detach()-before[names[id(p)]]).square().sum() for p in parameters]).sum().sqrt()
        magnitude=torch.stack([before[names[id(p)]].square().sum() for p in parameters]).sum().sqrt()
        gradients=[p.grad.detach().square().sum() for p in parameters if p.grad is not None]
        result[group['name']]=dict(lr=group['lr'],update_norm=float(delta),
            relative_update=float(delta/magnitude.clamp_min(1e-12)),
            gradient_norm=float(torch.stack(gradients).sum().sqrt()) if gradients else 0.)
    return result


def inputs(cache,phase,phase_step,step,seed,device):
    index=random.Random(seed+100003*step).randrange(len(cache))
    batch=cache[index];y,active,xyz=[batch[k].to(device) for k in ('y','active','xyz')]
    layout='3' if phase=='main' else ('1','2','3','mixed')[(phase_step-1)%4]
    q=(torch.randint(1,4,active.shape,generator=torch.Generator().manual_seed(seed+300007*step)).to(device)
       if layout=='mixed' else torch.full(active.shape,int(layout),dtype=torch.long,device=device))
    return y,q.masked_fill(~active,0),xyz,index,('all_prefixes' if phase=='main' else layout)


def loss_for_batch(model,teacher,y,q,xyz,geometry,centers,args,progress):
    phase=progress['successive_phase'];step=progress['step']+1
    signal,scales=model.channel_encoder(y.detach(),q,10.,model.channel_decoder)
    mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(signal)
    generator=torch.Generator(device=y.device).manual_seed(args.seed+200003*step)
    received=channel(signal.reshape(-1,2),10.,'awgn',generator).reshape_as(signal)*mask
    stages,final=model.channel_decoder(received,mask,q,10.,scales=scales,delivered_xyz=xyz,return_stages=True)
    if phase=='main':outputs=[stages[...,i,:] for i in range(3)]
    elif phase=='neighbor':outputs=[final]
    else:outputs=[stages[...,-1,:],final]
    views,_=color_directions(xyz[q>0],geometry,centers,args.response_directions,args.seed+900001*step)
    terms=[common_terms(model,teacher,y,value,q,xyz,geometry,views,args.response_weight) for value in outputs]
    common=torch.stack([t[0] for t in terms]).mean();color=torch.stack([t[1] for t in terms]).mean()
    loss,stats=add_color(common,color,outputs,progress['color_state'],step=step,every=args.color_every,
        start=args.color_start,stat_start=args.color_stat_start,relative=args.color_relative_gradient)
    stats.update(latent_mse=float(torch.stack([t[2]['latent'] for t in terms]).mean().detach()),
                 response_mse=float(torch.stack([t[2]['response'] for t in terms]).mean().detach()),
                 main_latent_mse=float((stages[...,-1,:][q>0]-y[q>0]).square().mean().detach()),
                 final_latent_mse=float((final[q>0]-y[q>0]).square().mean().detach()),
                 prefix_latent_mse=[float((stages[...,i,:][q>0]-y[q>0]).square().mean().detach()) for i in range(3)])
    noiseless=loss.new_zeros(())
    if phase!='neighbor' and step%args.noiseless_every==0:
        clean_stages,_=model.channel_decoder(signal,mask,q,10.,scales=scales,delivered_xyz=xyz,return_stages=True)
        clean_outputs=[clean_stages[...,i,:] for i in range(3)] if phase=='main' else [clean_stages[...,-1,:]]
        noiseless=torch.stack([(value[q>0]-y.detach()[q>0]).square().mean() for value in clean_outputs]).mean()
        loss=loss+args.noiseless_weight*noiseless
    from .successive_channel import present_scales
    present=present_scales(q)
    stats.update(noiseless_main_mse=float(noiseless.detach()),noiseless_weight=args.noiseless_weight,
                 response_weight=args.response_weight,scale_content_bits=16*int(present.sum()),
                 scale_stream_bytes=8+2*int(present.sum()),
                 mean_complex_power=float(signal.detach().square().sum()/mask.sum().clamp_min(1)*2),
                 main_outputs_supervised=len(outputs),objective='successive_common_plus_auxiliary_color')
    if not torch.isfinite(loss):raise RuntimeError('nonfinite successive objective; no update')
    return loss,stats
