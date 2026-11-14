"""Source-aligned multi-SNR continuation from a fixed 10 dB shared codec."""
import math

import torch
from torch.nn import functional as F

from .channel_supervision_controls import color_directions
from .local_response import local_response_loss, directional_color_loss
from .modular_training import latent_reconstruction_loss
from .position_delivery import delivered_positions
from .successive_objective import add_color


def parser():
    from .successive_multisnr import parser as base_parser
    p = base_parser()
    p.description = __doc__
    p.set_defaults(source_aligned=True, channel_lr=1e-5)
    p.add_argument('--render-every', type=int, default=4)
    p.add_argument('--anchor-every', type=int, default=4)
    p.add_argument('--anchor-weight', type=float, default=.1)
    p.add_argument('--side-link-policy', choices=('fixed_16qam_v1','snr_qpsk_half_v1'), default='snr_qpsk_half_v1')
    p.add_argument('--monitor-scenes', nargs='+', default=['truck','playroom'])
    p.add_argument('--monitor-every', type=int, default=250, help='updates per scene')
    p.add_argument('--monitor-early', nargs='*', type=int, default=[25,100])
    p.add_argument('--monitor-trials', type=int, default=2)
    return p


def validate(args, specs):
    if min(args.render_every,args.anchor_every,args.monitor_every,args.monitor_trials) < 1:
        raise ValueError('positive render/anchor/monitor intervals required')
    if not math.isfinite(args.anchor_weight) or args.anchor_weight < 0:
        raise ValueError('finite nonnegative anchor weight required')
    if not args.monitor_scenes or not set(args.monitor_scenes) <= {s['name'] for s in specs}:
        raise ValueError('monitor scenes must be training scenes with held-out validation cameras')
    if len(set(args.monitor_scenes)) != len(args.monitor_scenes) or any(x<1 for x in args.monitor_early):
        raise ValueError('unique monitor scenes and positive early visits required')


def paired_decode(model, features, y, q, snr, noise, *, with_main=False):
    if with_main:
        from .codec import prefix_mask
        from .successive_channel import encode_scales,decode_scales
        condition=model.resolve_condition_snr(snr)
        signal,scales=model.channel_encoder(y.detach(),q,condition,model.channel_decoder)
        mask=prefix_mask(q.flatten(),model.cfg.rates).reshape_as(signal)
        if not model.training: scales=decode_scales(encode_scales(scales,q),q,dtype=signal.dtype)
        side=delivered_positions(features[...,:3],q,model.cfg)
        stages,final=model.channel_decoder((signal+noise)*mask,mask,q,condition,
            scales=scales,delivered_xyz=side,return_stages=True)
        return [(value,model.decode_representation(value,q,side)) for value in (stages[...,-1,:],final)]
    _, recovered, _ = model.forward_channel_batches(features,features[...,:3],
        F.one_hot(q,4).to(features),snr,'awgn',paired_noise=True,
        _semantic_y=y.detach(),_channel_noise=noise)
    pred = model.decode_representation(recovered,q,delivered_positions(features[...,:3],q,model.cfg))
    return recovered, pred


def source_batch_loss(model, features, cached, geometry, centers, args, progress, snr, layout):
    """All three delivered prefixes share one source block and AWGN realization.

    Each tier executes its own encoder/decoder, including conditional block scales
    and neighborhood selection. Targets are original PLY attributes in normalized
    coordinates. A q3 intermediate state is not substituted for a delivered q1.
    """
    device = next(model.parameters()).device
    f = features.to(device); y = cached['y'].to(device).detach()
    active = cached['active'].to(device)
    step = progress['step']+1
    generator = torch.Generator(device=device).manual_seed(args.seed+200003*step)
    noise = torch.randn((*active.shape,2*model.cfg.rates[-1]),generator=generator,device=device,dtype=f.dtype)
    noise *= math.sqrt(10**(-snr/10)/2)
    views,_ = color_directions(f[...,:3][active],geometry,centers,args.response_directions,args.seed+900001*step)
    from .multisnr_training import tiers
    maps = [active.long()*q for q in (1,2,3)]
    if layout == 'mixed': maps.append(tiers(active,'mixed',args.seed+300007*step))
    common, colors, recovered_values, diagnostics = [], [], [], []
    for q in maps:
        endpoint_stats=[]
        for recovered,pred in paired_decode(model,f,y,q,snr,noise,with_main=True):
            latent,_ = latent_reconstruction_loss(y,recovered,q)
            response,_ = local_response_loss(pred[active],f[active],geometry,model,directions=views)
            color,_ = directional_color_loss(pred[active],f[active],model,views)
            common.append(args.latent_weight*latent+args.response_weight*response)
            colors.append(color); recovered_values.append(recovered)
            endpoint_stats.append(dict(latent_mse=float(latent.detach()),source_response_mse=float(response.detach()),
                                       source_color_mse=float(color.detach())))
        diagnostics.append(dict(endpoint_stats[-1],main=endpoint_stats[0]))
    loss,stats = add_color(torch.stack(common).mean(),torch.stack(colors).mean(),recovered_values,
        progress['color_state'],step=step,every=args.color_every,start=args.color_start,
        stat_start=args.color_stat_start,relative=args.color_relative_gradient)
    stats.update(objective='source_ply_response+latent_mse+source_color',prefixes=diagnostics[:3],
                 latent_mse=sum(v['latent_mse'] for v in diagnostics)/len(diagnostics),
                 supervised_layouts=['1','2','3']+(['mixed'] if layout=='mixed' else []))
    # Source-aligned noiseless constraint, with its own actual tier execution.
    if step%args.noiseless_every == 0:
        q=maps[(step//args.noiseless_every-1)%3]
        recovered,pred=paired_decode(model,f,y,q,snr,torch.zeros_like(noise))
        latent,_=latent_reconstruction_loss(y,recovered,q)
        response,_=local_response_loss(pred[active],f[active],geometry,model,directions=views)
        clean=args.latent_weight*latent+args.response_weight*response
        loss=loss+args.noiseless_weight*clean
        stats['noiseless_source_objective']=float(clean.detach())
    stats['source_batch_loss']=float(loss.detach())
    return loss,stats


def anchor_loss(model, teacher, features, cached, q, geometry, seed):
    """Preserve the pretrained 10 dB noisy mapping, including its semantic decoder."""
    device=next(model.parameters()).device
    f=features.to(device); y=cached['y'].to(device); q=q.to(device); active=q>0
    teacher.set_scene_geometry(geometry)
    noise=torch.randn((*q.shape,2*model.cfg.rates[-1]),device=device,dtype=f.dtype,
                      generator=torch.Generator(device=device).manual_seed(seed))*math.sqrt(.05)
    with torch.no_grad(): target_y,target=paired_decode(teacher,f,y,q,10.,noise)
    recovered,pred=paired_decode(model,f,y,q,10.,noise)
    latent=(recovered[active]-target_y[active]).square().mean()
    attributes=(pred[active][...,3:]-target[active][...,3:]).square().mean()
    return latent+attributes,dict(anchor_latent_mse=float(latent.detach()),
                                  anchor_attribute_mse=float(attributes.detach()),anchor_snr_db=10.)


if __name__ == '__main__':
    from .successive_multisnr import main
    main(parser().parse_args())
