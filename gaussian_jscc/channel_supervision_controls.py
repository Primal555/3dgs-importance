"""Two independent random-channel experiments; semantic teacher stays frozen.

color: 3 latent+joint-response updates, then 1 independent directional RGB update.
separated: latent, pure shape footprint, activated opacity, directional RGB.
Task frequencies are explicit engineering trade-offs, not optimal loss weights.
"""
import math
import torch
from torch.nn import functional as F

from . import channel_response_training as base
from .data import to_raw
from .local_response import (camera_directions, directional_color_loss, footprint,
                             local_response_loss, stencil, view_frames)
from .modular_training import latent_reconstruction_loss


# Measured at seed42 RANDOM initialization in the preceding scratch experiment.
# Reused explicitly for A; no new four-batch calibration, not an optimum claim.
BASE_RESPONSE_WEIGHT=0.10235305148738072


def parser():
    p=base.parser()
    p.description=__doc__
    p.set_defaults(channel_init='random',steps=8000,response_weight=None)
    p.add_argument('--variant',choices=('color','separated'),required=True)
    return p


def scheduled_task(variant,step):
    if type(step) is not int or step<1: raise ValueError('positive one-based update required')
    if variant=='color':
        task='color' if step%4==0 else 'response'
        ordinal=step//4 if task=='color' else step-step//4
    elif variant=='separated':
        task=('latent','shape','opacity','color')[(step-1)%4]
        ordinal=(step-1)//4+1
    else: raise ValueError('unknown supervision variant')
    return task,('1','2','3','mixed')[(ordinal-1)%4],ordinal


def color_directions(unit,geometry,centers,count,seed):
    if type(count) is not int or count<2 or count%2:
        raise ValueError('even direction count >=2 required')
    centers=torch.as_tensor(centers,device=unit.device,dtype=unit.dtype)
    if centers.ndim!=2 or centers.shape[-1]!=3 or not len(centers) or not torch.isfinite(centers).all():
        raise ValueError('finite nonempty training camera centers required')
    generator=torch.Generator().manual_seed(seed)
    ids=torch.randint(len(centers),(count//2,),generator=generator)
    camera=camera_directions(geometry.denormalize(unit.detach()),centers[ids.to(unit.device)])
    sphere=F.normalize(torch.randn(count//2,3,generator=generator),dim=-1).to(unit)
    return torch.cat((camera,sphere[None].expand(len(unit),-1,-1)),1),ids.tolist()


def shape_footprint_loss(pred,target,geometry,model,directions):
    """Peak-one projected Gaussian profiles, same origin and physical scale.

    Both detached source and prediction supports are probed. Neither profile
    contains SH, opacity or predicted XYZ. Reuses the existing relative 1e-5
    covariance floor for extremely thin ellipsoids; not a gradient clip.
    """
    if pred.ndim!=2 or pred.shape!=target.shape or not len(pred):
        raise ValueError('shape supervision requires matching nonempty feature rows')
    if (directions.ndim not in (2,3) or directions.shape[-1]!=3 or directions.shape[-2]<1
            or (directions.ndim==3 and directions.shape[0]!=len(pred))
            or not torch.isfinite(directions).all() or (directions.norm(dim=-1)<1e-8).any()):
        raise ValueError('finite nonzero shared or per-point directions required')
    raw=to_raw(pred,geometry,model)
    with torch.no_grad():
        source=to_raw(target.detach(),geometry,model)
        _,frames=view_frames(directions.detach().to(pred))
        source_chol,source_radius=footprint(source,frames)
    decoded_chol,decoded_radius=footprint(raw,frames)
    with torch.no_grad():
        points=stencil(pred)
        source_probes=torch.einsum('nvij,pj->nvpi',source_chol,points)
        decoded_probes=torch.einsum('nvij,pj->nvpi',decoded_chol.detach(),points)
        decoded_probes*=(decoded_radius.detach()-source_radius).exp()[:,None,None,None]
        probes=torch.cat((source_probes,decoded_probes),-2)

    def profile(chol,radius):
        offsets=probes*(source_radius-radius).exp()[:,None,None,None]
        whitened=torch.linalg.solve_triangular(chol,offsets.transpose(-1,-2),upper=False)
        return (-.5*whitened.square().sum(-2)).exp()

    with torch.no_grad(): reference=profile(source_chol,source_radius)
    loss=(profile(decoded_chol,decoded_radius)-reference).square().mean()
    if not torch.isfinite(loss): raise RuntimeError('nonfinite independent shape loss')
    return loss


def opacity_loss(pred,target,model):
    if pred.ndim!=2 or pred.shape!=target.shape or not len(pred):
        raise ValueError('opacity supervision requires matching nonempty feature rows')
    predicted=(pred[:,3]*model.attr_std[0]+model.attr_mean[0]).sigmoid()
    reference=(target.detach()[:,3]*model.attr_std[0]+model.attr_mean[0]).sigmoid()
    loss=(predicted-reference).square().mean()
    if not torch.isfinite(loss): raise RuntimeError('nonfinite independent opacity loss')
    return loss


def supervision_loss(model,teacher,y,recovered,q,xyz,geometry,views,color_views,task,weight):
    if task not in ('response','color','latent','shape','opacity'): raise ValueError('unknown task')
    enabled=torch.is_grad_enabled()
    latent,metrics=latent_reconstruction_loss(y,recovered,q,len(model.cfg.rates))
    with torch.set_grad_enabled(enabled and task!='latent'):
        pred,target=base.teacher_feature_pair(model,teacher,y,recovered,q,xyz)
    # Inactive objectives are diagnostics, never added to the backward graph.
    with torch.set_grad_enabled(enabled and task=='response'):
        response,parts=local_response_loss(pred,target,geometry,model,directions=views)
    with torch.set_grad_enabled(enabled and task=='color'):
        color,color_stats=directional_color_loss(pred,target,model,color_views)
    with torch.set_grad_enabled(enabled and task=='shape'):
        shape=shape_footprint_loss(pred,target,geometry,model,views)
    with torch.set_grad_enabled(enabled and task=='opacity'):
        opacity=opacity_loss(pred,target,model)
    loss={'response':latent+weight*response,'latent':latent,'color':color,'shape':shape,'opacity':opacity}[task]
    if not torch.isfinite(loss): raise RuntimeError('nonfinite supervised channel loss; no update')
    return loss,dict(latent_mse=float(latent.detach()),
        latent_nmse=float(latent.detach()/metrics['latent_target_power'].clamp_min(1e-12)),
        response_mse=float(response.detach()),response_weight=weight,
        weighted_latent_loss=float(latent.detach()) if task in ('response','latent') else 0.,
        weighted_response_mse=float(weight*response.detach()) if task=='response' else 0.,
        shape_mse=float(shape.detach()),opacity_mse=float(opacity.detach()),
        directional_color_mse=float(color.detach()),
        directional_color_bias={c:color_stats[f'directional_color_{c}_bias'] for c in 'rgb'},
        local_black_mse=parts['local_black_mse'],local_white_mse=parts['local_white_mse'],
        objective='independent_channel_'+task,task=task)


class Supervision:
    def __init__(self,args):
        self.args=args; self.variant=args.variant

    def add_to_plan(self,plan):
        if self.args.channel_init!='random': raise ValueError('these experiments require RANDOM channel initialization')
        if self.args.response_directions<2 or self.args.response_directions%2:
            raise ValueError('even direction count >=2 required')
        if self.variant=='separated' and self.args.response_weight!=0:
            raise ValueError('separated objectives use scheduling, not a response weight')
        plan['supervision']=dict(version=1,variant=self.variant,
            schedule='response,response,response,color' if self.variant=='color' else 'latent,shape,opacity,color',
            tier_schedule='independent q1/q2/q3/mixed cursor for each task',
            color_directions='half TRAIN cameras, half isotropic; RGB before clipping; independent of shape/opacity',
            shape='peak-one isolated orthographic footprint; shared physical frame; no color/opacity',
            opacity='activated opacity MSE; no shape/color',
            targets='frozen clean semantic decoder D(y), not PLY attributes or photos',
            weighting='explicit task frequencies; not optimal or gradient-balanced weights',
            response_weight_used=self.variant=='color')
        plan['objective']='scheduled '+plan['supervision']['schedule']
        plan['training_objective']=dict(version='channel_supervision_v1',variant=self.variant,
            channel_bootstrap='scheduled_independent_objectives',
            schedule=plan['supervision']['schedule'].split(','),
            response_directions=self.args.response_directions,
            targets=plan['supervision']['targets'])
        plan['calibration']=('explicit preceding random-initialization response weight; no recalibration'
            if self.variant=='color' else 'none: independent objectives, no weighted sum')

    def bind(self,scene,device):
        self.centers=torch.stack([torch.as_tensor(c.camera_center).detach().cpu() for c in scene.cameras]).to(device)

    def selection(self,step):
        return scheduled_task(self.variant,step)

    def loss(self,model,teacher,y,recovered,q,xyz,geometry,views,weight,step,selection):
        color_views,ids=color_directions(xyz[q>0],geometry,self.centers,self.args.response_directions,
                                       self.args.seed+900001*step)
        loss,stats=supervision_loss(model,teacher,y,recovered,q,xyz,geometry,views,color_views,selection[0],weight)
        stats.update(task_step=selection[2],color_training_camera_indices=ids,
                     color_camera_directions=len(ids),color_uniform_directions=len(ids),supervision_variant=self.variant)
        return loss,stats


def main(args,max_updates=None):
    if args.response_weight is None:
        args.response_weight=BASE_RESPONSE_WEIGHT if args.variant=='color' else 0.
    if not math.isfinite(args.response_weight): raise ValueError('finite response weight required')
    return base.main(args,max_updates=max_updates,supervision=Supervision(args))


if __name__=='__main__': main(parser().parse_args())
