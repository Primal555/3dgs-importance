"""Common reconstruction on EVERY update; weak, measured color assistance."""
import math
import torch

from .channel_response_training import teacher_feature_pair
from .local_response import local_response_loss,directional_color_loss
from .modular_training import latent_reconstruction_loss


def common_terms(model,teacher,y,recovered,q,xyz,geometry,views,response_weight):
    latent,_=latent_reconstruction_loss(y,recovered,q,4)
    predicted,target=teacher_feature_pair(model,teacher,y,recovered,q,xyz)
    response,_=local_response_loss(predicted,target,geometry,model,directions=views)
    color,_=directional_color_loss(predicted,target,model,views)
    return latent+response_weight*response,color,dict(latent=latent,response=response,color=color)


def add_color(common,color,outputs,state,*,step,every,start,stat_start,relative):
    weight=0.;common_power=color_power=0.
    if step%every==0 and step>stat_start and common.requires_grad and color.requires_grad:
        a=torch.autograd.grad(common,outputs,retain_graph=True,allow_unused=True)
        b=torch.autograd.grad(color,outputs,retain_graph=True,allow_unused=True)
        common_power=sum(float(g.detach().double().square().sum()) for g in a if g is not None)
        color_power=sum(float(g.detach().double().square().sum()) for g in b if g is not None)
        if not all(math.isfinite(v) for v in (common_power,color_power)):
            raise RuntimeError('nonfinite objective gradients before optimizer update')
        if common_power>0 and color_power>0:
            first=not state.get('observations')
            for key,value in [('common_squared',common_power),('color_squared',color_power)]:
                state[key]=value if first else .99*state[key]+.01*value
            state['observations']=state.get('observations',0)+1
            if step>start:
                weight=min(1.,relative*math.sqrt(state['common_squared']/state['color_squared']),
                           relative*math.sqrt(common_power/color_power))
    loss=common+weight*color
    return loss,dict(common_loss=float(common.detach()),color_mse=float(color.detach()),color_weight=weight,
                     color_weighted_loss=weight*float(color.detach()),common_gradient_squared=common_power,
                     color_gradient_squared=color_power,color_ema_observations=state.get('observations',0))
