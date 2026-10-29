"""Depth-layer transmittance bounds and signed whole-block candidate response."""
import torch
from torch.nn import functional as F

from .response_value_features import project_footprints

LAYERS=12


@torch.no_grad()
def layer_context(scene,camera,degree,render):
    raw=scene[:,:-1]
    xyz=torch.cat((raw[:,:3],torch.ones_like(raw[:,:1])),-1)
    z=(xyz@camera.world_view_transform.to(raw))[:,2]
    active=(scene[:,-1]>0)&(z>.2)
    depths=z[active]
    edges=torch.quantile(depths,torch.arange(1,LAYERS,device=z.device,dtype=z.dtype)/LAYERS) if len(depths) else z.new_zeros(LAYERS-1)
    bins=torch.bucketize(z.contiguous(),edges)
    maps=[]
    for start in range(0,LAYERS,3):
        colors=(bins[:,None]==torch.arange(start,start+3,device=z.device)[None]).to(raw)
        maps.append(render(raw,camera,degree,False,scene[:,-1],colors=colors))
    weights=torch.cat(maps).clamp_min(0)
    boundary=torch.cat((torch.ones_like(weights[:1]),(1-weights.cumsum(0)).clamp(0,1)))
    return dict(edges=edges,boundary=boundary,render_calls=LAYERS//3)


def transmittance_bounds(context,depth,coordinates,camera):
    """[B,P] depths, [B,H,W,2] pixels -> [B,P,lower/upper,H,W]."""
    b,h,w,_=coordinates.shape
    boundary=context['boundary']
    grid=(coordinates+.5)/coordinates.new_tensor([camera.image_width,camera.image_height])*2-1
    sampled=F.grid_sample(boundary[None],grid.reshape(1,b*h,w,2).to(boundary),align_corners=False,
                          padding_mode='border')[0].reshape(LAYERS+1,b,h,w).permute(1,0,2,3)
    bins=torch.bucketize(depth.contiguous().to(context['edges']),context['edges'])
    ids=torch.arange(b,device=depth.device)[:,None]
    return torch.stack((sampled[ids,bins+1],sampled[ids,bins]),2)


def _grid(center,cov,valid,camera,dtype):
    extent=3*cov.diagonal(dim1=-2,dim2=-1).sqrt()
    lo=(center-extent).masked_fill(~valid[...,None],torch.inf).amin(1)
    hi=(center+extent).masked_fill(~valid[...,None],-torch.inf).amax(1)
    alive=valid.any(1)
    lo=torch.where(alive[:,None],lo,0);hi=torch.where(alive[:,None],hi,0)
    limit=lo.new_tensor([camera.image_width-1,camera.image_height-1])
    lo=lo.clamp_min(0).minimum(limit);hi=hi.clamp_min(0).minimum(limit)
    axis=torch.linspace(-1,1,7,device=lo.device,dtype=lo.dtype)
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');unit=torch.stack((xx,yy),-1)
    coords=(hi+lo)[:,None,None]/2+unit[None]*(hi-lo)[:,None,None]/2
    return coords.to(dtype),((hi-lo)/limit.clamp_min(1)).to(dtype),alive


def _alpha(raw,center,cov,active,coordinates):
    delta=coordinates[:,None].double()-center[:,:,None,None]
    distance=torch.einsum('bphwi,bpij,bphwj->bphw',delta,torch.linalg.inv(cov),delta)
    alpha=(torch.exp(-.5*distance)*raw[:,:,3,None,None].sigmoid()).clamp_max(.99)
    return torch.where((alpha>=1/255)&active[:,:,None,None],alpha,0).to(raw)


def _colors(raw,camera,degree):
    from utils.sh_utils import eval_sh
    dc=raw[...,11:14].unsqueeze(-2)
    rest=raw[...,14:].reshape(*raw.shape[:-1],3,(degree+1)**2-1).transpose(-1,-2)
    sh=torch.cat((dc,rest),-2).transpose(-1,-2)
    direction=F.normalize(raw[...,:3]-camera.camera_center.to(raw),dim=-1)
    return (eval_sh(degree,sh,direction)+.5).clamp_min(0)


def _pool(x):
    return F.adaptive_avg_pool2d(x.reshape(-1,1,7,7),(3,3)).reshape(x.shape[0],-1)


@torch.no_grad()
def visible_block_views(blocks,tiers,columns,cameras,baseline,contexts,degree):
    """blocks [B,4,P,D], tiers [B,4,P]; 272 features per camera.

    Each candidate uses the baseline visibility bounds. These response maps
    approximate changes inside the block; exact interventions provide labels.
    """
    b,k,p,d=blocks.shape;result=[]
    for camera,(image,target,_),ctx in zip(cameras,baseline,contexts):
        center,cov,z,visible=project_footprints(blocks.reshape(-1,d),camera)
        center=center.reshape(b,k*p,2);cov=cov.reshape(b,k*p,2,2)
        z=z.reshape(b,k*p);active=visible.reshape(b,k*p)&(tiers.reshape(b,k*p)>0)
        grid,extent,alive=_grid(center,cov,active,camera,blocks.dtype)
        raw=blocks.reshape(b,k*p,d)
        alpha=_alpha(raw,center,cov,active,grid)
        bounds=transmittance_bounds(ctx,z,grid,camera)
        color=_colors(raw,camera,degree)
        # Keep the tier axis: sum neighbors only, preserving signed RGB changes.
        weights=(alpha[:,:,None]*bounds).reshape(b,k,p,2,7,7)
        colors=color.reshape(b,k,p,3)
        rgb=torch.einsum('bkplhw,bkpc->bklchw',weights,colors)
        change=_pool((rgb[:,1:]-rgb[:,:1]).reshape(b,18,7,7))
        image_grid=(grid+.5)/grid.new_tensor([camera.image_width,camera.image_height])*2-1
        maps=torch.cat((image,image-target))[None]
        patch=F.grid_sample(maps,image_grid.reshape(1,b*7,7,2).to(maps),align_corners=False)[0]
        patch=patch.reshape(6,b,7,7).permute(1,0,2,3)
        ids=torch.arange(b,device=blocks.device)[:,None]
        selected=columns[:,None]+torch.arange(1,4,device=blocks.device)[None]*p
        sc,sv,sz=center[ids,selected],cov[ids,selected],z[ids,selected]
        sa=active[ids,selected];sr=raw[ids,selected]
        selected_grid,_,selected_alive=_grid(sc,sv,sa,camera,blocks.dtype)
        selected_alpha=_alpha(sr,sc,sv,sa,selected_grid)
        selected_bounds=transmittance_bounds(ctx,sz,selected_grid,camera)
        selected_features=_pool(selected_alpha[:,:,None]*selected_bounds)*selected_alive[:,None]
        feature=torch.cat((change,_pool(patch),selected_features,extent),-1)*alive[:,None]
        result.append(feature.to(blocks))
    output=torch.stack(result,1)
    if output.shape[-1]!=272 or not torch.isfinite(output).all():raise ValueError('invalid visible response features')
    return output
