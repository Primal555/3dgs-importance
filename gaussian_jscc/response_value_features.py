"""Projected candidate footprints and frozen local image/depth context."""
import math
import torch
from torch.nn import functional as F


def _rotation(quaternion):
    w,x,y,z=F.normalize(quaternion,dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)


@torch.no_grad()
def project_footprints(raw,camera):
    # High-resolution thin ellipsoids need the covariance's .3-pixel floor to
    # survive cancellation; feature construction is small and runs in float64.
    raw=raw.double()
    transform=camera.world_view_transform.to(raw)
    xyz=torch.cat((raw[:,:3],torch.ones_like(raw[:,:1])),-1)@transform
    depth=xyz[:,2];z=depth.clamp_min(1e-4)
    fx=camera.image_width/(2*math.tan(camera.FoVx/2));fy=camera.image_height/(2*math.tan(camera.FoVy/2))
    center=torch.stack((fx*xyz[:,0]/z+(camera.image_width-1)/2,
                        fy*xyz[:,1]/z+(camera.image_height-1)/2),-1)
    rotation=_rotation(raw[:,7:11]);scales=raw[:,4:7].clamp(-20,10).exp()
    factor=rotation*scales[:,None,:];world=factor@factor.transpose(-1,-2)
    view=transform[:3,:3];cov=view.T@world@view
    jac=raw.new_zeros(len(raw),2,3);jac[:,0,0]=fx/z;jac[:,1,1]=fy/z
    ratio_x=(xyz[:,0]/z).clamp(-1.3*math.tan(camera.FoVx/2),1.3*math.tan(camera.FoVx/2))
    ratio_y=(xyz[:,1]/z).clamp(-1.3*math.tan(camera.FoVy/2),1.3*math.tan(camera.FoVy/2))
    jac[:,0,2]=-fx*ratio_x/z;jac[:,1,2]=-fy*ratio_y/z
    cov=jac@cov@jac.transpose(-1,-2)
    # Preserve positive definiteness even at the codec's largest valid scales.
    jitter=.3+cov.diagonal(dim1=-2,dim2=-1).amax(-1)*torch.finfo(raw.dtype).eps*16
    cov=cov+torch.eye(2,device=raw.device,dtype=raw.dtype)*jitter[:,None,None]
    extent=3*cov.diagonal(dim1=-2,dim2=-1).sqrt()
    visible=(depth>.2)&(center[:,0]+extent[:,0]>=0)&(center[:,0]-extent[:,0]<camera.image_width)
    visible&=(center[:,1]+extent[:,1]>=0)&(center[:,1]-extent[:,1]<camera.image_height)
    return center.nan_to_num(),cov.nan_to_num(),depth,visible


@torch.no_grad()
def depth_context(scene,camera,degree,render,depth_scale):
    raw=scene[:,:-1];world=torch.cat((raw[:,:3],torch.ones_like(raw[:,:1])),-1)
    depth=(world@camera.world_view_transform.to(raw))[:,2]/depth_scale
    colors=torch.stack((torch.ones_like(depth),depth,depth.square()),-1)
    moments=render(raw,camera,degree,False,scene[:,-1],colors=colors)
    alpha=moments[:1].clamp(0,1);den=alpha.clamp_min(1e-6)
    mean=moments[1:2]/den;var=(moments[2:3]/den-mean.square()).clamp_min(0)
    return torch.cat((alpha,mean,var),0)


@torch.no_grad()
def patch_views(candidates,cameras,baseline,depth_maps,degree,depth_scale):
    """candidates [N,4,D]; patches cover all three positive footprints."""
    from utils.sh_utils import eval_sh
    n=len(candidates);device=candidates.device;output=[]
    axis=torch.linspace(-1,1,7,device=device,dtype=candidates.dtype)
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');unit=torch.stack((xx,yy),-1)
    for camera,(image,target,_),depth_map in zip(cameras,baseline,depth_maps):
        center,cov,z,visible=project_footprints(candidates.reshape(-1,candidates.shape[-1]),camera)
        center,cov,z,visible=(t.reshape(n,4,*t.shape[1:]) for t in (center,cov,z,visible))
        extent=3*cov[:,1:].diagonal(dim1=-2,dim2=-1).sqrt()
        lo=(center[:,1:]-extent).amin(1);hi=(center[:,1:]+extent).amax(1)
        # Visible image intersection keeps oversized ellipsoids fully represented on-screen.
        limits=candidates.new_tensor([camera.image_width-1,camera.image_height-1])
        lo=lo.clamp_min(0).minimum(limits);hi=hi.clamp_min(0).minimum(limits)
        coordinates=(hi+lo)[:,None,None,:]/2+unit[None]*(hi-lo)[:,None,None,:]/2
        grid=(coordinates+.5)/candidates.new_tensor([camera.image_width,camera.image_height])*2-1
        maps=torch.cat((image,image-target,depth_map),0)[None]
        context=F.grid_sample(maps,grid.reshape(1,n*7,7,2).to(maps),align_corners=False)[0]
        context=context.reshape(9,n,7,7).permute(1,0,2,3).flatten(1)
        point=(coordinates[:,None]-center[:,:,None,None])
        inv=torch.linalg.inv(cov)
        distance=torch.einsum('nkhwi,nkij,nkhwj->nkhw',point,inv,point)
        alpha=candidates[:,:,3].sigmoid()*visible
        alpha[:,0]=0
        footprint=(-.5*distance).exp()*alpha[:,:,None,None]
        dc=candidates[:,:,11:14].reshape(n,4,1,3)
        rest=candidates[:,:,14:].reshape(n,4,3,(degree+1)**2-1).transpose(-1,-2)
        sh=torch.cat((dc,rest),-2).transpose(-1,-2)
        direction=F.normalize(candidates[:,:,:3]-camera.camera_center.to(candidates),dim=-1)
        color=(eval_sh(degree,sh,direction)+.5).clamp_min(0)
        diag=cov.diagonal(dim1=-2,dim2=-1)
        meta=torch.cat((torch.sign(z)[...,None]*torch.log1p(z.abs()[...,None]/depth_scale),
            center/candidates.new_tensor([camera.image_width,camera.image_height]),
            torch.log1p(diag[:,:,:1]),(cov[:,:,0,1]/diag.prod(-1).sqrt().clamp_min(1e-8))[...,None],
            torch.log1p(diag[:,:,1:]),visible[...,None].to(candidates),color),-1)
        local=torch.cat((footprint.flatten(2),meta),-1).flatten(1)
        alive=visible[:,1:].any(-1)
        output.append((torch.cat((context,local),-1)*alive[:,None]).to(candidates))
    values=torch.stack(output,1)
    if not torch.isfinite(values).all():raise ValueError('nonfinite projected response context')
    return values
