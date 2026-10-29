"""Compact attributes and actual baseline visibility, with exact paired labels."""
import time

import numpy as np
import torch

from .pixel_benefit_features import exact_values
from .pixel_benefit_window import PixelWindow

FEATURE_DIM = 158


@torch.no_grad()
def static_features(raw, neighbors=32, workers=1):
    """RAP-inspired statistics; XYZ KNN is queried once in bounded chunks.

    Eight attributes, global z-scores and local z-scores. The SH descriptor is
    coefficient energy, not a camera-dependent colour or a learned score.
    """
    from scipy.spatial import cKDTree
    raw = torch.as_tensor(raw).detach().cpu().float()
    n = len(raw)
    if not n or neighbors < 1:
        raise ValueError('nonempty points and positive neighbor count required')
    xyz = raw[:, :3].double().numpy()
    tree = cKDTree(xyz)
    k = min(n, neighbors+1)
    attributes = torch.empty(n,8)
    attributes[:,1] = raw[:,14:].square().mean(-1).sqrt().log1p() if raw.shape[1]>14 else 0
    attributes[:,2:5] = raw[:,4:7].sort(-1).values
    attributes[:,5] = raw[:,4:7].sum(-1)
    attributes[:,6] = raw[:,3].sigmoid()
    attributes[:,7] = (raw[:,11:14]*.28209479177387814+.5).mean(-1)
    # Only a chunk of KNN indices is retained; the second query uses the same tree.
    def query(first, last):
        distance, indices = tree.query(xyz[first:last], k=k, workers=workers)
        if k == 1:
            return np.zeros((last-first,1)), np.arange(first,last)[:,None]
        # Explicitly remove self even for coincident points with arbitrary tie order.
        ids = np.arange(first,last)[:,None]
        order = np.argsort(indices == ids, axis=1, kind='stable')[:,:k-1]
        return np.take_along_axis(distance,order,1), np.take_along_axis(indices,order,1)
    for first in range(0,n,8192):
        last=min(n,first+8192)
        distances,_ = query(first,last)
        attributes[first:last,0] = torch.from_numpy(distances.mean(1)).float().clamp_min(1e-12).log()
    global_mean = attributes.mean(0)
    global_std = attributes.std(0,unbiased=False).clamp_min(1e-6)
    local = torch.empty_like(attributes)
    for first in range(0,n,8192):
        last=min(n,first+8192)
        _,indices = query(first,last)
        values = attributes[torch.from_numpy(indices)]
        std = values.std(1,unbiased=False).clamp_min(global_std*.05).clamp_min(1e-6)
        local[first:last] = (attributes[first:last]-values.mean(1))/std
    return torch.cat((attributes, ((attributes-global_mean)/global_std).clamp(-8,8), local.clamp(-8,8)),1)


@torch.no_grad()
def visibility_statistics(scene, target):
    """Per-point alpha*T moments using the renderer's exact baseline ordering.

    Columns: potential mass, actual mass, squared mass, max weight, accepted
    unoccluded alpha mass, target RGB weighted sum, residual RGB weighted sum, pixel count.
    Potential mass also describes currently disabled points before early exit.
    """
    target = target.to(scene.image).contiguous()
    if target.shape != scene.image.shape:
        raise ValueError('target shape must match capture')
    if scene.projected.is_cuda:
        from .pixel_local_renderer import _extension
        return _extension().visibility(scene.projected.contiguous(), scene.tile_ranges.contiguous(),
            scene.tile_ids.contiguous(), scene.image.contiguous(), target, scene.height, scene.width)
    p = scene.projected
    output = p.new_zeros(len(p),12)
    axis=torch.arange(16,device=p.device); yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    gx=(scene.width+15)//16
    for tile,(begin,end) in enumerate(scene.tile_ranges.tolist()):
        x=xx+(tile%gx)*16; y=yy+(tile//gx)*16
        valid=(x<scene.width)&(y<scene.height)
        sx,sy=x.clamp_max(scene.width-1),y.clamp_max(scene.height-1)
        ref=target[:,sy,sx]; residual=scene.image[:,sy,sx]-ref
        trans=torch.ones_like(x,dtype=p.dtype); done=~valid
        for i in scene.tile_ids[begin:end].tolist():
            point=p[i]; dx=point[0]-x; dy=point[1]-y
            power=-.5*(point[3]*dx.square()+point[5]*dy.square())-point[4]*dx*dy
            alpha=(point[6]*power.exp()).clamp_max(.99)
            footprint=valid&(power<=0)&(alpha>=1/255)
            eligible=(~done)&footprint
            next_trans=trans*(1-point[7]*alpha)
            stops=eligible&(next_trans<1e-4)
            accepted=eligible&~stops
            weight=torch.where(accepted,alpha*trans,0)
            output[i,0]+=weight.sum(); output[i,1]+=weight.sum()*point[7]
            output[i,2]+=weight.square().sum(); output[i,3]=torch.maximum(output[i,3],weight.max())
            output[i,4]+=torch.where(footprint,alpha,0).sum()
            output[i,5:8]+=(ref*weight).sum((1,2))
            output[i,8:11]+=(residual*weight).sum((1,2))
            output[i,11]+=accepted.sum()
            trans=torch.where(accepted,next_trans,trans); done|=stops
    return output


def _summary(stats, area):
    result=stats.clone()
    mass=stats[...,0:1].clamp_min(1e-12)
    result[...,5:11]=stats[...,5:11]/mass
    result[...,0:3]=torch.log1p(stats[...,0:3])
    result[...,4]=torch.log1p(stats[...,4])
    result[...,11]=stats[...,11]/area
    return result


def _state_attributes(raw, projected):
    mask=projected[...,7:8]
    quat=torch.nn.functional.normalize(raw[...,7:11],dim=-1)
    quat=quat*torch.where(quat[...,:1]<0,-1.,1.)
    energy=raw[...,14:].square().mean(-1,keepdim=True).sqrt().log1p() if raw.shape[-1]>14 else mask*0
    attributes=torch.cat((projected[...,8:11],raw[...,3:4].sigmoid(),raw[...,4:7],
        projected[...,3:6],projected[...,11:12].clamp_min(0).log1p(),
        projected[...,2:3].clamp_min(0).log1p(),torch.ones_like(mask),energy,quat),-1)
    return attributes*mask


@torch.no_grad()
def candidate_features(static, stats, raw, projected, point, height, width, d_ref, lambda_s):
    """All four states share descriptors; no tier index or first-order input."""
    states=_state_attributes(raw,projected)
    weights=stats[:,0].clamp_min(0)
    # Unoccluded mass backs up the descriptor of fully hidden contributors.
    weights=weights+.05*stats[:,4].clamp_min(0)
    weights=weights/weights.sum().clamp_min(1e-12)
    delta=states-states[:1]
    common=torch.cat((static.to(raw),_summary(stats[point],height*width),
        _summary(stats.sum(0),height*width),raw.new_tensor([d_ref,lambda_s])))
    features=torch.cat((common.expand(4,-1),states[:,point],(states*weights[None,:,None]).sum(1),
        delta[:,point],(delta*weights[None,:,None]).sum(1),
        (delta.abs()*weights[None,:,None]).sum(1),delta.abs().amax(1)),1)
    if features.shape != (4,FEATURE_DIM) or not torch.isfinite(features).all():
        raise ValueError('invalid physical feature values')
    return features


class PhysicalWindow(PixelWindow):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.static = None

    def state_dict(self):
        return dict(super().state_dict(), static=self.static)

    def load_state_dict(self,state):
        super().load_state_dict(state)
        self.static = state['static']
        if self.static is not None and (self.static.shape != (len(self.table.logits),24) or
                                       not torch.isfinite(self.static).all()):
            raise ValueError('invalid static physical feature state')

    def prepare_camera(self,tile,target):
        if self.static is None:
            self.static=static_features(self.scene.raw,self.args.neighbors,self.args.cpu_threads)
        tile._physical_stats=visibility_statistics(tile,target)

    def capture_inputs(self,mixed):
        # q0 decoder attributes are placeholders. Zero-mask source probes keep
        # their real location and footprint without adding any rendered colour.
        inactive=(mixed[:,-1]==0).nonzero(as_tuple=True)[0]
        if not len(inactive):
            return mixed
        result=mixed.clone()
        sources=self.render_sources[inactive.cpu()]
        result[inactive,:-1]=self.scene.raw[sources].to(mixed)
        return result

    def build_view(self,tile,changed,projected,target,*,source,block_ids,raw,with_labels):
        start=time.perf_counter()
        point=int((block_ids==source).nonzero(as_tuple=True)[0].item())
        features=candidate_features(self.static[source],tile._physical_stats[changed],raw,projected,point,
            tile.height,tile.width,self.reference['d_ref'],self.reference['lambda_s'])
        result=dict(inputs={'features':features},height=tile.height,width=tile.width,
                    d_ref=self.reference['d_ref'],tile_count=0,feature_seconds=time.perf_counter()-start)
        if with_labels:
            start=time.perf_counter()
            result['values']=exact_values(tile,changed,projected,target,self.reference['lambda_s'],self.reference['d_ref']).cpu()
            result['tile_count']=len(tile.affected_tiles(changed,projected,halo_tiles=1))
            result['teacher_seconds']=time.perf_counter()-start
        return result
