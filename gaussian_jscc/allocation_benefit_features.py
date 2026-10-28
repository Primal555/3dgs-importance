"""Streaming sender-side action features from this window's immutable baseline."""
import math

import torch
from torch.nn import functional as F


class FeatureProtocol:
    version = 'benefit_features_v1'

    def __init__(self, attr_dim, latent_dim):
        self.attr_dim, self.latent_dim = attr_dim, latent_dim
        self.point_dim, self.action_dim = 3+3*attr_dim+latent_dim+3, attr_dim+33
        point = torch.ones(self.point_dim, dtype=torch.bool); point[-3:] = False
        action = torch.ones(self.action_dim, dtype=torch.bool)
        action[:9] = False; action[29+attr_dim:] = False
        views = torch.ones(12, dtype=torch.bool); views[3] = False
        self.continuous = dict(point=point, action=action, views=views, aux=torch.ones(15,dtype=torch.bool))

    def identity(self):
        return dict(version=self.version, attr_dim=self.attr_dim, latent_dim=self.latent_dim,
                    point_dim=self.point_dim, action_dim=self.action_dim, view_dim=12, aux_dim=15,
                    signed_log_fields=['camera_depth','gradient_group_norms','candidate_neighbor_norms'])


def prepare_views(baseline):
    prepared = []
    for image,target,_ in baseline:
        residual = (image-target).float()
        prepared.append((F.avg_pool2d(residual[None],3,1,1),
                         F.avg_pool2d(residual.abs()[None],3,1,1)))
    return prepared


@torch.no_grad()
def view_features(xyz, cameras, baseline, mse, structure, *, prepared=None, depth_scale=1.):
    prepared = prepare_views(baseline) if prepared is None else prepared
    ones = torch.ones_like(xyz[:,:1]); world = torch.cat((xyz,ones),-1)
    values = []
    for camera,(signed,absolute) in zip(cameras,prepared):
        if not hasattr(camera,'world_view_transform') or not hasattr(camera,'full_proj_transform'):
            raise ValueError('benefit features require camera view/projection transforms')
        depth = (world@camera.world_view_transform.to(world))[:,2]
        clip = world@camera.full_proj_transform.to(world)
        w = clip[:,3]; denominator = torch.where(w.abs()>1e-8,w,torch.ones_like(w))
        uv = clip[:,:2]/denominator[:,None]
        inside = (depth>0)&(w>1e-8)&(uv.abs()<=1).all(-1)&torch.isfinite(uv).all(-1)
        grid = uv.nan_to_num().clamp(-10,10)[None,:,None,:]
        def sample(image):
            result = F.grid_sample(image.to(xyz),grid,align_corners=False,padding_mode='zeros')[0,:,:,0].T
            return result*inside[:,None]
        z = torch.sign(depth)*torch.log1p(depth.abs()/max(float(depth_scale),1e-8))
        values.append(torch.cat((z[:,None],uv.nan_to_num().clamp(-10,10),inside[:,None].to(xyz),
                                 sample(signed),sample(absolute),
                                 xyz.new_full((len(xyz),1),mse),xyz.new_full((len(xyz),1),structure)),-1))
    return torch.stack(values,1)


@torch.no_grad()
def keep_rates(q, valid, rates, *, xyz, bits):
    if not math.isfinite(bits) or bits<=0:
        raise ValueError('positive reliable bits/use required')
    thresholds = torch.arange(1,4,device=q.device)
    present = (q[...,None]>=thresholds)&valid[...,None]
    others = present.sum(1,keepdim=True)-present.long()
    # [..., target tier, segment threshold]. Remove the target point first.
    new = (others[...,None,:]>0)|(thresholds[:,None]>=thresholds[None,:])
    extra = new.sum(-1)-(others>0).sum(-1,keepdim=True)
    return (torch.as_tensor(rates[1:],device=q.device,dtype=torch.float64)+xyz+extra*16/bits)*valid[...,None]


def attribute_groups(width):
    return (slice(0,1),slice(1,4),slice(4,8),slice(8,11),slice(11,width))


class BenefitFeatures:
    def __init__(self, model, geometry, task, d_ref):
        self.model, self.geometry, self.task, self.d_ref = model,geometry,task,d_ref
        self.weight = task.distortion.lambda_s
        self.protocol = FeatureProtocol(len(model.attr_mean),model.cfg.latent_dim)
        self.prepared = prepare_views(task.baseline)

    @torch.no_grad()
    def basis(self, f, ids, entry, scene_slice, gradients, conditional):
        q,positive,y,_,_ = entry; valid = ids>=0
        attrs = torch.zeros(*q.shape,len(self.model.attr_mean),device=f.device)
        attrs[valid] = (scene_slice[:,:-1][:,3:]-self.model.attr_mean)/self.model.attr_std
        point = torch.cat((f,attrs,attrs-f[...,3:],y*(q>0)[...,None],
                           (q>0)[...,None],(q==0)[...,None],(q>0)[...,None]),-1).float()
        v = torch.zeros(*q.shape,len(self.task.cameras),12,device=f.device)
        v[valid] = view_features(scene_slice[:,:3],self.task.cameras,self.task.baseline,
            self.task.stats['source_mse'],self.task.stats['source_structure_error'],prepared=self.prepared,
            depth_scale=float(self.geometry.span.max()))
        hist = torch.stack([((q==k)&valid).sum(-1) for k in range(4)],-1).float()
        hist = hist/valid.sum(-1,keepdim=True).clamp_min(1)
        weighted = [gradients[0]/self.d_ref, gradients[1]*(self.weight/self.d_ref)]
        norms = []
        for gradient in weighted:
            for group in attribute_groups(len(self.model.attr_mean)):
                value = gradient[...,3:-1][...,group].norm(dim=-1)
                norms.append(torch.log1p(value))
        gate = torch.stack([g[...,-1] for g in weighted],-1)
        return dict(point=point,views=v,q=q,origin=torch.where(q>0,q,positive),valid=valid,
                    hist=hist,conditional=conditional,gate=gate,norms=torch.stack(norms,-1),ids=ids)

    @torch.no_grad()
    def make(self, basis, rows, columns, targets, branch, *, delta=None, parts=None):
        b = basis; rows,columns,targets = rows.long(),columns.long(),targets.long()
        n=len(rows); old=b['q'][rows,columns]
        rates=torch.as_tensor(self.model.cfg.rates,device=rows.device,dtype=torch.float32)
        change=torch.zeros(n,len(self.model.attr_mean),device=rows.device)
        neighbor=torch.zeros(n,10,device=rows.device)
        if delta is not None:
            change=delta[rows,columns,3:]/self.model.attr_std
            other=b['valid']&(b['q']>0);other=other.clone();other[rows,columns]=False
            normalized=delta[...,3:]/self.model.attr_std
            selected=normalized[rows].abs()*other[rows,...,None]
            count=other[rows].sum(-1).clamp_min(1)
            means,maxima=[],[]
            for group in attribute_groups(len(self.model.attr_mean)):
                values=selected[...,group]
                if values.shape[-1]:
                    means.append(values.sum((-1,-2))/(count*values.shape[-1]))
                    maxima.append(values.amax((-1,-2)))
                else:
                    means.append(count.float()*0);maxima.append(count.float()*0)
            neighbor=torch.log1p(torch.stack(means+maxima,-1))
        marker=torch.tensor([1.,0.] if branch=='positive' else [0.,1.],device=rows.device).expand(n,-1)
        action=torch.cat((marker,F.one_hot(old,4).float(),F.one_hot(targets-1,3).float(),
            torch.stack((rates[old],rates[targets],rates[targets]-rates[old]),-1),b['hist'][rows],
            b['conditional'][rows,columns].float(),change,neighbor,
            torch.full((n,1),float(delta is not None),device=rows.device),
            F.one_hot(b['origin'][rows,columns]-1,3).float()),-1)
        estimate=b['gate'][rows,columns] if parts is None else parts
        aux=torch.cat((estimate,estimate.sum(-1,keepdim=True),b['gate'][rows,columns],
                       b['norms'][rows,columns]),-1).float()
        return dict(point=b['point'][rows,columns],action=action,views=b['views'][rows,columns],
                    aux=aux,available=torch.ones(n,1,device=rows.device))
