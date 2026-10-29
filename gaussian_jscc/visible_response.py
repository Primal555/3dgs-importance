"""Unified physical values with separately scaled common and tier supervision."""
import torch
from torch.nn import functional as F

from .response_value import ResponseProtocol, ResponseValueNetwork, ResponseValueLearner
from .allocation_benefit_learning import BenefitLearner

FAMILY='visible_response_value_v1'


def supervision_scales(records):
    if not records or any(r['validation'] for r in records):
        raise ValueError('supervision scales require training-only records')
    values=torch.stack([r['target'].double().cpu() for r in records if r['branch']=='keep'])
    if values.shape[1:]!=(3,2) or not torch.isfinite(values).all():
        raise ValueError('complete finite three-tier values required')
    common=values.mean(1);centered=values-common[:,None]
    pairs=torch.stack((values[:,1]-values[:,0],values[:,2]-values[:,0],values[:,2]-values[:,1]),1)
    common_scale=torch.quantile(common.abs(),.9,dim=0).clamp_min(1e-10)
    floor=(common_scale*1e-3).clamp_min(1e-10)
    return dict(common=common_scale,
        centered=torch.quantile(centered.abs().reshape(-1,2),.9,dim=0).maximum(floor),
        pair=torch.quantile(pairs.abs().reshape(-1,2),.9,dim=0).maximum(floor))


class VisibleResponseProtocol(ResponseProtocol):
    version='visible_response_patch_values_v1'

    def __init__(self,attr_dim,latent_dim=None):
        super().__init__(attr_dim,latent_dim)
        self.view_dim+=272
        self.continuous['views']=torch.cat((self.continuous['views'],torch.ones(272,dtype=torch.bool)))

    def identity(self):
        return dict(super().identity(),depth_layers=12,visibility_features=272,
            context='local patch; depth-bound transmittance; signed block RGB response',
            supervision='common/centered/pair training-only q90 scales')


class VisibleResponseNetwork(ResponseValueNetwork):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for name in ('common','centered','pair'):
            self.register_buffer(name+'_scale',torch.ones(2)*1e-6)

    @torch.no_grad()
    def fit_supervision(self,records):
        for name,scale in supervision_scales(records).items():getattr(self,name+'_scale').copy_(scale)

    @torch.no_grad()
    def fit(self,records):
        super().fit(records)
        self.fit_supervision(records)


class VisibleResponseLearner(ResponseValueLearner):
    def losses(self,prediction,target,rates,conditional,groups,scales,branches):
        # Values retain physical units; scaling is confined to supervision.
        error=prediction-target
        sizes=torch.bincount(groups,minlength=len(scales)).clamp_min(1)
        means=error.new_zeros(len(scales),2).index_add(0,groups,error)/sizes[:,None]
        def penalty(value,scale):
            normalized=value/scale
            return F.smooth_l1_loss(normalized,torch.zeros_like(normalized),reduction='none').mean(-1)
        common=penalty(means,self.network.common_scale)
        centered=penalty(error-means[groups],self.network.centered_scale)
        centered=torch.zeros_like(scales).index_add(0,groups,centered)/sizes
        positive=penalty(error,self.network.pair_scale)
        positive=torch.zeros_like(scales).index_add(0,groups,positive)/sizes
        return torch.where(branches==0,positive,(common+centered)/2)

    def state_dict(self):return dict(BenefitLearner.state_dict(self),objective_variant=FAMILY)

    def load_state_dict(self,state):
        if state.get('objective_variant')!=FAMILY:raise ValueError('visible response checkpoint family mismatch')
        BenefitLearner.load_state_dict(self,state)
