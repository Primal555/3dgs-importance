"""Shared per-point retention value and signed adjacent-tier increments."""
import torch
from torch import nn

from .allocation_benefit import BenefitNetwork,FEATURE_KEYS
from .allocation_benefit_learning import BenefitLearner
from .factored_benefit_loss import decision_loss_parts


FAMILY='learned_benefit_factored_v2'
OBJECTIVE='factored_decision_v2'


class FactoredBenefitNetwork(BenefitNetwork):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.dimensions['action']<12:raise ValueError('factored benefit requires the tier action feature protocol')
        self.heads=nn.ModuleList([self.heads[0]])
        self.keep_value_head=nn.Linear(128,2)
        self.keep_increment_head=nn.Linear(128,4)
        self.register_buffer('common_scale',torch.tensor(1e-6))
        self.register_buffer('increment_scale',torch.tensor(1e-6))

    def _hidden(self,x,branches,masked):
        if set(x)!=set(FEATURE_KEYS):raise ValueError('benefit feature protocol mismatch')
        action=self._normalize('action',x['action'])
        target_fields=torch.zeros(self.dimensions['action'],device=action.device,dtype=torch.bool)
        target_fields[6:9]=True;target_fields[10:12]=True
        action=action.masked_fill((branches==1)[:,None]&target_fields[None],0.)
        point=self.point_encoder(self._normalize('point',x['point']))
        encoded_action=self.action_encoder(action)
        views=self.view_encoder(self._normalize('views',x['views']))
        views=torch.cat((views.mean(-2),views.amax(-2)),-1)
        available=torch.zeros_like(x['available']) if masked else x['available']
        aux=self.aux_encoder(self._normalize('aux',x['aux'])*available)*available
        return self.fusion(torch.cat((point,encoded_action,views,aux,available),-1))

    def _factors(self,hidden):
        base=self.keep_value_head(hidden)*self.common_scale
        increments=self.keep_increment_head(hidden).reshape(-1,2,2)*self.increment_scale
        if self.mse_only:
            gate=base.new_tensor([1.,0.]);base=base*gate;increments=increments*gate
        return base,increments

    @staticmethod
    def _values(base,increments):
        relative=torch.cat((torch.zeros_like(base[:,None]),increments.cumsum(1)),1)
        return base[:,None]+relative-relative.mean(1,keepdim=True)

    def keep_factors(self,features,*,masked=False):
        branches=torch.ones(len(features['point']),dtype=torch.long,device=features['point'].device)
        return self._factors(self._hidden(features,branches,masked))

    def keep_values(self,features,*,masked=False):
        return self._values(*self.keep_factors(features,masked=masked))

    def forward(self,features,branch,*,masked=False):
        n=len(features['point']);device=features['point'].device
        if isinstance(branch,str):
            if branch not in ('positive','keep'):raise ValueError('unknown benefit branch')
            branches=torch.full((n,),int(branch=='keep'),device=device,dtype=torch.long)
        else:branches=branch.to(device=device,dtype=torch.long)
        hidden=self._hidden(features,branches,masked)
        positive=self.heads[0](hidden)*self.scales[0]
        values=self._values(*self._factors(hidden))
        target=features['action'][:,6:9].argmax(-1)
        result=torch.where((branches==1)[:,None],values[torch.arange(n,device=device),target],positive)
        return result*result.new_tensor([1.,0.]) if self.mse_only else result

    @torch.no_grad()
    def fit_structure(self,records):
        if not records or any(r['validation'] for r in records):raise ValueError('structure scales require training labels')
        values=[r['target'].detach().double().cpu() for r in records if r['branch']=='keep']
        if not values or any(v.shape!=(3,2) for v in values):raise ValueError('complete keep groups required')
        values=torch.stack(values)
        self.common_scale.copy_(torch.quantile(values.mean(1).abs().amax(-1),.9).clamp_min(1e-12))
        self.increment_scale.copy_(torch.quantile(values.diff(dim=1).abs().amax(-1).flatten(),.9).clamp_min(1e-12))

    @torch.no_grad()
    def fit(self,records):
        super().fit(records);self.fit_structure(records)


class DecisionBenefitLearner(BenefitLearner):
    def losses(self,prediction,target,rates,conditional,groups,scales,branches):
        return decision_loss_parts(prediction,target,rates,conditional,groups,scales,branches,
            self.network.common_scale,self.network.increment_scale)['total']

    def state_dict(self):
        return dict(super().state_dict(),objective_variant=OBJECTIVE)

    def load_state_dict(self,state):
        if state.get('objective_variant')!=OBJECTIVE:raise ValueError('factored objective checkpoint mismatch')
        super().load_state_dict(state)


def fresh_factored(source,seed):
    """Random new weights/Adam; reuse immutable replay and feature normalization."""
    old=source.network
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        net=FactoredBenefitNetwork(*(old.dimensions[k] for k in ('point','action','views','aux')),
            continuous={k:getattr(old,k+'_continuous').cpu() for k in old.dimensions},mse_only=old.mse_only)
    saved=old.state_dict()
    for key,value in net.state_dict().items():
        if key.endswith(('_mean','_std','_continuous')) or key in ('scales','ready'):
            value.copy_(saved[key].cpu())
    net.fit_structure(source.replay.records());net.to(source.device)
    lr,fit_windows,steps,batch,mask,mask_weight=source.settings
    learner=DecisionBenefitLearner(net,static_source=source.replay.static_source,capacity=source.replay.capacity,
        seed=seed,lr=lr,fit_windows=fit_windows,updates_per_window=steps,batch_size=batch,
        mask_fraction=mask,mask_weight=mask_weight)
    learner.replay.load_state_dict(source.replay.state_dict())
    learner.validation.load_state_dict(source.validation.state_dict())
    learner.generator.set_state(source.generator.get_state())
    return learner
