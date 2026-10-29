"""Calibrated four-state values; no tangent input or classification margin."""
import torch
from torch import nn
from torch.nn import functional as F

from .allocation_benefit import mlp,Residual
from .allocation_benefit_learning import BenefitLearner
from .structured_evidence import BoundedMirrorTable

FAMILY='render_response_value_v1'


class ResponseMirrorTable(BoundedMirrorTable):
    @torch.no_grad()
    def update(self,keep_costs,tier_ids,tier_costs,keep_lr,tier_lr,global_gain):
        untouched=torch.ones(len(self.keep_logits),device=self.keep_logits.device,dtype=torch.bool)
        untouched[tier_ids.to(untouched.device)]=False
        saved=self.keep_logits[untouched].clone()
        super().update(keep_costs,tier_ids,tier_costs,keep_lr,tier_lr,global_gain)
        self.keep_logits[untouched]=saved


class ResponseProtocol:
    version='response_patch_values_v1'

    def __init__(self,attr_dim,latent_dim=None):
        self.attr_dim=attr_dim
        self.point_dim=3+2*attr_dim+4
        self.action_dim=12+4*attr_dim+40
        self.view_dim=9*49+4*(49+10)
        point=torch.ones(self.point_dim,dtype=torch.bool);point[-4:]=False
        views=torch.ones(self.view_dim,dtype=torch.bool)
        for tier in range(4):views[9*49+tier*59+49+6]=False
        self.continuous=dict(point=point,
            action=torch.arange(self.action_dim)>=12,views=torch.ones(self.view_dim,dtype=torch.bool),
            aux=torch.zeros(15,dtype=torch.bool))
        self.continuous['views']=views

    def identity(self):
        return dict(version=self.version,attr_dim=self.attr_dim,point_dim=self.point_dim,
            action_dim=self.action_dim,view_dim=self.view_dim,patch_size=7,
            context='RGB/residual/alpha/depth moments',first_order_input=False)


class ResponseValueNetwork(nn.Module):
    def __init__(self,point_dim,action_dim,view_dim=677,aux_dim=15,*,mse_only=False,continuous=None):
        super().__init__();self.mse_only=mse_only
        self.dimensions=dict(point=point_dim,action=action_dim,views=view_dim,aux=aux_dim)
        self.point_encoder=mlp(point_dim,64);self.action_encoder=mlp(action_dim-12,64)
        self.view_encoder=mlp(view_dim,64)
        self.fusion=nn.Sequential(nn.Linear(256,128),nn.GELU(),Residual(128),Residual(128),nn.Linear(128,6))
        for key,width in self.dimensions.items():
            self.register_buffer(key+'_mean',torch.zeros(width));self.register_buffer(key+'_std',torch.ones(width))
            mask=torch.ones(width,dtype=torch.bool) if continuous is None else continuous[key].bool().clone()
            if key=='action':mask[:12]=False
            self.register_buffer(key+'_continuous',mask)
        self.register_buffer('scales',torch.ones(2)*1e-6)
        self.register_buffer('component_scale',torch.ones(2)*1e-6)
        self.register_buffer('ready',torch.tensor(False))

    def values(self,x):
        def norm(key):return (x[key]-getattr(self,key+'_mean'))/getattr(self,key+'_std')
        p=self.point_encoder(norm('point'));a=self.action_encoder(norm('action')[:,12:])
        v=self.view_encoder(norm('views'));v=torch.cat((v.mean(1),v.amax(1)),1)
        result=self.fusion(torch.cat((p,a,v),1)).reshape(-1,3,2)*self.component_scale
        if self.mse_only:result=result*result.new_tensor([1.,0.])
        return torch.cat((torch.zeros_like(result[:,:1]),result),1)

    def forward(self,x,branch,*,masked=False):
        values=self.values(x);n=len(values);ids=torch.arange(n,device=values.device)
        old=x['action'][:,2:6].argmax(-1);new=x['action'][:,6:9].argmax(-1)+1
        positive=branch=='positive' if isinstance(branch,str) else branch.to(values.device)==0
        if isinstance(positive,bool):positive=torch.full_like(old,positive,dtype=torch.bool)
        return values[ids,new]-torch.where(positive[:,None],values[ids,old],torch.zeros_like(values[:,0]))

    @torch.no_grad()
    def fit(self,records):
        if bool(self.ready) or not records or any(r['validation'] for r in records):
            raise ValueError('value normalization needs fresh training-only records')
        for key in ('point','action','views'):
            width=self.dimensions[key]
            data=torch.cat([r['features'][key].reshape(-1,width).double().cpu() for r in records])
            if not torch.isfinite(data).all():raise ValueError('nonfinite response feature')
            mean=data.mean(0);std=data.std(0,correction=0)
            std=torch.where(std>1e-4,std,torch.ones_like(std))
            mask=getattr(self,key+'_continuous').cpu()
            mean=torch.where(mask,mean,0.);std=torch.where(mask,std,1.)
            getattr(self,key+'_mean').copy_(mean);getattr(self,key+'_std').copy_(std)
        targets=torch.cat([r['target'].double() for r in records if r['branch']=='keep'])
        self.component_scale.copy_(torch.quantile(targets.abs(),.9,dim=0).clamp_min(1e-10))
        scale=torch.quantile(targets.abs().amax(-1),.9).clamp_min(1e-10)
        self.scales.fill_(scale);self.ready.fill_(True)


class ResponseValueLearner(BenefitLearner):
    def losses(self,prediction,target,rates,conditional,groups,scales,branches):
        error=(prediction-target)/scales[groups,None]
        rows=F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none').mean(-1)
        rows=rows+.25*F.smooth_l1_loss(error.sum(-1),torch.zeros_like(rows),reduction='none')
        sizes=torch.bincount(groups,minlength=len(scales))
        return torch.zeros_like(scales).scatter_add_(0,groups,rows)/sizes.clamp_min(1)

    def state_dict(self):return dict(super().state_dict(),objective_variant=FAMILY)

    def load_state_dict(self,state):
        if state.get('objective_variant')!=FAMILY:raise ValueError('response value checkpoint family mismatch')
        super().load_state_dict(state)
