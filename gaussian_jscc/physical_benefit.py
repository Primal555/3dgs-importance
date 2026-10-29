"""Shared four-state benefit scorer and cost-weighted RD decision supervision."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .physical_benefit_features import FEATURE_DIM
from .pixel_benefit_learning import summarize_pixel_values


class PhysicalBenefitNetwork(nn.Module):
    def __init__(self,hidden=64,dropout=.05):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(FEATURE_DIM,hidden),nn.Linear(hidden,hidden//2),nn.Linear(hidden//2,2)])
        self.dropout=float(dropout)
        self.register_buffer('input_mean',torch.zeros(FEATURE_DIM))
        self.register_buffer('input_scale',torch.ones(FEATURE_DIM))
        self.register_buffer('output_scale',torch.ones(2))

    def forward(self,features):
        x=(features-self.input_mean)/self.input_scale
        for layer in self.layers[:-1]:
            x=F.silu(layer(x))
            if self.training and self.dropout:
                # Shared across all states; identical candidates always cancel.
                mask=F.dropout(torch.ones_like(x[...,:1,:]),p=self.dropout,training=True)
                x=x*mask
        scores=self.layers[-1](x)*self.output_scale
        difference=scores[...,1:,:]-scores[...,:1,:]
        identical=(features[...,1:,:]==features[...,:1,:]).all(-1,keepdim=True)
        return difference.masked_fill(identical,0.)


def decision_loss(prediction,truth,rates,scale,tie_fraction=.02):
    """Six unordered RD pairs, weighted by exact true excess cost.

    The fixed training RD scale preserves cost units. Near ties contribute zero;
    group weights are not divided by their own gap (which would amplify ties).
    """
    scale=torch.as_tensor(scale,device=prediction.device,dtype=prediction.dtype).clamp_min(1e-8)
    p=torch.cat((prediction.new_zeros(*prediction.shape[:-2],1),prediction.sum(-1)+rates),-1)/scale
    t=torch.cat((truth.new_zeros(*truth.shape[:-2],1),truth.sum(-1)+rates),-1)/scale
    i,j=torch.triu_indices(4,4,offset=1,device=p.device)
    actual=t[...,j]-t[...,i]; estimate=p[...,j]-p[...,i]
    weight=torch.where(actual.abs()>tie_fraction,actual.abs(),0).detach()
    return (weight*F.softplus(-actual.sign()*estimate)).mean(-1)


class PhysicalLearner:
    def __init__(self,replay,device,hidden=64,lr=1e-4,dropout=.05,weight_decay=1e-4,
                 decision_weight=1.,tie_fraction=.02):
        self.replay,self.device=replay,torch.device(device)
        self.network=PhysicalBenefitNetwork(hidden,dropout).to(self.device)
        self.optimizer=torch.optim.AdamW(self.network.parameters(),lr=lr,weight_decay=weight_decay)
        self.decision_weight,self.tie_fraction=float(decision_weight),float(tie_fraction)
        self.scales=None; self.updates=0

    def fit_scales(self):
        if self.scales is not None:
            raise ValueError('training scales already fixed')
        sx=torch.zeros(FEATURE_DIM,dtype=torch.float64); sx2=torch.zeros_like(sx)
        sy=torch.zeros(2,dtype=torch.float64); count_x=count_y=0.
        for group in self.replay.training_records():
            weight=float(group.get('importance_weight',1.))
            if not math.isfinite(weight) or weight<=0:
                raise ValueError('invalid source importance')
            for view in group['views']:
                x=view['inputs']['features'].double().cpu(); y=view['values'].double().cpu()
                sx+=x.sum(0)*weight; sx2+=x.square().sum(0)*weight; count_x+=len(x)*weight
                sy+=y.square().sum(0)*weight; count_y+=len(y)*weight
        if not count_x or not count_y:
            raise ValueError('empty training replay')
        mean=sx/count_x; std=(sx2/count_x-mean.square()).clamp_min(0).sqrt()
        std=std.clamp_min(torch.maximum(mean.abs()*1e-3,mean.new_full(mean.shape,1e-3)))
        scale=(sy/count_y).sqrt(); scale=scale.clamp_min(max(1e-8,float(scale.max())*1e-3))
        self.scales={'value':scale,'rd':scale.sum()}
        self.network.input_mean.copy_(mean); self.network.input_scale.copy_(std)
        self.network.output_scale.copy_(scale)
        return {k:v.clone() for k,v in self.scales.items()}

    def _group_loss(self,group):
        x=torch.stack([v['inputs']['features'] for v in group['views']]).to(self.device,dtype=torch.float32)
        target=torch.stack([v['values'] for v in group['views']]).to(self.device,dtype=torch.float64)
        pred=self.network(x).double()
        scale=self.scales['value'].to(self.device)
        regression=F.smooth_l1_loss(pred/scale,target/scale)
        rates=torch.as_tensor(group['rates'],device=self.device,dtype=torch.float64)
        p,t=pred.mean(0),target.mean(0)
        decision=decision_loss(p,t,rates,self.scales['rd'],self.tie_fraction)
        with torch.no_grad():
            pc=torch.cat((p.new_zeros(1),p.sum(-1)+rates))
            tc=torch.cat((t.new_zeros(1),t.sum(-1)+rates))
            regret=tc[pc.argmin()]-tc.min()
        return regression+self.decision_weight*decision,regression,decision,regret

    def train_updates(self,count,batch_groups=4):
        if self.scales is None or count<0 or batch_groups<=0:
            raise ValueError('fixed training scales and valid update counts required')
        stats={'updates':self.updates,'loss':None}
        self.network.train()
        for _ in range(count):
            groups=self.replay.sample(batch_groups)
            evaluated=[self._group_loss(g) for g in groups]
            weights=[]
            for g in groups:
                weight=float(g.get('importance_weight',1.))*float(g.get('replay_importance_weight',1.))
                if not math.isfinite(weight) or weight<=0: raise ValueError('invalid importance weight')
                weights.append(weight)
            loss=sum(e[0]*w for e,w in zip(evaluated,weights))/len(groups)
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            norm=nn.utils.clip_grad_norm_(self.network.parameters(),10.)
            if not torch.isfinite(norm): raise ValueError('nonfinite physical benefit gradient')
            self.optimizer.step(); self.updates+=1
            for group,e in zip(groups,evaluated):
                if 'replay_record_id' in group:
                    self.replay.update_priority(group['replay_record_id'],float(e[0].detach()))
            stats=dict(loss=float(loss.detach()),regression_loss=sum(float(e[1].detach()) for e in evaluated)/len(groups),
                decision_loss=sum(float(e[2].detach()) for e in evaluated)/len(groups),
                train_regret=sum(float(e[3]) for e in evaluated)/len(groups),grad_norm=float(norm),updates=self.updates)
        return stats

    @torch.inference_mode()
    def predict(self,inputs,height,width,d_ref):
        self.network.eval()
        return self.network(inputs['features'].to(self.device,dtype=torch.float32)).double()

    def validate(self):
        rows=[]
        for group in self.replay.validation_records():
            views=group['views']
            pred=torch.stack([self.predict(v['inputs'],v['height'],v['width'],v['d_ref']).cpu() for v in views]).mean(0)
            truth=torch.stack([v['values'].double().cpu() for v in views]).mean(0)
            rows.append(dict(source=group['source'],step=group['step'],rates=group['rates'],truth=truth,prediction=pred))
        report=summarize_pixel_values(rows)
        report.update(network_updates=self.updates,validation_groups=len(rows),scope='physical features, paired exact whole-block values')
        return report

    def state_dict(self):
        return dict(network=self.network.state_dict(),optimizer=self.optimizer.state_dict(),scales=self.scales,updates=self.updates)

    def load_state_dict(self,state):
        self.network.load_state_dict(state['network']); self.optimizer.load_state_dict(state['optimizer'])
        self.scales=None if state['scales'] is None else {k:v.cpu().clone() for k,v in state['scales'].items()}
        self.updates=int(state['updates'])
