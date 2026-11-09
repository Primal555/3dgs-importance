"""Independent RD magnitude/ranking tasks with parameter-gradient balancing."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .benefit_isolated_positive_model import IsolatedPositiveLearner
from .physical_benefit import PhysicalLearner

OBJECTIVE='continuous_rd_rank_gradnorm_v1'


def rd_losses(prediction, truth, rates, scale, temperature, regression_factor):
    if temperature<=0 or regression_factor<=0:
        raise ValueError('positive temperature and regression coefficient required')
    error=prediction-truth
    magnitude=regression_factor*((error-error.mean(-2,keepdim=True))/scale).square().mean((-2,-1))
    i,j=torch.triu_indices(3,3,offset=1,device=prediction.device)
    predicted_cost=prediction.sum(-1)+rates
    true_cost=truth.sum(-1)+rates
    logits=-(predicted_cost[...,j]-predicted_cost[...,i])/temperature
    teacher=(-(true_cost[...,j]-true_cost[...,i])/temperature).detach()
    probability=torch.sigmoid(teacher)
    # Target entropy is constant in network parameters; subtracting it gives
    # a zero-at-teacher progress measure for GradNorm, with RankNet gradients.
    ranking=(F.binary_cross_entropy_with_logits(logits,probability,reduction='none')-
        F.binary_cross_entropy_with_logits(teacher,probability,reduction='none')).clamp_min(0.).mean(-1)
    return dict(magnitude=magnitude,ranking=ranking)


class RankGradNormLearner(IsolatedPositiveLearner):
    def configure(self,temperature,regression_factor):
        if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(regression_factor) or regression_factor<=0:
            raise ValueError('invalid RD supervision configuration')
        self.rank_temperature=float(temperature);self.regression_factor=float(regression_factor)

    def _batch_losses(self,groups):
        if not groups or any(g.get('supervision')=='action_gap' or not g['views'] for g in groups):
            raise ValueError('nonempty complete teacher groups required')
        sizes=[len(g['views']) for g in groups]
        p=self.network(*self.pack([v['inputs'] for g in groups for v in g['views']])).double()
        t=torch.stack([v['values'] for g in groups for v in g['views']]).to(p)
        rates=torch.stack([torch.as_tensor(g['rates']) for g in groups]).to(p)
        owner=torch.repeat_interleave(torch.arange(len(groups),device=p.device),torch.tensor(sizes,device=p.device))
        per_view=rd_losses(p,t,rates[owner],self.scales['rd'],self.rank_temperature,self.regression_factor)
        magnitude=torch.stack([v.mean() for v in per_view['magnitude'].split(sizes)])
        mean_p=torch.stack([v.mean(0) for v in p.split(sizes)])
        mean_t=torch.stack([v.mean(0) for v in t.split(sizes)])
        ranking=rd_losses(mean_p,mean_t,rates,self.scales['rd'],self.rank_temperature,self.regression_factor)['ranking']
        return dict(magnitude=magnitude,ranking=ranking)

    def state_dict(self):
        return dict(PhysicalLearner.state_dict(self),direction_objective=OBJECTIVE,
                    rank_temperature=self.rank_temperature,regression_factor=self.regression_factor)

    def load_state_dict(self,state):
        if state.get('direction_objective')==OBJECTIVE:
            if state['rank_temperature']!=self.rank_temperature or state['regression_factor']!=self.regression_factor:
                raise ValueError('RD objective resume mismatch')
            PhysicalLearner.load_state_dict(self,state)
        else:
            if hasattr(self,'rank_temperature'):raise ValueError('wrong objective checkpoint')
            super().load_state_dict(state)


class GradNormBalancer:
    def __init__(self,names,reference_losses,*,device,alpha=1.5,lr=.025,ema=.95):
        if not names or len(names)!=len(reference_losses) or len(set(names))!=len(names):
            raise ValueError('matching unique task names and references required')
        if not math.isfinite(alpha) or alpha<0 or not math.isfinite(lr) or lr<=0 or not 0<=ema<1:
            raise ValueError('invalid GradNorm hyperparameters')
        self.names=list(names);self.alpha=alpha;self.ema=ema
        self.reference=torch.as_tensor(reference_losses,dtype=torch.float64,device=device).detach().clone()
        if not torch.isfinite(self.reference).all() or (self.reference<0).any():raise ValueError('invalid reference losses')
        self.reference.clamp_min_(1e-12);self.progress=self.reference.clone()
        self.logits=nn.Parameter(torch.zeros(len(names),dtype=torch.float64,device=device))
        self.optimizer=torch.optim.Adam([self.logits],lr=lr);self.steps=0

    def weights(self):
        return len(self.names)*self.logits.softmax(-1)

    def measure(self,losses,parameters):
        if losses.shape!=self.reference.shape or not torch.isfinite(losses).all():
            raise ValueError('finite task losses with the configured shape required')
        base=[]
        for loss in losses:
            gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            squares=[g.detach().double().square().sum() for g in gradients if g is not None]
            base.append(torch.stack(squares).sum().sqrt() if squares else loss.detach()*0)
        return self.measure_gradients(losses,torch.stack(base).detach())

    def measure_gradients(self,losses,base):
        """Update task statistics after accumulating complete gradient vectors."""
        losses=losses.detach().to(self.reference)
        base=base.detach().to(self.reference)
        if (losses.shape!=self.reference.shape or base.shape!=self.reference.shape or
                not torch.isfinite(losses).all() or not torch.isfinite(base).all() or (base<0).any()):
            raise ValueError('finite task losses and nonnegative matching gradient norms required')
        self.progress.mul_(self.ema).add_(losses.detach(),alpha=1-self.ema)
        rate=self.progress/self.reference;rate=rate/rate.mean().clamp_min(1e-30)
        weights=self.weights();weighted=weights*base
        target=(weighted.mean()*rate.pow(self.alpha)).detach()
        objective=(weighted-target).abs().sum()
        self.optimizer.zero_grad(set_to_none=True)
        self.logits.grad=torch.autograd.grad(objective,self.logits)[0]
        self.steps+=1
        return dict(names=self.names,losses=losses.detach().cpu().tolist(),weights=weights.detach().cpu().tolist(),
            base_gradient_norms=base.cpu().tolist(),weighted_gradient_norms=weighted.detach().cpu().tolist(),
            target_gradient_norms=target.cpu().tolist(),relative_training_rates=rate.cpu().tolist(),
            balancing_loss=float(objective.detach()))

    def update_weights(self):
        self.optimizer.step()

    def state_dict(self):
        return dict(names=self.names,reference=self.reference.cpu(),progress=self.progress.cpu(),
                    logits=self.logits.detach().cpu(),optimizer=self.optimizer.state_dict(),steps=self.steps)

    def load_state_dict(self,state):
        if state['names']!=self.names or not torch.equal(state['reference'].to(self.reference),self.reference):
            raise ValueError('GradNorm references differ')
        with torch.no_grad():self.logits.copy_(state['logits'].to(self.logits))
        self.progress.copy_(state['progress'].to(self.progress));self.optimizer.load_state_dict(state['optimizer'])
        self.steps=int(state['steps'])


class StratifiedCycles:
    def __init__(self,labels,batch_size,seed):
        self.labels=list(labels);self.strata=sorted(set(labels))
        if not labels or batch_size<len(self.strata) or any(i not in (0,1) for i in labels):
            raise ValueError('batch must cover each nonempty training stratum')
        self.members={s:[i for i,k in enumerate(labels) if k==s] for s in self.strata}
        tail=max(1,min(batch_size-1,round(batch_size*labels.count(1)/len(labels)))) if len(self.strata)==2 else None
        self.quotas={s:(tail if s==1 else batch_size-tail) if tail is not None else batch_size for s in self.strata}
        self.generator=torch.Generator().manual_seed(seed);self.orders={s:[] for s in self.strata}
        self.cursors={s:0 for s in self.strata};self.visits=[0]*len(labels)

    def next(self):
        chosen=[]
        for s in self.strata:
            remain=self.quotas[s]
            while remain:
                if self.cursors[s]==len(self.orders[s]):
                    ids=torch.randperm(len(self.members[s]),generator=self.generator).tolist()
                    self.orders[s]=[self.members[s][i] for i in ids];self.cursors[s]=0
                take=min(remain,len(self.orders[s])-self.cursors[s]);start=self.cursors[s]
                chosen.extend(self.orders[s][start:start+take]);self.cursors[s]+=take;remain-=take
        permutation=torch.randperm(len(chosen),generator=self.generator).tolist()
        chosen=[chosen[i] for i in permutation]
        for i in chosen:self.visits[i]+=1
        return chosen

    def state_dict(self):
        return copy.deepcopy(dict(labels=self.labels,quotas=self.quotas,orders=self.orders,cursors=self.cursors,
                    visits=self.visits,generator=self.generator.get_state()))

    def load_state_dict(self,state):
        if state['labels']!=self.labels or state['quotas']!=self.quotas:raise ValueError('training strata differ')
        state=copy.deepcopy(state)
        self.orders=state['orders'];self.cursors=state['cursors'];self.visits=state['visits']
        self.generator.set_state(state['generator'])
