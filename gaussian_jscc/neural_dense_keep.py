"""Dense learned keep gates: bounded-memory render VJP and paired calibration."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .allocation_benefit_replay import cpu_copy
from .physical_benefit_features import _summary
from .structured_evidence import BoundedMirrorTable

FAMILY='neural_dense_keep_tiers_v1'
INPUT_DIM=69


def smoothed_gate(logits, previous, noise, blend):
    probability=(1-blend)*previous.double()+blend*logits.double().sigmoid()
    probability=probability.clamp(1e-12,1-1e-12)
    scores=torch.stack(((1-probability).log(),probability.log()),-1)
    gate=(scores+noise.double()).softmax(-1)[...,1]
    return probability,gate


def mixed_keep_label(deltas,conditional,rates,lambda_s,d_ref,beta,c_ref):
    difference=deltas[1:]-deltas[:1]
    rd=(difference[:,0]+lambda_s*difference[:,1])/d_ref+beta*rates/c_ref
    return (rd*conditional.to(rd)).sum()


class DenseKeepNetwork(nn.Module):
    def __init__(self,input_dim=INPUT_DIM,hidden=64):
        super().__init__()
        self.hidden=nn.Sequential(nn.Linear(input_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.output=nn.Linear(hidden,1)
        nn.init.zeros_(self.output.weight);nn.init.constant_(self.output.bias,math.log(99.))

    def forward(self,features):return self.output(self.hidden(torch.asinh(features))).squeeze(-1)


class NeuralKeepTable(BoundedMirrorTable):
    neural_keep=True
    @torch.no_grad()
    def update(self,*args,**kwargs):
        keep=self.keep_logits.clone()
        super().update(*args,**kwargs)
        self.keep_logits.copy_(keep)


def attributes(raw):
    energy=raw[:,14:].square().mean(-1,keepdim=True).sqrt().log1p() if raw.shape[1]>14 else raw[:,:1]*0
    return torch.cat((raw[:,3:4].sigmoid(),raw[:,4:7],raw[:,11:14],energy),-1)


class DenseKeepPolicy:
    def __init__(self,model,scene,table,static_packet,args,calibration_scale,positions,validation):
        self.model,self.scene,self.table,self.args=model,scene,table,args
        self.device=next(model.parameters()).device;self.count=len(scene.raw)
        self.network=DenseKeepNetwork(hidden=args.keep_hidden).to(self.device)
        self.optimizer=torch.optim.Adam(self.network.parameters(),lr=args.keep_network_lr)
        self.static=torch.empty(self.count,27)
        self.static[scene.order]=torch.cat((static_packet,scene.geometry.normalize(scene.raw[:,:3]).float()),-1).cpu()
        self.observation=torch.zeros(self.count,36)
        self.observation[scene.order,:8]=attributes(scene.raw).cpu()
        self.positions,self.validation=positions.cpu(),validation.cpu()
        self.calibration_scale=max(float(calibration_scale),1e-8)
        self.generator=torch.Generator().manual_seed(args.seed+88271)
        self.visits=torch.zeros(self.count,dtype=torch.long);self.updates=0
        self.inputs=self.previous=self.noise=None

    @torch.no_grad()
    def prepare(self,side_proxy):
        conditional=self.table.logits.softmax(-1).float().cpu()
        controls=torch.tensor([self.args.snr/10.,self.args.beta,
            side_proxy['xyz_per_retained']/max(self.model.cfg.rates[-1],1)]).expand(self.count,-1)
        self.inputs=torch.cat((self.static,self.observation,conditional,controls),-1)
        if self.inputs.shape!=(self.count,INPUT_DIM) or not torch.isfinite(self.inputs).all():
            raise ValueError('invalid dense keep features')
        self.previous=self.table.keep_logits.softmax(-1)[:,1].detach().cpu().double()
        probability=torch.empty(self.count,dtype=torch.float64)
        for start in range(0,self.count,self.args.keep_chunk):
            end=min(start+self.args.keep_chunk,self.count)
            logits=self.network(self.inputs[start:end].to(self.device)).double()
            probability[start:end]=((1-self.args.keep_blend)*self.previous[start:end].to(self.device)+
                self.args.keep_blend*logits.sigmoid()).clamp(1e-12,1-1e-12).cpu()
        self.table.keep_logits.copy_(torch.stack(((1-probability).log(),probability.log()),-1).to(self.table.keep_logits))
        self.noise=torch.empty(self.count,2,dtype=torch.float64)

    @torch.no_grad()
    def sample(self,ids,snr):
        safe=ids.clamp_min(0);keep,tier=self.table.branch_scores(safe,snr)
        keep_noise=-torch.empty_like(keep).exponential_().log()
        tier_noise=-torch.empty_like(tier).exponential_().log()
        valid=ids>=0;self.noise[ids[valid].cpu()]=keep_noise[valid].cpu()
        return self.table.sample(safe,snr,noise=torch.cat((keep_noise,tier_noise),-1))

    @torch.no_grad()
    def observe(self,scene,render_sources,captures):
        if not captures:return
        profiles=torch.stack([_summary(tile._physical_stats,tile.height*tile.width) for tile in captures])
        packet=torch.cat((attributes(scene[:,:-1]),F.one_hot((scene[:,-1]>0).long(),4).to(scene),
                          profiles.mean(0),profiles.amax(0)),-1)
        # The current tier, including q0, is supplied below by the provider.
        if packet.shape!=(self.count,36) or not torch.isfinite(packet).all():raise ValueError('invalid keep observation')
        self.observation[render_sources]=packet.float().cpu()

    @torch.no_grad()
    def set_tiers(self,q):
        self.observation[:,8:12]=F.one_hot(q.cpu(),4).float()

    def state_dict(self):
        return dict(version=FAMILY,network=cpu_copy(self.network.state_dict()),optimizer=cpu_copy(self.optimizer.state_dict()),
            observation=self.observation.clone(),visits=self.visits.clone(),rng=self.generator.get_state(),updates=self.updates,
            calibration_scale=self.calibration_scale)

    def load_state_dict(self,state):
        if state['version']!=FAMILY or state['observation'].shape!=self.observation.shape:
            raise ValueError('dense keep state mismatch')
        if state['calibration_scale']!=self.calibration_scale:raise ValueError('keep calibration scale changed')
        self.network.load_state_dict(state['network']);self.optimizer.load_state_dict(state['optimizer'])
        self.observation=state['observation'].cpu().clone();self.visits=state['visits'].cpu().clone()
        self.generator.set_state(state['rng'].cpu());self.updates=state['updates']

    def train(self,gradient,side_proxy,labels,d_ref,c_ref,render_sources):
        if self.inputs is None or self.noise is None:raise RuntimeError('keep sampling context missing')
        image=torch.empty(self.count,dtype=torch.float64);image[render_sources]=gradient[:,-1].detach().double().cpu()/d_ref
        conditional=self.table.logits.softmax(-1).detach().cpu().double()
        units=(conditional*torch.tensor(self.model.cfg.rates[1:],dtype=torch.float64)).sum(-1)
        units+=side_proxy['xyz_per_retained']+side_proxy.get('scale_per_retained',0.)
        self.optimizer.zero_grad(set_to_none=True);total=0.
        for start in range(0,self.count,self.args.keep_chunk):
            end=min(start+self.args.keep_chunk,self.count);sl=slice(start,end)
            probability,gate=smoothed_gate(self.network(self.inputs[sl].to(self.device)),self.previous[sl].to(self.device),
                                         self.noise[sl].to(self.device),self.args.keep_blend)
            loss=(image[sl].to(self.device)*gate).sum()+self.args.beta*(units[sl].to(self.device)*probability).sum()/c_ref
            if not torch.isfinite(loss):raise RuntimeError('nonfinite dense keep objective')
            loss.backward();total+=float(loss.detach())
        calibration=0.;rows=[r for r in labels if not r['validation']]
        if rows:
            ids=torch.tensor([r['source'] for r in rows]);truth=torch.tensor([r['delta_rd'] for r in rows],device=self.device,dtype=torch.float64)
            logits=self.network(self.inputs[ids].to(self.device)).double()
            target=torch.sigmoid(-truth/self.calibration_scale)
            weights=truth.abs()/(truth.abs()+self.calibration_scale)
            loss=(weights*F.binary_cross_entropy_with_logits(logits,target,reduction='none')).mean()
            (self.args.keep_calibration_weight*loss).backward();calibration=float(loss.detach())
        norm=nn.utils.clip_grad_norm_(self.network.parameters(),10.,error_if_nonfinite=True)
        self.optimizer.step();self.updates+=1
        result=dict(keep_network_updates=self.updates,dense_keep_rows=self.count,
            keep_image_feedback_rows=int((image!=0).sum()),keep_network_grad_norm=float(norm),
            keep_dense_surrogate=total,keep_calibration_loss=calibration,keep_calibration_training_groups=len(rows),
            keep_expected_ten_draw_drop=float(self.table.keep_logits.softmax(-1)[:,0].pow(10).mean()),
            keep_mean_probability=float(self.table.keep_logits.softmax(-1)[:,1].mean()),keep_calibration_records=labels)
        for name,subset in [('training',rows),('validation',[r for r in labels if r['validation']])]:
            result['keep_'+name+'_groups']=len(subset)
            if subset:
                with torch.no_grad():
                    ids=torch.tensor([r['source'] for r in subset]);prediction=self.network(self.inputs[ids].to(self.device)).double().cpu()
                truth=torch.tensor([r['delta_rd'] for r in subset],dtype=torch.float64)
                wrong=(prediction>=0)!=(truth<0)
                result['keep_'+name+'_regret']=float((truth.abs()*wrong).mean())
                result['keep_'+name+'_rd_bias']=float((-prediction*self.calibration_scale-truth).mean())
        return result
