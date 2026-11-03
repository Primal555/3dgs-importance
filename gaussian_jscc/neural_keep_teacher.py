"""Complete q0-q3 paired interventions for the learned keep head."""
import math
import time
import torch
from .allocation_benefit_features import keep_rates
from .budgeted_allocation_kernel import raw_batches,set_rng
from .neural_dense_keep import mixed_keep_label
from .optimization import preserved_rng
from .pixel_benefit_window import decode_groups


@torch.no_grad()
def collect_keep_labels(policy,task,cache,scene,q,id_batches,side_proxy,budget,d_ref,c_ref):
    start=time.perf_counter();limit=min(policy.count,budget//3)
    labels=[];available=torch.ones(policy.count,dtype=torch.bool);device=scene.device
    for slot in range(limit):
        order=torch.randperm(policy.count,generator=policy.generator)
        desired=(slot+side_proxy['step'])%2==0
        pool=order[available[order]&((q[order]>0)==desired)]
        if not len(pool):pool=order[available[order]]
        if (slot+side_proxy['step'])%3==1:
            score=(policy.table.keep_logits[pool.to(policy.device)].softmax(-1)[:,1]-.5).abs().cpu()
        else:score=policy.visits[pool]
        source=int(pool[score.argmin()]);available[source]=False
        batch,row,col,_=policy.positions[source].tolist()
        f=policy.scene.groups[batch].to(device);ids=id_batches[batch].to(device);valid=ids>=0
        tiers,_,y,actual,rng=cache[batch]
        with preserved_rng(device):
            set_rng(rng,device)
            noise=torch.randn((*tiers.shape,2*policy.model.cfg.rates[-1]),device=device,dtype=f.dtype)
            noise*=math.sqrt(10**(-float(policy.args.snr)/10)/2)
        recovered,states=decode_groups(policy.model,f,tiers,torch.tensor([row],device=device),
            torch.tensor([col],device=device),actual,y,noise,policy.args.snr)
        raw=raw_batches(recovered[0],policy.scene.geometry,policy.model)
        states=states[0];old=int(tiers[row,col]);deltas=scene.new_zeros(4,2,dtype=torch.float64)
        first=int(policy.positions[source,3])-int(valid[row,:col].sum());count=int(valid[row].sum())
        for target in range(4):
            if target==old:continue
            changed=scene.detach().clone()
            changed[first:first+count]=torch.cat((raw[target,valid[row]],(states[target,valid[row]]>0).to(raw)[:,None]),-1)
            deltas[target]=task.delta_components(changed)
        rates=keep_rates(tiers,valid,policy.model.cfg.rates,xyz=side_proxy['xyz_per_retained'],
                         bits=side_proxy.get('bits_per_use',2.))[row,col].double()
        conditional=policy.table.logits[source].softmax(-1).detach()
        delta=mixed_keep_label(deltas,conditional,rates,task.distortion.lambda_s,d_ref,policy.args.beta,c_ref)
        labels.append(dict(source=source,step=side_proxy['step'],delta_rd=float(delta),
                           validation=bool(policy.validation[source]),conditional=conditional.cpu().tolist(),
                           distortion_deltas=deltas.cpu().tolist(),marginal_channel_uses=rates.cpu().tolist(),
                           reference_tier=old,cost_model='payload_scale_and_average_xyz_proxy'))
        policy.visits[source]+=1
    return labels,dict(neural_keep_label_groups=len(labels),neural_keep_render_calls=3*len(labels),
                       neural_keep_teacher_seconds=time.perf_counter()-start)
