"""Observed action gaps and full-state teachers share one online learner."""
import torch
from torch.nn import functional as F
from .calibrated_spatial_benefit import CalibratedSpatialLearner
from .spatial_benefit import pack_inputs


def gap_losses(prediction,groups,scale,decision_weight):
    rows=torch.arange(len(groups),device=prediction.device)
    old=torch.tensor([g['old_tier']-1 for g in groups],device=prediction.device)
    new=torch.tensor([g['new_tier']-1 for g in groups],device=prediction.device)
    estimate=prediction[rows,new]-prediction[rows,old]
    truth=torch.stack([g['delta_values'] for g in groups]).to(estimate)
    scale=torch.as_tensor(scale,device=estimate.device,dtype=estimate.dtype).clamp_min(1e-8)
    regression=((estimate-truth)/scale).square().mean(-1)
    rates=estimate.new_tensor([g['rate_delta'] for g in groups])
    p=(estimate.sum(-1)+rates)/scale;t=(truth.sum(-1)+rates)/scale
    target=torch.sigmoid(-t)
    kl=(F.binary_cross_entropy_with_logits(-p,target,reduction='none')-
        F.binary_cross_entropy_with_logits(-t,target,reduction='none')).clamp_min(0.)
    decision=kl*t.abs()
    regret=torch.where(p*t<0,(truth.sum(-1)+rates).abs(),0.)
    zero=regression*0
    return dict(absolute=regression,common=zero,pair=zero,decision=decision,
                total=regression+decision_weight*decision,regret=regret.detach())


class CorrectionLearner(CalibratedSpatialLearner):
    def _batch_losses(self,groups):
        full=[i for i,g in enumerate(groups) if g.get('supervision')!='action_gap']
        gaps=[i for i,g in enumerate(groups) if g.get('supervision')=='action_gap']
        if not gaps:return super()._batch_losses(groups)
        batches=[]
        if full:batches.append((full,super()._batch_losses([groups[i] for i in full])))
        selected=[groups[i] for i in gaps]
        inputs=[v['inputs'] for g in selected for v in g['views']]
        sizes=[len(g['views']) for g in selected]
        prediction=self.network(*pack_inputs(inputs,self.device)).double()
        prediction=torch.stack([v.mean(0) for v in prediction.split(sizes)])
        batches.append((gaps,gap_losses(prediction,selected,self.scales['rd'],self.decision_weight)))
        return {key:sum(value[key].new_zeros(len(groups)).index_copy(0,
                    torch.tensor(indices,device=self.device),value[key])
                    for indices,value in batches) for key in batches[0][1]}
