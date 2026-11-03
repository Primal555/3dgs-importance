"""Independent positive-tier step limits and measured log-odds movement."""
import math
import torch
from .allocation_mirror import MirrorTierTable
from .structured_evidence import BoundedMirrorTable


class TierStepTable(BoundedMirrorTable):
    def __init__(self,*args,tier_max_log_odds=1.,**kwargs):
        super().__init__(*args,**kwargs)
        if tier_max_log_odds is not None and (not math.isfinite(tier_max_log_odds) or tier_max_log_odds<=0):
            raise ValueError('positive finite tier limit or explicit None required')
        self.tier_max_log_odds=tier_max_log_odds

    @torch.no_grad()
    def update(self,keep_costs,tier_ids,tier_costs,keep_lr,tier_lr,global_gain):
        def bounded(costs,lr,trust,cap):
            values=costs.double();span=values.amax(-1)-values.amin(-1)
            confidence=torch.ones_like(span) if trust is None else trust.to(span).clamp(.1,1.)
            raw=lr*global_gain*span
            if not torch.isfinite(raw).all():raise ValueError('nonfinite requested mirror step')
            scale=torch.ones_like(span) if cap is None else (cap*confidence/raw.clamp_min(1e-300)).clamp_max(1.)
            return (values-values.amin(-1,keepdim=True))*scale[:,None],raw,scale,confidence
        keep,_,ks,_=bounded(keep_costs,keep_lr,self.keep_trust,self.max_log_odds)
        tier,raw,scale,trust=bounded(tier_costs,tier_lr,self.tier_trust,self.tier_max_log_odds)
        ids=tier_ids.to(self.logits.device).long();before=self.logits[ids].clone()
        requested=before-tier_lr*global_gain*(tier.to(before)-tier.to(before).amin(-1,keepdim=True))
        requested-=requested.amax(-1,keepdim=True)
        MirrorTierTable.update(self,keep,tier_ids,tier,keep_lr,tier_lr,global_gain)
        after=self.logits[ids];movement=after-before
        applied=movement.amax(-1)-movement.amin(-1)
        def average(v):return float(v.double().mean()) if v.numel() else 0.
        quantiles=torch.quantile(applied,applied.new_tensor([.5,.9,.99])).tolist() if len(ids) else [0.,0.,0.]
        raw_quantiles=torch.quantile(raw,raw.new_tensor([.5,.9,.99])).tolist() if len(ids) else [0.,0.,0.]
        self.update_stats=dict(keep_step_limited_fraction=average(ks<1),tier_step_limited_fraction=average(scale<1),
            max_log_odds=self.max_log_odds,tier_max_log_odds=self.tier_max_log_odds,
            tier_update_mode='unbounded' if self.tier_max_log_odds is None else 'bounded',
            tier_update_rows=len(ids),tier_raw_log_odds_step_mean=average(raw),tier_raw_log_odds_step_quantiles=raw_quantiles,
            tier_applied_log_odds_step_mean=average(applied),tier_applied_log_odds_step_quantiles=quantiles,
            tier_update_scale_mean=average(scale),
            tier_actual_to_raw_step_ratio=float(applied.sum()/raw.sum()) if float(raw.sum())>0 else 1.,
            tier_candidate_trust_mean=average(trust),tier_numeric_floor_fraction=average((requested < -700).any(-1)),
            tier_probability_peak_before_mean=average(before.softmax(-1).amax(-1)),
            tier_probability_peak_after_mean=average(after.softmax(-1).amax(-1)),
            tier_probability_tv_mean=average((after.softmax(-1)-before.softmax(-1)).abs().sum(-1)*.5))
