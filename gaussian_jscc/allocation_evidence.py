"""Coverage/demand selection and fresh, centered per-point action evidence.

Selection scores prioritize compute only. They never rescale the RD objective.
Random masks are samples; context invalidation follows distribution drift.
"""
import math
import torch
from .allocation_mirror import BlockVisitScheduler


class EvidenceScheduler(BlockVisitScheduler):
    def __init__(self, id_batches, *, rounds=2, policy='coverage_demand', memory=True,
                 half_life=1000., context_tv=.1, finite_samples=8):
        super().__init__(id_batches)
        if (not 1 <= rounds <= 8 or policy not in ('coverage_demand', 'round_robin')
                or not math.isfinite(half_life) or half_life <= 0
                or not math.isfinite(context_tv) or not 0 < context_tv <= 1
                or finite_samples < 0):
            raise ValueError('invalid evidence allocation settings')
        self.rounds, self.policy, self.memory = rounds, policy, memory
        self.half_life, self.context_tv, self.finite_samples = half_life, context_tv, finite_samples
        n = len(self.counts)
        self.mean = torch.zeros(n, 3, dtype=torch.float64)
        self.second = torch.zeros_like(self.mean)
        self.weight = torch.zeros(n, dtype=torch.float64)
        self.observations = torch.zeros(n, dtype=torch.long)
        self.evidence_step = torch.zeros(n, dtype=torch.long)
        self.finite_counts = torch.zeros(n, dtype=torch.long)
        self.context = torch.zeros(n, 4, dtype=torch.float32)
        self.context_ready = False
        self.invalidations = 0

    def select_rounds(self, q, influence, probabilities, step):
        q, influence = q.detach().cpu(), influence.detach().double().cpu()
        probabilities = probabilities.detach().float().cpu()
        if (q.shape != self.counts.shape or influence.shape != q.shape
                or probabilities.shape != (len(q), 4) or step < 1
                or not torch.isfinite(influence).all() or not torch.isfinite(probabilities).all()):
            raise ValueError('invalid candidate selection data')
        uncertainty = 1 - probabilities[:, 1:].div(probabilities[:, 1:].sum(-1, keepdim=True).clamp_min(1e-30)).amax(-1)
        variance = (self.second-self.mean.square()).clamp_min(0).mean(-1).sqrt()
        selected = []
        for ids in self.ids:
            valid = ids >= 0; safe = ids.clamp_min(0)
            if self.context_ready:
                drift = ((probabilities[safe]-self.context[safe]).abs().sum(-1)*.5*valid).sum(-1)/valid.sum(-1).clamp_min(1)
                changed = drift > self.context_tv
                reset = ids[valid & changed[:, None]]
                if len(reset):
                    self.mean[reset] = 0; self.second[reset] = 0; self.weight[reset] = 0
                    variance[reset] = 0
                    self.context[reset] = probabilities[reset]
                    self.invalidations += int(changed.sum())
            available = valid & (q[safe] > 0)
            batch_rounds = []
            for r in range(self.rounds):
                if self.policy == 'round_robin' or r == 0:
                    # Strict least-visits priority; then oldest evidence, then source order.
                    priority = self.counts[safe].double()*(step+1)+self.last[safe].double()
                    score = -priority
                else:
                    impact = influence[safe].abs()
                    impact = impact/impact.amax(-1, keepdim=True).clamp_min(1e-30)
                    variation = variance[safe]/variance[safe].amax(-1, keepdim=True).clamp_min(1e-30)
                    age = (step-self.last[safe]).double()/step
                    score = impact*(.25+uncertainty[safe])*(1+variation)*(1+age)
                score = score.masked_fill(~available, -torch.inf)
                col = score.argmax(-1); active = available.any(-1)
                col = col.masked_fill(~active, -1); batch_rounds.append(col)
                rows = torch.arange(len(ids))[active]
                available[rows, col[active]] = False
            selected.append(batch_rounds)
        if not self.context_ready:
            self.context.copy_(probabilities); self.context_ready = True
        return [[batch[r] for batch in selected] for r in range(self.rounds)]

    def selected_ids(self, rounds):
        result = []
        for columns in rounds:
            for ids, col in zip(self.ids, columns):
                active = col >= 0
                result.append(ids[torch.arange(len(ids))[active], col[active]])
        return torch.cat(result) if result else torch.empty(0, dtype=torch.long)

    def accumulate(self, ids, image, rate, step):
        device = image.device
        ids = ids.detach().cpu().long()
        image, rate = image.detach().double().cpu(), rate.detach().double().cpu()
        if (image.shape != (len(ids), 3) or rate.shape != image.shape or len(ids.unique()) != len(ids)
                or ((ids < 0) | (ids >= len(self.counts))).any() or step < 1
                or not torch.isfinite(image).all() or not torch.isfinite(rate).all()
                or (self.evidence_step[ids] >= step).any()):
            raise ValueError('invalid or replayed action evidence')
        centered = image-image.mean(-1, keepdim=True)
        decay = torch.exp2(-(step-self.evidence_step[ids]).double()/self.half_life)
        old = self.weight[ids]*decay
        new = old+1
        self.mean[ids] = (self.mean[ids]*old[:, None]+centered)/new[:, None]
        self.second[ids] = (self.second[ids]*old[:, None]+centered.square())/new[:, None]
        self.weight[ids] = new; self.evidence_step[ids] = step; self.observations[ids] += 1
        return ((self.mean[ids] if self.memory else centered)+rate).to(device)

    def diagnostics(self, q, step=None):
        kept = q.detach().cpu() > 0
        step = int(self.last.max()) if step is None else step
        weight = self.weight*torch.exp2(-(step-self.evidence_step).clamp_min(0).double()/self.half_life)
        def average(x): return float(x[kept].double().mean()) if kept.any() else 0.
        return dict(retained_candidate_evaluations_mean=average(self.counts),
                    retained_effective_evidence_mean=average(weight),
                    evidence_context_invalidations=self.invalidations,
                    finite_evaluated_rows=int((self.finite_counts > 0).sum()))

    def state_dict(self):
        return dict(super().state_dict(), evidence=dict(
            settings=[self.rounds, self.policy, self.memory, self.half_life, self.context_tv, self.finite_samples],
            mean=self.mean.clone(), second=self.second.clone(), weight=self.weight.clone(),
            observations=self.observations.clone(), evidence_step=self.evidence_step.clone(),
            finite_counts=self.finite_counts.clone(), context=self.context.clone(),
            context_ready=self.context_ready, invalidations=self.invalidations))

    def load_state_dict(self, state):
        data = state['evidence']
        if data['settings'] != self.state_dict()['evidence']['settings']:
            raise ValueError('evidence resume settings mismatch')
        super().load_state_dict(state)
        for name in ('mean', 'second', 'weight', 'observations', 'evidence_step', 'finite_counts', 'context'):
            tensor = data[name].cpu()
            if tensor.shape != getattr(self, name).shape or not torch.isfinite(tensor).all():
                raise ValueError('invalid evidence checkpoint tensor: '+name)
            setattr(self, name, tensor.clone())
        self.context_ready, self.invalidations = data['context_ready'], data['invalidations']


def summarize_evidence(out, selected, seed):
    """Small numerical artifacts; seed comparison does not claim image robustness."""
    import csv
    import json
    from pathlib import Path
    from .allocation import deployment_tiers
    from .allocation_mirror import MirrorTierTable
    from .modular_experiment import json_write
    out = Path(out)
    table = MirrorTierTable(len(selected['table']['logits']))
    table.load_state_dict(selected['table'])
    p = table.probabilities(torch.arange(len(table.logits)),10.)
    q = selected['latest_q'].cpu(); kept = q > 0
    confidence = table.logits.softmax(-1).amax(-1)
    alternatives = []
    for other in (seed+1,seed+2):
        changed = deployment_tiers(p,seed=other,draws=10)
        common = kept & (changed > 0)
        alternatives.append(dict(seed=other,tier_counts=torch.bincount(changed,minlength=4).tolist(),
            all_tier_change_fraction=float((changed!=q).double().mean()),
            positive_change_fraction=float((changed[common]!=q[common]).double().mean()) if common.any() else None,
            note='same probabilities; no additional rendering; saved deployment remains fixed'))
    observations = selected['blocks']['evidence']['observations']
    json_write(out/'comparison/evidence.json',dict(
        retained_evaluations_mean=float(observations[kept].double().mean()) if kept.any() else 0.,
        retained_max_positive_probability_median=float(confidence[kept].median()) if kept.any() else None,
        retained_confidence_below_04=float((confidence[kept]<.4).double().mean()) if kept.any() else None,
        deployment_seed_sensitivity=alternatives))
    rows = [json.loads(s) for s in (out/'metrics/validation.jsonl').read_text(encoding='utf-8').splitlines()]
    columns=('event','step','total_channel_uses','source_psnr','source_ssim','source_lpips',
             'photo_psnr','photo_ssim','photo_lpips','rd_score')
    with (out/'comparison/uniform_controls.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({k:r.get(k) for k in columns} for r in rows
                        if r['event'] in ('initial','selected','latest','uniform_q2','uniform_q3'))
