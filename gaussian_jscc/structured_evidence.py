"""Continuous component evidence and uncertainty-limited mirror steps.

Primary history receives one resolved observation (exact replaces approximate).
Diagnostic exact/approximate histories do not feed extra samples into it.
"""
import math
import torch
from .allocation_evidence import EvidenceScheduler
from .allocation_mirror import MirrorTierTable


class StructuredEvidence(EvidenceScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = len(self.counts)
        self.mean = torch.zeros(n, 3, 2, dtype=torch.float64)
        self.second = torch.zeros_like(self.mean)
        self.keep_mean = torch.zeros_like(self.mean); self.keep_second = torch.zeros_like(self.mean)
        self.keep_weight = torch.zeros(n, 3, dtype=torch.float64)
        self.keep_observations = torch.zeros(n, 3, dtype=torch.long)
        self.keep_step = torch.zeros(n, 3, dtype=torch.long)
        self.refresh = torch.zeros(n, dtype=torch.bool)
        self.refresh_events = 0
        for branch in ('positive', 'keep'):
            for kind in ('approx', 'exact'):
                setattr(self, f'{branch}_{kind}_mean', torch.zeros_like(self.mean))
                setattr(self, f'{branch}_{kind}_count', torch.zeros(n, 3, dtype=torch.long))
        self.error = torch.zeros(n, dtype=torch.float64)
        self.keep_trust = torch.ones(n, dtype=torch.float64)
        self.tier_trust = torch.ones(n, dtype=torch.float64)

    def select_rounds(self, q, influence, probabilities, step):
        # Reuse the coverage contract, but never clear evidence on context drift.
        q, influence, probabilities = q.cpu(), influence.double().cpu(), probabilities.float().cpu()
        if probabilities.shape != (len(self.counts), 4) or not torch.isfinite(probabilities).all():
            raise ValueError('invalid selection probabilities')
        if self.context_ready:
            for ids in self.ids:
                valid = ids >= 0; safe = ids.clamp_min(0)
                drift = ((probabilities[safe]-self.context[safe]).abs().sum(-1)*.5*valid).sum(-1)/valid.sum(-1).clamp_min(1)
                changed = ids[valid & (drift > self.context_tv)[:, None]]
                self.refresh[changed] = True
                self.context[changed] = probabilities[changed]
                self.refresh_events += int((drift > self.context_tv).sum())
        else:
            self.context.copy_(probabilities); self.context_ready = True
        selected = []
        variation = (self.second-self.mean.square()).clamp_min(0).sum((1, 2)).sqrt()
        for ids in self.ids:
            safe = ids.clamp_min(0); available = (ids >= 0) & (q[safe] > 0)
            batch = []
            for r in range(self.rounds):
                if self.policy == 'round_robin' or r == 0:
                    score = -(self.counts[safe].double()*(step+1)+self.last[safe])
                else:
                    def norm(x): return x/x.amax(-1, keepdim=True).clamp_min(1e-30)
                    score = norm(influence[safe].abs())+norm(variation[safe])+self.refresh[safe].double()
                    score += (step-self.last[safe]).double()/max(step, 1)
                col = score.masked_fill(~available, -torch.inf).argmax(-1)
                active = available.any(-1); col = col.masked_fill(~active, -1); batch.append(col)
                available[torch.arange(len(ids))[active], col[active]] = False
            selected.append(batch)
        return [[batch[r] for batch in selected] for r in range(self.rounds)]

    @staticmethod
    def _combine(parts, weight): return parts[..., 0]+weight*parts[..., 1]

    def _trust(self, mean, second, weight, age, refresh, error, lambda_s):
        magnitude = self._combine(mean, lambda_s).abs().mean(-1).clamp_min(1e-12)
        std = self._combine((second-mean.square()).clamp_min(0).sqrt(), lambda_s).mean(-1)
        return .1+.9*(weight/(weight+1))/(1+std/magnitude+error/magnitude) * torch.exp2(-age/self.half_life) * torch.where(refresh, .5, 1.)

    def _diagnostic(self, branch, kind, ids, slots, values):
        mean = getattr(self, f'{branch}_{kind}_mean'); count = getattr(self, f'{branch}_{kind}_count')
        old = count[ids, slots].double()
        mean[ids, slots] = (mean[ids, slots]*old[:, None]+values)/(old+1)[:, None]
        count[ids, slots] += 1

    def accumulate_components(self, ids, parts, rate, step, lambda_s, *, approximate=None, exact=None):
        device = parts.device; ids = ids.detach().cpu().long(); parts = parts.detach().double().cpu()
        rate = rate.detach().double().cpu()
        if (parts.shape != (len(ids), 3, 2) or rate.shape != (len(ids), 3) or
                len(ids.unique()) != len(ids) or ((ids < 0) | (ids >= len(self.counts))).any() or
                step < 1 or (self.evidence_step[ids] >= step).any() or
                not torch.isfinite(parts).all() or not torch.isfinite(rate).all()):
            raise ValueError('invalid or replayed component evidence')
        centered = parts-parts.mean(1, keepdim=True)
        old = self.weight[ids]*torch.exp2(-(step-self.evidence_step[ids]).double()/self.half_life)
        new = old+1
        self.mean[ids] = (self.mean[ids]*old[:, None, None]+centered)/new[:, None, None]
        self.second[ids] = (self.second[ids]*old[:, None, None]+centered.square())/new[:, None, None]
        self.weight[ids] = new; self.observations[ids] += 1; self.evidence_step[ids] = step
        approx = parts if approximate is None else approximate.detach().double().cpu()
        mask = torch.zeros(len(ids), 3, dtype=torch.bool) if exact is None else exact.cpu()
        if approx.shape != parts.shape or mask.shape != parts.shape[:2]: raise ValueError('invalid diagnostic components')
        for slot in range(3):
            self._diagnostic('positive', 'approx', ids, torch.full_like(ids, slot), approx[:, slot])
            checked = mask[:, slot]
            self._diagnostic('positive', 'exact', ids[checked], torch.full_like(ids[checked], slot), parts[checked, slot])
        checked = mask.any(-1)
        error = self._combine(parts-approx, lambda_s).abs().amax(-1)
        self.error[ids[checked]] = .8*self.error[ids[checked]]+.2*error[checked]
        self.tier_trust[ids] = self._trust(self.mean[ids], self.second[ids], new,
            torch.zeros_like(new), self.refresh[ids], self.error[ids], lambda_s)
        self.refresh[ids] = False
        return (self._combine(self.mean[ids] if self.memory else centered, lambda_s)+rate).to(device)

    def keep_ready(self, step):
        age = (step-self.keep_step).double()
        weight = self.keep_weight*torch.exp2(-age/self.half_life)
        return ((weight >= .5) & (age <= self.half_life)).all(-1)

    def accumulate_keep(self, tiers, parts, conditional, step, lambda_s, *, approximate=None, exact=None):
        device = parts.device; parts = parts.detach().double().cpu()
        tiers = tiers.detach().cpu().long(); conditional = conditional.detach().double().cpu()
        ids = torch.arange(len(self.counts)); slots = tiers-1
        if (tiers.shape != ids.shape or parts.shape != (len(ids), 2) or
                ((tiers < 1) | (tiers > 3)).any() or (self.keep_step[ids, slots] >= step).any() or
                not torch.isfinite(parts).all()): raise ValueError('invalid or replayed keep evidence')
        old = self.keep_weight[ids, slots]*torch.exp2(-(step-self.keep_step[ids, slots]).double()/self.half_life)
        new = old+1
        self.keep_mean[ids, slots] = (self.keep_mean[ids, slots]*old[:, None]+parts)/new[:, None]
        self.keep_second[ids, slots] = (self.keep_second[ids, slots]*old[:, None]+parts.square())/new[:, None]
        self.keep_weight[ids, slots] = new; self.keep_observations[ids, slots] += 1; self.keep_step[ids, slots] = step
        approx = parts if approximate is None else approximate.detach().double().cpu()
        checked = torch.zeros(len(ids), dtype=torch.bool) if exact is None else exact.cpu()
        self._diagnostic('keep', 'approx', ids, slots, approx)
        self._diagnostic('keep', 'exact', ids[checked], slots[checked], parts[checked])
        error = self._combine(parts-approx, lambda_s).abs()
        self.error[checked] = .8*self.error[checked]+.2*error[checked]
        values = self._combine(self.keep_mean, lambda_s)
        ready = self.keep_ready(step)
        # Missing tiers are never filled with zero as measured evidence.
        result = torch.where(ready, (values*conditional).sum(-1), values[ids, slots])
        mean = self.keep_mean[ids, slots][:, None]; second = self.keep_second[ids, slots][:, None]
        self.keep_trust = self._trust(mean, second, new, torch.zeros_like(new), self.refresh, self.error, lambda_s)
        return (result if self.memory else self._combine(parts, lambda_s)).to(device)

    def diagnostics(self, q, step=None):
        step = int(self.last.max()) if step is None else step
        kept = q.cpu() > 0; weight = self.weight*torch.exp2(-(step-self.evidence_step).double()/self.half_life)
        def summary(x):
            return dict(mean=float(x.double().mean()) if x.numel() else 0.,
                zero_fraction=float((x == 0).double().mean()) if x.numel() else 0.,
                quantiles=torch.quantile(x.double(), torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)).tolist() if x.numel() else [])
        return dict(retained_candidate_evaluations_mean=float(self.counts[kept].double().mean()) if kept.any() else 0.,
            retained_effective_evidence_mean=float(weight[kept].mean()) if kept.any() else 0.,
            evidence_context_invalidations=0, evidence_refresh_events=self.refresh_events,
            positive_history=summary(weight[kept]), positive_observations=summary(self.observations[kept]),
            keep_history=summary((self.keep_weight*torch.exp2(-(step-self.keep_step).double()/self.half_life)).flatten()),
            keep_all_tiers_ready_fraction=float(self.keep_ready(step).double().mean()),
            finite_evaluated_rows=int((self.finite_counts > 0).sum()))

    @staticmethod
    def _tensor_names():
        names = ('keep_mean', 'keep_second', 'keep_weight', 'keep_observations', 'keep_step',
                 'refresh', 'error', 'keep_trust', 'tier_trust')
        names += tuple(f'{b}_{k}_{s}' for b in ('positive', 'keep') for k in ('approx', 'exact') for s in ('mean', 'count'))
        return names

    def state_dict(self):
        state = super().state_dict()
        state['structured'] = dict(version=1, refresh_events=self.refresh_events,
            tensors={name:getattr(self, name).clone() for name in self._tensor_names()})
        return state

    def load_state_dict(self, state):
        if state.get('structured', {}).get('version') != 1: raise ValueError('structured history family mismatch')
        expected = self._tensor_names()
        if set(state['structured']['tensors']) != set(expected):
            raise ValueError('incomplete structured history checkpoint')
        super().load_state_dict(state)
        for name, value in state['structured']['tensors'].items():
            current = getattr(self, name)
            if value.shape != current.shape or value.dtype != current.dtype or not torch.isfinite(value).all():
                raise ValueError('invalid structured history '+name)
            setattr(self, name, value.cpu().clone())
        self.refresh_events = state['structured']['refresh_events']


def risk_choices(ids, action_scores, q, scheduler, limit, step):
    """Independent actions: boundary/move, uncertainty/disagreement, coverage.

    Stable rotating ties avoid a permanent low-source-ID preference.
    """
    ids, scores, q = ids.cpu(), action_scores.detach().double().cpu().clone(), q.cpu()
    if not len(ids) or limit <= 0: return []
    scores[torch.arange(len(ids)), q[ids]-1] = torch.inf
    boundary = scores.abs().amin(-1)
    move = torch.nan_to_num(scores, posinf=0.).amax(-1)-torch.nan_to_num(scores, posinf=0.).amin(-1)
    boundary = boundary/(1+move)
    variance = (scheduler.second[ids]-scheduler.mean[ids].square()).clamp_min(0).sum((1, 2)).sqrt()
    risk = variance+scheduler.error[ids]+scheduler.refresh[ids].double()
    action_counts = scheduler.positive_exact_count[ids].clone()
    action_counts[torch.arange(len(ids)), q[ids]-1] = torch.iinfo(torch.long).max
    least_action_checks = action_counts.amin(-1)
    rotation = torch.roll(torch.arange(len(ids)), shifts=-(step % len(ids)))
    orders = [rotation[torch.argsort(boundary[rotation], stable=True)].tolist(),
              rotation[torch.argsort(risk[rotation], descending=True, stable=True)].tolist(),
              rotation[torch.argsort(least_action_checks[rotation], stable=True)].tolist()]
    used, result = set(), []
    for slot in range(min(limit, len(ids))):
        group = (slot+step-1) % 3
        index = next(i for i in orders[group] if i not in used); used.add(index)
        if group == 2:
            alternatives = [t for t in (1, 2, 3) if t != int(q[ids[index]])]
            minimum = min(int(action_counts[index, t-1]) for t in alternatives)
            tied = [t for t in alternatives if int(action_counts[index, t-1]) == minimum]
            tier = tied[(step+int(ids[index])) % len(tied)]
        else: tier = int(scores[index].abs().argmin())+1
        result.append((index, tier, ('boundary', 'uncertainty', 'coverage')[group]))
    return result


class BoundedMirrorTable(MirrorTierTable):
    def __init__(self, *args, max_log_odds=.25, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(max_log_odds) or max_log_odds <= 0: raise ValueError('invalid odds bound')
        self.max_log_odds = max_log_odds
        self.keep_trust = None; self.tier_trust = None; self.update_stats = {}

    def update(self, keep_costs, tier_ids, tier_costs, keep_lr, tier_lr, global_gain):
        fractions = []
        def bounded(costs, lr, trust):
            values = costs.double(); span = values.amax(-1)-values.amin(-1)
            confidence = torch.ones_like(span) if trust is None else trust.to(span).clamp(.1, 1.)
            scale = (self.max_log_odds*confidence/(lr*global_gain*span).clamp_min(1e-300)).clamp_max(1.)
            fractions.append(float((scale < 1).double().mean()) if scale.numel() else 0.)
            return (values-values.amin(-1, keepdim=True))*scale[:, None]
        keep = bounded(keep_costs, keep_lr, self.keep_trust)
        tier = bounded(tier_costs, tier_lr, self.tier_trust)
        super().update(keep, tier_ids, tier, keep_lr, tier_lr, global_gain)
        self.update_stats = dict(keep_step_limited_fraction=fractions[0], tier_step_limited_fraction=fractions[1],
            max_log_odds=self.max_log_odds)
