"""Independent exact residuals and view-limited uncertainty of mean RD costs."""
import torch
from .allocation_correction import ActionCorrections
from .structured_evidence import StructuredEvidence


class CalibratedEvidence(StructuredEvidence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = len(self.counts)
        self.positive_corrections = ActionCorrections(9, self.half_life, self.context_tv)
        self.keep_corrections = ActionCorrections(6, self.half_life, self.context_tv)
        self.positive_weight2 = torch.zeros(n, dtype=torch.float64)
        self.keep_weight2 = torch.zeros(n, 3, dtype=torch.float64)
        self.positive_effective_samples = torch.zeros(n, dtype=torch.float64)
        self.keep_effective_samples = torch.zeros(n, 3, dtype=torch.float64)
        self.positive_error = torch.zeros(n, dtype=torch.float64)
        self.keep_error = torch.zeros(n, dtype=torch.float64)
        self.action_context = torch.zeros(n, 4, dtype=torch.float32)
        self.current_q = torch.zeros(n, dtype=torch.long)
        self.view_count = self.window_size = None
        self.current_views = ()
        self.positive_reused = self.keep_reused = 0

    def configure_views(self, count, window):
        if not 1 <= window <= count: raise ValueError('invalid calibrated training view pool')
        if self.view_count is not None: raise ValueError('calibrated views already configured')
        self.view_count, self.window_size = count, window
        n, words = len(self.counts), (count+62)//63
        self.positive_view_bits = torch.zeros(n, words, dtype=torch.long)
        self.keep_view_bits = torch.zeros(n, 3, words, dtype=torch.long)
        self.positive_view_count = torch.zeros(n, dtype=torch.long)
        self.keep_view_count = torch.zeros(n, 3, dtype=torch.long)
        self.positive_corrections.configure_views(count,window)
        self.keep_corrections.configure_views(count,window)

    def set_view_window(self, indices):
        indices = tuple(int(i) for i in indices)
        if (self.view_count is None or len(indices) != self.window_size or len(set(indices)) != len(indices)
                or min(indices) < 0 or max(indices) >= self.view_count):
            raise ValueError('invalid calibrated training view window')
        self.current_views = indices

    def select_rounds(self, q, influence, probabilities, step):
        if not self.current_views: raise ValueError('set calibrated training views before allocation')
        result = super().select_rounds(q, influence, probabilities, step)
        self.current_q = q.cpu().clone()
        for ids in self.ids:
            valid = ids >= 0; safe = ids.clamp_min(0)
            hist = torch.stack([((q.cpu()[safe] == tier)&valid).sum(-1) for tier in range(4)], -1)
            hist = hist.float()/valid.sum(-1).clamp_min(1)[:, None]
            self.action_context[ids[valid]] = hist[:, None, :].expand(-1, ids.shape[1], -1)[valid]
        return result

    def _view_ess(self, branch, ids, slots, decay, weight):
        index = ids if slots is None else (ids, slots)
        w2 = getattr(self, branch+'_weight2')
        w2[index] = w2[index]*decay.square()+1
        bits = getattr(self, branch+'_view_bits'); seen = getattr(self, branch+'_view_count')
        for view in self.current_views:
            word, shift = divmod(view, 63); mask = 1 << shift
            target = (ids, word) if slots is None else (ids, slots, word)
            old = bits[target]
            seen[index] += ((old & mask) == 0).long(); bits[target] = old | mask
        ess = torch.minimum(weight.square()/w2[index].clamp_min(1e-30), seen[index].double()/self.window_size)
        getattr(self, branch+'_effective_samples')[index] = ess
        return ess.clamp_min(1)

    @staticmethod
    def _confidence(costs, stderr, ess, refresh):
        # Include the deterministic rate gap when judging the signal-to-error ratio.
        best = costs.argmin(-1); worst = costs.argmax(-1); rows = torch.arange(len(costs))
        signal = costs[rows, worst]-costs[rows, best]
        error = stderr[rows, best]+stderr[rows, worst]
        ratio = signal/(signal+error).clamp_min(1e-30)
        return .1+.9*(ess/(ess+1))*ratio*torch.where(refresh, .5, 1.)

    def _positive_keys(self, ids):
        return ids[:, None]*9+(self.current_q[ids, None]-1)*3+torch.arange(3)

    def _keep_keys(self, ids, slots):
        return ids*6+(self.current_q[ids] > 0).long()*3+slots

    def preview_positive(self, ids, parts, step):
        ids = ids.cpu(); output = parts.detach().double().cpu().clone()
        pos, mean, _ = self.positive_corrections.query(self._positive_keys(ids), self.action_context, step)
        self.positive_reused = len(pos)
        output.reshape(-1, 2)[pos] += mean
        return output

    def preview_keep(self, tiers, parts, step):
        ids = torch.arange(len(self.counts)); output = parts.detach().double().cpu().clone()
        pos, mean, _ = self.keep_corrections.query(self._keep_keys(ids, tiers.cpu()-1), self.action_context, step)
        self.keep_reused = len(pos)
        output[pos] += mean
        return output

    def accumulate_components(self, ids, parts, rate, step, lambda_s, *, approximate=None, exact=None):
        device = parts.device; ids = ids.cpu().long(); parts = parts.detach().double().cpu()
        approx = parts if approximate is None else approximate.detach().double().cpu()
        mask = torch.zeros(len(ids),3,dtype=torch.bool) if exact is None else exact.cpu()
        keys = self._positive_keys(ids)
        previous = self.preview_positive(ids, approx, step)
        chosen = mask.nonzero()
        if len(chosen):
            row, slot = chosen.unbind(-1)
            self.positive_corrections.update(keys[row,slot], (parts-approx)[row,slot], self.action_context, step,self.current_views)
            for tier in range(3):
                selected = mask[:,tier]
                self._diagnostic('positive','exact',ids[selected],torch.full_like(ids[selected],tier),parts[selected,tier])
            error = self._combine(parts-previous,lambda_s).abs().masked_fill(~mask,0).amax(-1)
            checked = mask.any(-1); self.positive_error[ids[checked]] = .8*self.positive_error[ids[checked]]+.2*error[checked]
        decay = torch.exp2(-(step-self.evidence_step[ids]).double()/self.half_life)
        refresh = self.refresh[ids].clone()
        # Primary history contains proxies only; finite residuals have their own weights.
        super().accumulate_components(ids,approx,rate,step,lambda_s)
        ess = self._view_ess('positive',ids,None,decay,self.weight[ids])
        values = self._combine(self.mean[ids] if self.memory else approx-approx.mean(1,keepdim=True),lambda_s)
        variance = (self.second[ids]-self.mean[ids].square()).clamp_min(0)
        stderr = self._combine(variance.sqrt(),lambda_s)/ess.sqrt()[:,None]
        pos, residual, error = self.positive_corrections.query(keys,self.action_context,step)
        values.reshape(-1)[pos] += self._combine(residual,lambda_s)
        stderr.reshape(-1)[pos] += self._combine(error,lambda_s)
        # The current paired finite intervention is authoritative for its action difference.
        if len(chosen):
            row, slot = chosen.unbind(-1); old = self.current_q[ids[row]]-1
            values[row,slot] = values[row,old]+self._combine(parts[row,slot],lambda_s)
        values += rate.detach().double().cpu()
        self.tier_trust[ids] = self._confidence(values,stderr,ess,refresh)
        self.error = torch.maximum(self.positive_error,self.keep_error)
        return values.to(device)

    def accumulate_keep(self, tiers, parts, conditional, step, lambda_s, *, approximate=None, exact=None):
        device = parts.device; parts = parts.detach().double().cpu(); tiers = tiers.cpu().long()
        conditional = conditional.detach().double().cpu(); ids = torch.arange(len(self.counts)); slots = tiers-1
        approx = parts if approximate is None else approximate.detach().double().cpu()
        mask = torch.zeros(len(ids),dtype=torch.bool) if exact is None else exact.cpu()
        previous = self.preview_keep(tiers,approx,step)
        if mask.any():
            self.keep_corrections.update(self._keep_keys(ids[mask],slots[mask]),(parts-approx)[mask],self.action_context,step,self.current_views)
            self._diagnostic('keep','exact',ids[mask],slots[mask],parts[mask])
            error = self._combine(parts-previous,lambda_s).abs()
            self.keep_error[mask] = .8*self.keep_error[mask]+.2*error[mask]
        decay = torch.exp2(-(step-self.keep_step[ids,slots]).double()/self.half_life)
        super().accumulate_keep(tiers,approx,conditional,step,lambda_s)
        self._view_ess('keep',ids,slots,decay,self.keep_weight[ids,slots])
        values = self._combine(self.keep_mean,lambda_s)
        variance = (self.keep_second-self.keep_mean.square()).clamp_min(0)
        stderr = self._combine(variance.sqrt(),lambda_s)/self.keep_effective_samples.clamp_min(1).sqrt()
        keys = ids[:,None]*6+(self.current_q>0).long()[:,None]*3+torch.arange(3)
        pos,residual,error = self.keep_corrections.query(keys,self.action_context,step)
        values.reshape(-1)[pos] += self._combine(residual,lambda_s)
        stderr.reshape(-1)[pos] += self._combine(error,lambda_s)
        if mask.any(): values[ids[mask],slots[mask]] = self._combine(parts[mask],lambda_s)
        ready = self.keep_ready(step)
        if self.memory:
            result = torch.where(ready,(values*conditional).sum(-1),values[ids,slots])
        else:
            result = self._combine(self.preview_keep(tiers,approx,step),lambda_s)
            result[mask] = self._combine(parts[mask],lambda_s)
        use_mixture = ready & self.memory
        se = torch.where(use_mixture,(stderr*conditional).sum(-1),stderr[ids,slots])
        # All slots contributing to the actual mixture constrain its support.
        support = 1/(conditional/self.keep_effective_samples.clamp_min(1)).sum(-1)
        ess = torch.where(use_mixture,support,self.keep_effective_samples[ids,slots].clamp_min(1))
        self.keep_standard_error, self.keep_support = se,ess
        self.error = torch.maximum(self.positive_error,self.keep_error)
        return result.to(device)

    def calibrate_keep_update(self, costs):
        self.keep_trust = self._confidence(costs.detach().double().cpu(),
            torch.stack((self.keep_standard_error*0,self.keep_standard_error),-1), self.keep_support,self.refresh)

    def diagnostics(self, q, step=None):
        result = super().diagnostics(q,step)
        kept = q.cpu()>0
        result.update(positive_mean_effective_samples=float(self.positive_effective_samples[kept].mean()) if kept.any() else 0.,
            keep_mean_effective_samples=float(self.keep_effective_samples.mean()),
            positive_mean_trust=float(self.tier_trust[kept].mean()) if kept.any() else 0.,
            keep_mean_trust=float(self.keep_trust.mean()),
            positive_correction_reused_actions=self.positive_reused,keep_correction_reused_actions=self.keep_reused,
            positive_correction_actions=len(self.positive_corrections.keys),keep_correction_actions=len(self.keep_corrections.keys))
        return result

    def state_dict(self):
        state = super().state_dict()
        names = ('positive_weight2','keep_weight2','positive_effective_samples','keep_effective_samples',
            'positive_error','keep_error','positive_view_bits','keep_view_bits','positive_view_count','keep_view_count')
        state['calibrated'] = dict(version=2, views=[self.view_count,self.window_size],
            tensors={k:getattr(self,k).clone() for k in names},
            positive=self.positive_corrections.state_dict(),keep=self.keep_corrections.state_dict())
        return state

    def load_state_dict(self,state):
        data = state.get('calibrated',{})
        if data.get('version') != 2 or data.get('views') != [self.view_count,self.window_size]:
            raise ValueError('calibrated evidence history mismatch; start a new allocation run')
        expected = {'positive_weight2','keep_weight2','positive_effective_samples','keep_effective_samples',
            'positive_error','keep_error','positive_view_bits','keep_view_bits','positive_view_count','keep_view_count'}
        if set(data.get('tensors',{})) != expected: raise ValueError('incomplete calibrated evidence history')
        super().load_state_dict(state)
        for name,value in data['tensors'].items():
            old = getattr(self,name)
            if value.shape != old.shape or value.dtype != old.dtype or not torch.isfinite(value).all() or (value<0).any():
                raise ValueError('invalid calibrated statistic '+name)
            setattr(self,name,value.cpu().clone())
        self.positive_corrections.load_state_dict(data['positive'],len(self.counts))
        self.keep_corrections.load_state_dict(data['keep'],len(self.counts))
