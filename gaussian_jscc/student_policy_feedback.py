"""Current hard-policy feedback and inclusion-corrected shared-network VJP."""
import torch
from torch.nn import functional as F
from .allocation_mirror import BlockVisitScheduler
from .student_policy_model import hard_tiers


class CurrentPolicy:
    """Read-only detached logits adapt the existing frozen-codec kernel."""
    def __init__(self, keep_logits, tier_logits):
        self.keep_logits = keep_logits.detach().double()
        self.logits = tier_logits.detach().double()
        self.keep_trust = self.tier_trust = None

    def sample(self, ids, snr):
        keep, tier = self.keep_logits[ids], self.logits[ids]
        positive = tier.argmax(-1)+1
        q = hard_tiers(keep.reshape(-1, 2), tier.reshape(-1, 3)).reshape(ids.shape)
        return q, (q > 0).to(keep), F.one_hot(positive-1, 3).to(keep), positive

    def probabilities(self, ids, snr):
        keep = self.keep_logits[ids].softmax(-1); tier = self.logits[ids].softmax(-1)
        return torch.cat((keep[..., :1], keep[..., 1:]*tier), -1)


class UniformFeedback(BlockVisitScheduler):
    """Current observations only; action-check coverage counts remain resumable."""
    def __init__(self, ids, rounds=2, finite_samples=32, seed=42):
        super().__init__(ids)
        if min(rounds, 1) < 1 or finite_samples < 0: raise ValueError('invalid feedback budgets')
        self.rounds, self.finite_samples = rounds, finite_samples
        self.generator = torch.Generator().manual_seed(seed)
        count = len(self.counts)
        self.inverse_inclusion = torch.zeros(count, dtype=torch.float64)
        self.finite_counts = torch.zeros(count, dtype=torch.long)
        self.positive_exact_count = torch.zeros(count, 3, dtype=torch.long)
        self.keep_exact_count = torch.zeros_like(self.positive_exact_count)
        self.keep_trust = torch.ones(count, dtype=torch.float64)
        self.tier_trust = torch.ones_like(self.keep_trust)

    def select_rounds(self, q, influence, probabilities, step):
        q = q.cpu(); self.inverse_inclusion.zero_(); selected = []
        for ids in self.ids:
            available = (ids >= 0) & (q[ids.clamp_min(0)] > 0)
            score = torch.rand(ids.shape, generator=self.generator).masked_fill(~available, torch.inf)
            k = min(self.rounds, ids.shape[1])
            columns = score.topk(k, dim=1, largest=False, sorted=True).indices
            columns = columns.masked_fill(~available.gather(1, columns), -1)
            rounds = [columns[:, r] for r in range(k)]
            rounds += [torch.full((len(ids),), -1, dtype=torch.long) for _ in range(self.rounds-k)]
            m = available.sum(-1); used = m.clamp_max(self.rounds)
            weights = m.double()/used.clamp_min(1)
            self.inverse_inclusion[ids[available]] = weights[:, None].expand_as(ids)[available]
            selected.append(rounds)
        return [[batch[r] for batch in selected] for r in range(self.rounds)]

    @staticmethod
    def _combine(parts, weight): return parts[..., 0]+weight*parts[..., 1]

    def positive_risk(self, ids): return torch.zeros(len(ids), dtype=torch.float64)
    def keep_risk(self): return torch.zeros(len(self.counts), dtype=torch.float64)

    def accumulate_components(self, ids, parts, rates, step, lambda_s, *, approximate=None, exact=None):
        if exact is not None: self.positive_exact_count[ids] += exact.cpu().long()
        return self._combine(parts-parts.mean(1, keepdim=True), lambda_s)+rates

    def accumulate_keep(self, tiers, parts, conditional, step, lambda_s, *, approximate=None, exact=None):
        if exact is not None:
            ids = exact.cpu().nonzero().flatten()
            self.keep_exact_count[ids, tiers[ids]-1] += 1
        return self._combine(parts, lambda_s)

    def diagnostics(self, q, step=None):
        return dict(feedback_history='current_context_only',
                    retained_candidate_evaluations_mean=float(self.counts[q > 0].double().mean()) if (q > 0).any() else 0.)

    def state_dict(self):
        return dict(super().state_dict(), uniform=dict(rounds=self.rounds, finite_samples=self.finite_samples,
            rng=self.generator.get_state(), finite_counts=self.finite_counts.clone(),
            positive_exact_count=self.positive_exact_count.clone(), keep_exact_count=self.keep_exact_count.clone()))

    def load_state_dict(self, state):
        super().load_state_dict(state); saved = state['uniform']
        if (self.rounds, self.finite_samples) != (saved['rounds'], saved['finite_samples']):
            raise ValueError('feedback budgets changed')
        self.generator.set_state(saved['rng'].cpu())
        for name in ('finite_counts', 'positive_exact_count', 'keep_exact_count'):
            old = getattr(self, name)
            if saved[name].shape != old.shape: raise ValueError('feedback state shape mismatch')
            old.copy_(saved[name])


def policy_surrogate(keep_logits, tier_logits, keep_costs, tier_ids, tier_costs, inverse_inclusion):
    ids = tier_ids.to(tier_logits.device)
    signal = (keep_costs[:, 1]-keep_costs[:, 0]).detach().to(keep_logits)
    keep = (keep_logits.softmax(-1)[:, 1]*signal).sum()
    if len(ids):
        weights = inverse_inclusion.detach().to(tier_logits)
        if weights.shape != ids.shape or (weights < 1).any(): raise ValueError('invalid inclusion weights')
        tier = (tier_logits[ids].softmax(-1)*tier_costs.detach().to(tier_logits)*weights[:, None]).sum()
    else: tier = tier_logits.sum()*0
    return keep+tier, dict(keep_surrogate=float(keep.detach()), tier_surrogate=float(tier.detach()),
        positive_sample_count=len(ids), inverse_inclusion_mean=float(inverse_inclusion.double().mean()) if len(ids) else 0.,
        gradient_scope='dense keep + inverse-inclusion positive; no N_source mirror gain')


def actual_tier_keep_cost(result, adapter, rates, beta, c_ref):
    """Match the gate's current hard conditional tier, charging payload once."""
    costs = result['keep_costs'].detach().clone()
    rates = costs.new_tensor(rates[1:])
    expected = (adapter.logits.softmax(-1).to(costs)*rates).sum(-1)
    actual = rates[adapter.logits.argmax(-1)]
    costs[:, 1] += beta*(actual-expected)/c_ref
    return costs
