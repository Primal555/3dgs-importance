"""Integer deployment maps and paired whole-scene RD acceptance."""
import math

import torch
from torch.nn import functional as F


class HardTierMap:
    def __init__(self, q, device='cpu'):
        self.set_map(q, device)

    def set_map(self, q, device=None):
        value = torch.as_tensor(q)
        if value.ndim != 1 or value.is_floating_point() or ((value < 0) | (value > 3)).any():
            raise ValueError('expected integer tiers [N] in 0..3')
        self.q = value.to(device=device or self.q.device, dtype=torch.long).clone()
        # The reused collector only uses these for an unused keep-branch mean.
        self.logits = F.one_hot((self.q-1).clamp_min(0), 3).double()*700.

    def sample(self, indices, snr):
        q = self.q[indices.to(self.q.device)]
        positive = q.clamp_min(1)
        return q, (q > 0).double(), F.one_hot(positive-1, 3).double(), positive

    def probabilities(self, indices, snr):
        return F.one_hot(self.q[indices.to(self.q.device)], 4).double()


class ProposalBank:
    def __init__(self, count):
        self.sums = torch.zeros(count, 3, dtype=torch.float64)
        self.visits = torch.zeros(count, dtype=torch.long)

    def add(self, ids, costs):
        ids, costs = ids.detach().cpu().long(), costs.detach().cpu().double()
        if (ids.ndim != 1 or costs.shape != (len(ids), 3) or len(ids.unique()) != len(ids)
                or ((ids < 0) | (ids >= len(self.visits))).any() or not torch.isfinite(costs).all()):
            raise ValueError('invalid candidate costs')
        self.sums[ids] += costs; self.visits[ids] += 1

    def propose(self, q, block_ids):
        q, block_ids = q.cpu().long(), block_ids.cpu().long()
        if q.shape != self.visits.shape or block_ids.shape != q.shape:
            raise ValueError('map/block shape mismatch')
        ids = (self.visits > 0).nonzero(as_tuple=True)[0]
        costs = torch.cat((torch.zeros(len(ids), 1, dtype=torch.float64),
                           self.sums[ids]/self.visits[ids, None]), -1)
        best = costs.argmin(-1)
        gain = costs.gather(1, q[ids, None]).squeeze(1)-costs.amin(-1)
        order = torch.argsort(gain, descending=True, stable=True)
        chosen, seen = [], set()
        for index in order.tolist():
            if gain[index] <= 0: break
            block = int(block_ids[ids[index]])
            if block not in seen:
                chosen.append(index); seen.add(block)
        index = torch.tensor(chosen, dtype=torch.long)
        return dict(ids=ids[index], tiers=best[index], gains=gain[index])

    def clear(self):
        self.sums.zero_(); self.visits.zero_()

    def state_dict(self):
        ids = (self.visits > 0).nonzero(as_tuple=True)[0]
        return dict(count=len(self.visits), ids=ids, sums=self.sums[ids].clone(), visits=self.visits[ids].clone())

    def load_state_dict(self, state):
        ids = state['ids'].long().cpu()
        if state['count'] != len(self.visits) or len(ids.unique()) != len(ids):
            raise ValueError('proposal bank identity mismatch')
        if state['sums'].shape != (len(ids), 3) or not torch.isfinite(state['sums']).all():
            raise ValueError('invalid saved proposal costs')
        if state['visits'].shape != ids.shape or (state['visits'] <= 0).any():
            raise ValueError('invalid saved proposal visits')
        self.clear(); self.sums[ids] = state['sums']; self.visits[ids] = state['visits']


def audit_proposals(q, proposal, baseline, measure, *, max_backtracks=2, min_gain=1e-6):
    if max_backtracks < 0 or not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError('invalid acceptance controls')
    before = float(baseline['rd_score'])
    if not math.isfinite(before): raise ValueError('nonfinite baseline RD')
    attempts, tried = [], set()
    ids, tiers = proposal['ids'], proposal['tiers']
    for backtrack in range(max_backtracks+1):
        size = math.ceil(len(ids)/(2**backtrack))
        if size == 0 or size in tried: continue
        tried.add(size)
        candidate = q.clone(); candidate[ids[:size]] = tiers[:size].to(candidate)
        measurement = measure(candidate)
        score = float(measurement['rd_score'])
        if not math.isfinite(score): raise ValueError('nonfinite candidate RD')
        accepted = score < before and before-score >= min_gain
        attempts.append(dict(changes=size, predicted_gain=float(proposal['gains'][:size].sum()),
                             rd_score=score, rd_delta=score-before, accepted=accepted))
        if accepted:
            return dict(q=candidate, measurement=measurement, accepted=True, changed=size, attempts=attempts)
    return dict(q=q.clone(), measurement=baseline, accepted=False, changed=0, attempts=attempts)
