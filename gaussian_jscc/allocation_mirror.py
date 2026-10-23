"""Action-cost mirror descent and exact-resume point/view coverage."""
import math
import torch
from .allocation import GaussianTierMask


class MirrorTierTable(GaussianTierMask):
    def __init__(self, count, keep_probability=.99):
        super().__init__(count, existence_prior=torch.full((count,), keep_probability))
        self.double().requires_grad_(False)

    @torch.no_grad()
    def update(self, keep_costs, tier_ids, tier_costs, keep_lr, tier_lr, global_gain):
        if any(not math.isfinite(v) or v <= 0 for v in (keep_lr, tier_lr, global_gain)):
            raise ValueError('positive finite mirror step sizes required')
        keep_costs = keep_costs.to(self.keep_logits)
        ids = tier_ids.to(self.logits.device).long()
        costs = tier_costs.to(self.logits)
        if (keep_costs.shape != self.keep_logits.shape or costs.shape != (len(ids), 3)
                or not torch.isfinite(keep_costs).all() or not torch.isfinite(costs).all()
                or ids.ndim != 1 or len(ids.unique()) != len(ids)
                or ((ids < 0) | (ids >= len(self.logits))).any()):
            raise ValueError('invalid mirror action costs/indices')
        def shift(logits, values, lr):
            # Remove the action-independent baseline before multiplying. Float64
            # preserves small scene-averaged costs; no per-row magnitude scaling.
            values = values - values.amin(-1, keepdim=True)
            result = logits - (lr*global_gain)*values
            if not torch.isfinite(result).all(): raise ValueError('mirror step overflow')
            result -= result.amax(-1, keepdim=True)
            # Numerical support floor, not a point/tier quota. Allows recovery
            # from a strongly disfavored action without exp underflow.
            return result.clamp_min(-700.)
        new_keep = shift(self.keep_logits, keep_costs, keep_lr)
        new_tier = shift(self.logits[ids], costs, tier_lr)
        self.keep_logits.copy_(new_keep); self.logits[ids] = new_tier


class BlockVisitScheduler:
    def __init__(self, id_batches):
        self.ids = [x.detach().cpu().long().clone() for x in id_batches]
        flat = torch.cat([x[x >= 0] for x in self.ids])
        if not len(flat) or sorted(flat.tolist()) != list(range(len(flat))):
            raise ValueError('blocks must contain each original source row exactly once')
        self.cursors = [torch.zeros(len(x), dtype=torch.long) for x in self.ids]
        self.counts = torch.zeros(len(flat), dtype=torch.long)
        self.last = torch.zeros_like(self.counts)

    def select(self, q):
        q = q.detach().cpu().long(); columns, chosen = [], []
        if q.shape != self.counts.shape: raise ValueError('tier count differs from source rows')
        for ids, cursors in zip(self.ids, self.cursors):
            selected = torch.full((len(ids),), -1, dtype=torch.long)
            # Tensorized scan across blocks; no per-point GPU synchronizations.
            width = ids.shape[1]
            order = (cursors[:, None] + torch.arange(width)) % width
            ordered_ids = ids.gather(1, order)
            present = (ordered_ids >= 0) & (q[ordered_ids.clamp_min(0)] > 0)
            active = present.any(-1); first = present.long().argmax(-1)
            selected[active] = order[active, first[active]]
            cursors[active] = (selected[active] + 1) % width
            chosen.append(ids[active, selected[active]]); columns.append(selected)
        return columns, torch.cat(chosen)

    def record(self, ids, step):
        ids = ids.detach().cpu().long()
        if step < 1 or len(ids.unique()) != len(ids): raise ValueError('invalid candidate visit')
        self.counts[ids] += 1; self.last[ids] = step

    def coverage(self, q):
        retained = q.detach().cpu() > 0
        return dict(evaluated_rows=int((self.counts > 0).sum()),
                    total_candidate_evaluations=int(self.counts.sum()),
                    complete_coverages=int(self.counts.min()),
                    unvisited_retained=int(((self.counts == 0) & retained).sum()),
                    retained_complete_coverages=int(self.counts[retained].min()) if retained.any() else 0)

    def state_dict(self):
        return dict(cursors=[x.clone() for x in self.cursors], counts=self.counts.clone(), last=self.last.clone())

    def load_state_dict(self, state):
        if len(state['cursors']) != len(self.ids): raise ValueError('block count changed')
        for old, new, ids in zip(self.cursors, state['cursors'], self.ids):
            if old.shape != new.shape or ((new < 0) | (new >= ids.shape[1])).any():
                raise ValueError('invalid block cursor')
        if state['counts'].shape != self.counts.shape or state['last'].shape != self.last.shape:
            raise ValueError('candidate coverage row count changed')
        self.cursors = [x.cpu().clone() for x in state['cursors']]
        self.counts, self.last = state['counts'].cpu().clone(), state['last'].cpu().clone()


class ViewWindowScheduler:
    def __init__(self, train_names, seed):
        self.names = list(train_names)
        if not self.names or len(set(self.names)) != len(self.names): raise ValueError('unique views required')
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.cursor, self.epochs = [], 0, 0

    def next(self, count):
        if not 1 <= count <= len(self.names): raise ValueError('window exceeds distinct training views')
        result = []
        while len(result) < count:
            if self.cursor == len(self.order):
                self.order = torch.randperm(len(self.names), generator=self.generator).tolist()
                self.cursor = 0; self.epochs += 1
            index = self.order[self.cursor]; self.cursor += 1
            if index not in result: result.append(index)
        return result

    def state_dict(self):
        return dict(names=self.names, order=self.order.copy(), cursor=self.cursor,
                    epochs=self.epochs, rng=self.generator.get_state())

    def load_state_dict(self, state):
        if state['names'] != self.names: raise ValueError('training views changed')
        if state['order'] and sorted(state['order']) != list(range(len(self.names))): raise ValueError('invalid view permutation')
        if not 0 <= state['cursor'] <= len(state['order']): raise ValueError('invalid view cursor')
        self.order, self.cursor, self.epochs = state['order'].copy(), state['cursor'], state['epochs']
        self.generator.set_state(state['rng'].cpu())
