"""All historical teachers, with indexed sampling and bounded read caching."""
import heapq
import math

import torch

from .pixel_benefit_replay import PixelReplay
from .replay_priority_index import PriorityIndex
from .spatial_benefit_replay import SpatialReplay


class FullHistorySpatialReplay(SpatialReplay):
    retains_all_training = True
    evict_on_capacity = False
    policy = 'full_history_coverage_priority_v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_index()

    def _reset_index(self):
        self.entries_by_id = {}
        self._positions = {}
        self._priority = PriorityIndex()
        self._buckets, self._bucket_positions = {}, []
        self._visit_heap = []
        self._draws = 0

    def _bucket_add(self, index, visits):
        if visits not in self._buckets:
            self._buckets[visits] = []
            heapq.heappush(self._visit_heap, visits)
        self._bucket_positions[index] = len(self._buckets[visits])
        self._buckets[visits].append(index)
        if len(self._visit_heap) > max(64, 2*len(self._buckets)):
            self._visit_heap = list(self._buckets)
            heapq.heapify(self._visit_heap)

    def _least(self):
        while self._visit_heap and self._visit_heap[0] not in self._buckets:
            heapq.heappop(self._visit_heap)
        value = self._visit_heap[0]
        return value, self._buckets[value]

    def add(self, record, validation=False):
        priority = self._priority.median()
        key = PixelReplay.add(self, record, validation)
        if key is not None and not validation:
            index = len(self.training)-1
            self.usage[key] = 0
            self.priorities[key] = priority
            self.entries_by_id[key] = self.training[index]
            self._positions[key] = index
            self._priority.add(index, priority)
            self._bucket_positions.append(0)
            self._bucket_add(index, 0)
        return key

    def update_priority(self, record_id, error):
        value = float(error)
        if record_id not in self.priorities:
            raise KeyError(record_id)
        if not math.isfinite(value) or value < 0:
            raise ValueError('priority must be finite and nonnegative')
        self._priority.update(self._positions[record_id], self.priorities[record_id], value)
        self.priorities[record_id] = value

    def mark_updated(self, record_ids):
        for key in record_ids:
            index = self._positions[key]
            visits = self.usage[key]
            bucket = self._buckets[visits]
            position = self._bucket_positions[index]
            last = bucket.pop()
            if position < len(bucket):
                bucket[position] = last
                self._bucket_positions[last] = position
            if not bucket:
                del self._buckets[visits]
            self.usage[key] = visits+1
            self._bucket_add(index, visits+1)
            self._draws += 1

    def sample(self, count):
        if count < 0 or (count and not self.training):
            raise ValueError('invalid replay draw count')
        if not count:
            return []
        n = len(self.training)
        least_visits, least = self._least()
        total = self._priority.total
        records = []
        for branch, unit in torch.rand((count, 2), generator=self.generator, dtype=torch.float64).tolist():
            if branch < .25:
                index = min(int(unit*n), n-1)
            elif branch < .75:
                index = least[min(int(unit*len(least)), len(least)-1)]
            else:
                index = self._priority.weighted(unit) if total > 0 else min(int(unit*n), n-1)
            entry = self.training[index]
            key = entry['id']
            probability = (.25/n + (.5/len(least) if self.usage[key] == least_visits else 0.)
                           + .25*(self.priorities[key]/total if total > 0 else 1/n))
            group = self._read(entry)
            group.update(replay_record_id=key, replay_probability=probability,
                         replay_importance_weight=1/(n*probability))
            records.append(group)
        return records

    def probabilities(self):
        """Explicit diagnostic; training draws use the incremental index."""
        n = len(self.training)
        if not n:
            raise ValueError('empty training replay')
        minimum, least = self._least()
        total = self._priority.total
        return torch.tensor([.25/n + (.5/len(least) if self.usage[e['id']] == minimum else 0.)
            + .25*(self.priorities[e['id']]/total if total > 0 else 1/n)
            for e in self.training], dtype=torch.float64)

    def coverage(self):
        n = len(self.training)
        histogram = sorted((v, len(b)) for v, b in self._buckets.items())
        def quantile(q):
            at = (n-1)*q
            low, high, before = 0, 0, 0
            for value, size in histogram:
                if before <= math.floor(at) < before+size:
                    low = value
                if before <= math.ceil(at) < before+size:
                    high = value
                before += size
            return low+(high-low)*(at-math.floor(at))
        return dict(active_groups=n, all_groups=n, all_group_draws=self._draws,
            never_updated_fraction=len(self._buckets.get(0, []))/n if n else 0.,
            min_visits=histogram[0][0] if n else 0, mean_visits=self._draws/n if n else 0.,
            p10_visits=quantile(.1) if n else 0., max_visits=histogram[-1][0] if n else 0,
            cache_bytes=self.cache_bytes, full_history=True)

    def state_dict(self):
        # Bucket ordering is needed for exact sampling resume after swap removals.
        return dict(super().state_dict(), full_history_policy=self.policy,
                    visit_buckets={k: list(v) for k, v in self._buckets.items()})

    def load_state_dict(self, state):
        if state.get('full_history_policy') != self.policy:
            raise ValueError('full history replay policy mismatch; use a new experiment')
        super().load_state_dict(state)
        self._reset_index()
        for index, entry in enumerate(self.training):
            key = entry['id']
            if key in self.entries_by_id or key not in self.priorities:
                raise ValueError('invalid full history record identity')
            priority = self.priorities[key]
            if not math.isfinite(priority) or priority < 0:
                raise ValueError('invalid historical priority')
            self.entries_by_id[key] = entry
            self._positions[key] = index
            self._priority.add(index, priority)
        if set(self.usage) != set(self.entries_by_id) or set(self.priorities) != set(self.entries_by_id):
            raise ValueError('full history record index mismatch')
        self._bucket_positions = [0]*len(self.training)
        seen = set()
        for visits, indices in state['visit_buckets'].items():
            if not isinstance(visits, int) or visits < 0 or not indices:
                raise ValueError('invalid historical visit bucket')
            for index in indices:
                if not isinstance(index, int) or not 0 <= index < len(self.training) or index in seen:
                    raise ValueError('invalid historical visit index')
                key = self.training[index]['id']
                if self.usage[key] != visits:
                    raise ValueError('historical visit count mismatch')
                seen.add(index)
                self._bucket_add(index, visits)
        if len(seen) != len(self.training):
            raise ValueError('missing historical visit index')
        self._draws = sum(self.usage.values())
