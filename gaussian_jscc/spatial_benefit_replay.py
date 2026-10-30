"""Coverage-aware replay with exact sampling corrections and bounded LRU reads."""
from collections import OrderedDict
import math

import torch

from .pixel_benefit_replay import PixelReplay, _distribution


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class SpatialReplay(PixelReplay):
    def __init__(self, *args, cache_mib=256, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isfinite(cache_mib) or cache_mib < 0:
            raise ValueError('invalid replay cache size')
        self.cache_limit = int(cache_mib*2**20)
        self.cache = OrderedDict(); self.cache_bytes = 0
        self.usage = {}

    def add(self, record, validation=False):
        record_id = super().add(record, validation)
        if record_id is not None and not validation:
            self.usage[record_id] = 0
            # New entries use the current typical priority. Coverage is explicit.
            old = [v for k, v in self.priorities.items() if k != record_id]
            self.priorities[record_id] = float(torch.tensor(old).median()) if old else 1.
        return record_id

    def _read(self, entry):
        key = entry['id']
        if key in self.cache:
            record, size = self.cache.pop(key)
            self.cache[key] = (record, size)
        else:
            record = super()._read(entry); size = tensor_bytes(record)
            if size <= self.cache_limit:
                while self.cache and self.cache_bytes+size > self.cache_limit:
                    _, (_, removed) = self.cache.popitem(last=False); self.cache_bytes -= removed
                self.cache[key] = (record, size); self.cache_bytes += size
        # Sampling annotations must never mutate a cached immutable shard.
        return dict(record)

    def probabilities(self):
        n = len(self.training)
        if not n:
            raise ValueError('empty training replay')
        visits = torch.tensor([self.usage[e['id']] for e in self.training])
        least = (visits == visits.min()).double(); least /= least.sum()
        hard = _distribution([self.priorities[e['id']] for e in self.training], n)
        return .25/n + .5*least + .25*hard

    def sample(self, count):
        if count < 0 or (count and not self.training):
            raise ValueError('invalid replay draw count')
        if not count:
            return []
        probabilities = self.probabilities()
        indices = torch.multinomial(probabilities, count, replacement=True, generator=self.generator)
        records = []
        for i in indices.tolist():
            entry = self.training[i]; record = self._read(entry); p = float(probabilities[i])
            record.update(replay_record_id=entry['id'], replay_probability=p,
                          replay_importance_weight=1/(len(self.training)*p))
            records.append(record)
        return records

    def mark_updated(self, record_ids):
        for key in record_ids:
            if key not in self.priorities:
                raise KeyError(key)
            self.usage[key] += 1

    def coverage(self):
        live = torch.tensor([self.usage[e['id']] for e in self.training], dtype=torch.float64)
        all_visits = list(self.usage.values())
        return dict(active_groups=len(live), all_groups=len(all_visits), all_group_draws=sum(all_visits),
            never_updated_fraction=float((live == 0).double().mean()) if len(live) else 0.,
            min_visits=int(live.min()) if len(live) else 0,
            mean_visits=float(live.mean()) if len(live) else 0.,
            p10_visits=float(torch.quantile(live, .1)) if len(live) else 0.,
            max_visits=int(live.max()) if len(live) else 0, cache_bytes=self.cache_bytes)

    def state_dict(self):
        return dict(super().state_dict(), spatial_replay_version=1, usage=dict(self.usage))

    def load_state_dict(self, state):
        if state.get('spatial_replay_version') != 1:
            raise ValueError('spatial replay version mismatch')
        usage = state['usage']
        if (any(e['id'] not in usage for e in state['training']) or
                any(not isinstance(v, int) or v < 0 for v in usage.values())):
            raise ValueError('invalid optimizer visit counters')
        super().load_state_dict(state)
        self.usage = dict(usage); self.cache.clear(); self.cache_bytes = 0
