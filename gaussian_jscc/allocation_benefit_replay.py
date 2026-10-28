"""CPU action groups with immutable dynamic context and deterministic reservoir sampling."""
import copy

import torch


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


class BenefitReplay:
    def __init__(self, capacity, seed, *, static_source=None, validation=False):
        if capacity < 1:
            raise ValueError('positive replay capacity required')
        self.capacity, self.validation = int(capacity), bool(validation)
        self.static_source = None if static_source is None else static_source.detach().cpu()
        self.static_width = 0 if static_source is None else static_source.shape[-1]
        self.generator = torch.Generator().manual_seed(seed)
        self.groups, self.seen = [], 0

    def add(self, record):
        if bool(record['validation']) != self.validation:
            raise ValueError('validation/training replay separation violated')
        data = cpu_copy(record)
        if self.static_width:
            data['features']['point'] = data['features']['point'][..., self.static_width:].clone()
        self.seen += 1
        if len(self.groups) < self.capacity:
            self.groups.append(data)
        else:
            index = int(torch.randint(self.seen, (), generator=self.generator))
            if index < self.capacity:
                self.groups[index] = data

    def materialize(self, record):
        data = cpu_copy(record)
        if self.static_width:
            point = data['features']['point']
            source = self.static_source[int(data['source'])].expand(len(point), -1)
            data['features']['point'] = torch.cat((source, point), -1)
        return data

    def records(self):
        return [self.materialize(r) for r in self.groups]

    def sample(self, count, step):
        if count < 1 or not self.groups:
            raise ValueError('nonempty replay and positive batch required')
        result = []
        for branch, size in (('positive', count-count//2), ('keep', count//2)):
            pool = [i for i,r in enumerate(self.groups) if r['branch'] == branch]
            if not pool:
                pool = list(range(len(self.groups)))
            recent = [i for i in pool if self.groups[i]['step'] >= step-127] or pool
            for j in range(size):
                choices = recent if j < size//2 else pool
                index = choices[int(torch.randint(len(choices), (), generator=self.generator))]
                result.append(self.materialize(self.groups[index]))
        return result

    def state_dict(self):
        return dict(version=1, capacity=self.capacity, validation=self.validation,
                    static_width=self.static_width, static_count=0 if self.static_source is None else len(self.static_source),
                    seen=self.seen, groups=cpu_copy(self.groups), rng=self.generator.get_state())

    def load_state_dict(self, state):
        identity = (self.capacity, self.validation, self.static_width,
                    0 if self.static_source is None else len(self.static_source))
        saved = tuple(state[k] for k in ('capacity', 'validation', 'static_width', 'static_count'))
        if state.get('version') != 1 or saved != identity or len(state['groups']) > self.capacity:
            raise ValueError('benefit replay checkpoint mismatch')
        if state['seen'] < len(state['groups']) or any(bool(r['validation']) != self.validation for r in state['groups']):
            raise ValueError('invalid replay records')
        self.seen, self.groups = state['seen'], cpu_copy(state['groups'])
        self.generator.set_state(state['rng'].cpu())
