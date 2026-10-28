"""Online supervised learning; label ingestion happens after this window's decisions."""
import math

import torch

from .allocation_benefit import FEATURE_KEYS, group_losses
from .allocation_benefit_replay import BenefitReplay, cpu_copy


def pack_groups(records, device):
    sizes = torch.tensor([len(r['target']) for r in records], dtype=torch.long, device=device)
    groups = torch.repeat_interleave(torch.arange(len(records), device=device), sizes)
    features = {k: torch.cat([r['features'][k] for r in records]).to(device=device, dtype=torch.float32)
                for k in FEATURE_KEYS}
    branch = torch.tensor([int(r['branch']=='keep') for r in records], device=device)
    target = torch.cat([r['target'] for r in records]).to(device=device, dtype=torch.float32)
    rates = torch.cat([r['rates'] for r in records]).to(device=device, dtype=torch.float32)
    conditional = torch.cat([r['conditional'] for r in records]).to(device=device, dtype=torch.float32)
    return features, target, rates, conditional, groups, branch


class BenefitLearner:
    def __init__(self, network, *, static_source=None, capacity=65536, seed=42, lr=1e-4,
                 fit_windows=50, updates_per_window=4, batch_size=128, mask_fraction=.25, mask_weight=.5):
        if (capacity<2 or fit_windows<1 or updates_per_window<1 or batch_size<2 or
                not math.isfinite(lr) or lr<=0 or not 0<=mask_fraction<=1 or
                not math.isfinite(mask_weight) or mask_weight<0):
            raise ValueError('invalid benefit learning settings')
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.replay = BenefitReplay(capacity, seed+1101, static_source=static_source)
        self.validation = BenefitReplay(capacity, seed+1102, static_source=static_source, validation=True)
        self.generator = torch.Generator().manual_seed(seed+1103)
        self.fit_windows, self.updates_per_window, self.batch_size = fit_windows, updates_per_window, batch_size
        self.mask_fraction, self.mask_weight = mask_fraction, mask_weight
        self.settings = [lr, fit_windows, updates_per_window, batch_size, mask_fraction, mask_weight]
        self.updates = 0
        self.normalization_records = []

    @property
    def device(self):
        return next(self.network.parameters()).device

    @torch.no_grad()
    def predict(self, features, branch, *, masked=False):
        self.network.eval()
        inputs = {k:v.to(device=self.device, dtype=torch.float32) for k,v in features.items()}
        return self.network(inputs, branch, masked=masked).double()

    def observe(self, records, step):
        for record in records:
            (self.validation if record['validation'] else self.replay).add(record)
            if step<=self.fit_windows and not record['validation']:
                self.normalization_records.append(cpu_copy(record))
        if step == self.fit_windows:
            self.network.fit(self.normalization_records)
            self.normalization_records.clear()
        stats = {}
        if step > self.fit_windows:
            if not bool(self.network.ready):
                raise RuntimeError('missing fixed benefit normalization')
            values = [self.train_step(step) for _ in range(self.updates_per_window)]
            stats = {k:sum(v[k] for v in values)/len(values) for k in values[0]}
        return dict(stats, network_ready=bool(self.network.ready), network_updates=self.updates,
                    replay_groups=len(self.replay.groups), validation_groups=len(self.validation.groups))

    def train_step(self, step):
        records = self.replay.sample(self.batch_size, step)
        x, target, rates, conditional, groups, branches = pack_groups(records, self.device)
        scales = self.network.scales[branches]
        self.network.train()
        predicted = self.network(x, branches[groups])
        losses = group_losses(predicted, target, rates, conditional, groups, scales)
        mask = torch.rand(len(records), generator=self.generator)<self.mask_fraction
        if mask.any() and self.mask_weight:
            mask = mask.to(self.device);selected = mask.nonzero(as_tuple=True)[0]
            rows = mask[groups];local_groups = mask.long().cumsum(0)[groups[rows]]-1
            masked = self.network({k:v[rows] for k,v in x.items()}, branches[groups[rows]], masked=True)
            masked_losses = group_losses(masked,target[rows],rates[rows],conditional[rows],local_groups,scales[selected])
            losses = losses+self.mask_weight*torch.zeros_like(losses).scatter(0,selected,masked_losses)
        parts = [losses[branches==index].mean() for index in (0,1) if (branches==index).any()]
        loss = torch.stack(parts).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite benefit training loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step(); self.updates += 1
        return dict(benefit_training_loss=float(loss.detach()),benefit_grad_norm=float(gradient),
                    benefit_grad_clipped=float(gradient>1))

    def state_dict(self):
        return dict(version=1, settings=self.settings, network=cpu_copy(self.network.state_dict()),
                    optimizer=cpu_copy(self.optimizer.state_dict()), replay=self.replay.state_dict(),
                    validation=self.validation.state_dict(), rng=self.generator.get_state(), updates=self.updates,
                    normalization_records=cpu_copy(self.normalization_records))

    def load_state_dict(self, state):
        if state.get('version') != 1 or state.get('settings') != self.settings:
            raise ValueError('benefit learner checkpoint settings mismatch')
        self.network.load_state_dict(state['network']); self.optimizer.load_state_dict(state['optimizer'])
        self.replay.load_state_dict(state['replay']); self.validation.load_state_dict(state['validation'])
        self.generator.set_state(state['rng'].cpu()); self.updates = state['updates']
        self.normalization_records = cpu_copy(state['normalization_records'])
