"""Learning and held-out evaluation of complete pixel intervention groups."""
import math
from collections import OrderedDict

import torch

from .pixel_benefit import PixelBenefitNetwork, integrate_density
from .response_value_report import summarize_response


def _views(group):
    if 'views' not in group:
        return [group]
    result = []
    for view in group['views']:
        item = dict(view)
        for key in ('source', 'step', 'importance_weight', 'rates'):
            if key in group and key not in item:
                item[key] = group[key]
        result.append(item)
    return result


def summarize_pixel_values(rows):
    """Report the same paired decision metrics as the response-value learner."""
    records = []
    for row in rows:
        truth = torch.as_tensor(row['truth'], dtype=torch.float64).cpu()
        pred = torch.as_tensor(row['prediction'], dtype=torch.float64).cpu()
        rates = torch.as_tensor(row['rates'], dtype=torch.float64).cpu()
        common = dict(source=row.get('source', 0), step=row.get('step', 0))
        records.append(dict(common, branch='keep', old=0, new=[1, 2, 3],
                            target=truth, prediction=pred, rates=rates,
                            conditional=torch.full((3,), 1/3, dtype=torch.float64)))
        for old in range(3):
            for new in range(3):
                if old == new:
                    continue
                records.append(dict(common, branch='positive', old=old+1, new=[new+1],
                    target=(truth[new]-truth[old])[None], prediction=(pred[new]-pred[old])[None],
                    rates=(rates[new]-rates[old])[None], conditional=torch.ones(1)))
    report = summarize_response(records)
    report['scope'] = 'complete per-view four-state groups; normalized distortion and rate'
    return report


class PixelLearner:
    def __init__(self, replay, device, hidden=32, lr=1e-4, tile_batch=8):
        if tile_batch <= 0:
            raise ValueError('tile_batch must be positive')
        self.replay, self.device, self.tile_batch = replay, torch.device(device), int(tile_batch)
        self.network = PixelBenefitNetwork(hidden=hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.updates = 0
        self.scales = None
        self._scene_cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def begin_window(self):
        self._scene_cache.clear()
        self.cache_hits = self.cache_misses = 0

    def fit_scales(self):
        if self.scales is not None:
            raise ValueError('training scales are already fixed')
        squares = torch.zeros(2, dtype=torch.float64)
        values = torch.zeros(2, dtype=torch.float64)
        pixels = 0.
        count = 0
        for group in self.replay.training_records():
            weight = float(group.get('importance_weight', 1.))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError('importance weight must be finite and positive')
            for record in _views(group):
                density = record['density'].double().cpu()
                valid = record['inputs']['valid'].double().cpu()
                mask = valid[:, None]
                if not mask.sum():
                    continue
                squares += (density.square()*mask).sum(dim=(0, 1, 3, 4)) * weight
                pixels += float(mask.sum()) * 3 * weight
                value = integrate_density(density, valid, record['height'], record['width'], record['d_ref'])
                values += value.square().sum(dim=0) * weight
                count += 3 * weight
        if not pixels or not count:
            raise ValueError('training replay has no valid pixels')
        map_rms = (squares/pixels).sqrt()
        value_rms = (values/count).sqrt()
        def floor(rms):
            return rms.clamp_min(max(1e-8, float(rms.max())*1e-3))
        self.scales = {'map': floor(map_rms), 'value': floor(value_rms)}
        self.network.set_output_scale(self.scales['map'])
        return {key: value.clone() for key, value in self.scales.items()}

    def _chunks(self, inputs):
        count = inputs['image'].shape[0]
        for start in range(0, count, self.tile_batch):
            stop = min(start+self.tile_batch, count)
            chunk = {}
            for key, values in inputs.items():
                selected = values[start:stop]
                if key in ('context', 'context_mask') and isinstance(values, (tuple, list)):
                    selected = torch.nn.utils.rnn.pad_sequence(selected, batch_first=True, padding_value=0)
                chunk[key] = selected.to(self.device)
            yield start, stop, chunk

    @torch.no_grad()
    def _view_error(self, record):
        """Evaluate complete-group loss and retain only its six value errors."""
        inputs = record['inputs']
        area = float(inputs['valid'].sum())
        if not area:
            return 0., None, area
        map_scale = self.scales['map'].to(self.device).reshape(1, 1, 2, 1, 1)
        value_scale = self.scales['value'].to(self.device).reshape(1, 2)
        map_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        error = torch.zeros((3, 2), device=self.device, dtype=torch.float64)
        for start, stop, chunk in self._chunks(inputs):
            pred = self.network(chunk)
            target = record['density'][start:stop].to(self.device)
            mask = chunk['valid'][:, None]
            difference = pred-target
            map_sum += ((difference/map_scale).square()*mask).sum()
            error += integrate_density(difference, chunk['valid'], record['height'], record['width'], record['d_ref'])
        loss = map_sum/(area*6) + (error/value_scale).square().mean()
        return float(loss), error, area

    def _backward_view(self, record, error, area, weight):
        if not area:
            return
        map_scale = self.scales['map'].to(self.device).reshape(1, 1, 2, 1, 1)
        value_scale = self.scales['value'].to(self.device).reshape(1, 2)
        value_coefficient = (2*error/value_scale.square()/6).detach()
        for start, stop, chunk in self._chunks(record['inputs']):
            pred = self.network(chunk)
            target = record['density'][start:stop].to(self.device)
            mask = chunk['valid'][:, None]
            map_loss = (((pred-target)/map_scale).square()*mask).sum()/(area*6)
            contribution = integrate_density(pred, chunk['valid'], record['height'], record['width'], record['d_ref'])
            (weight*(map_loss + (value_coefficient*contribution).sum())).backward()

    def train_updates(self, count, batch_groups=4):
        self._scene_cache.clear()
        if self.scales is None:
            raise ValueError('fit training scales before training')
        if count < 0 or batch_groups <= 0:
            raise ValueError('invalid training count or batch size')
        stats = {'loss': None, 'grad_norm': None, 'updates': self.updates}
        self.network.train()
        for _ in range(count):
            groups = self.replay.sample(batch_groups)
            losses, priorities, per_group = [], [], []
            for group in groups:
                records = _views(group)
                if not records:
                    raise ValueError('group has no views')
                evaluated = [self._view_error(record) for record in records]
                group_loss = sum(item[0] for item in evaluated) / len(records)
                source_weight = float(group.get('importance_weight', 1.))
                replay_weight = float(group.get('replay_importance_weight', 1.))
                if (not math.isfinite(replay_weight) or replay_weight <= 0 or
                    not math.isfinite(source_weight) or source_weight <= 0):
                    raise ValueError('importance weights must be finite and positive')
                weight = source_weight * replay_weight
                losses.append(group_loss * weight)
                priorities.append(group_loss)
                per_group.append((records, evaluated, weight/len(groups)/len(records)))
            self.optimizer.zero_grad(set_to_none=True)
            for records, evaluated, weight in per_group:
                for record, (_, error, area) in zip(records, evaluated):
                    self._backward_view(record, error, area, weight)
            norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), float('inf'))
            if not torch.isfinite(norm):
                raise ValueError('nonfinite gradient')
            self.optimizer.step()
            self.updates += 1
            for group, priority in zip(groups, priorities):
                if 'replay_record_id' in group:
                    self.replay.update_priority(group['replay_record_id'], priority)
            stats = {'loss': sum(losses)/len(losses), 'grad_norm': float(norm), 'updates': self.updates}
        return stats

    @torch.inference_mode()
    def predict(self, inputs, height, width, d_ref, cache_keys=None):
        self.network.eval()
        if cache_keys is not None and len(cache_keys) != inputs['image'].shape[0]:
            raise ValueError('cache keys must match tile count')
        value = self.network.output_scale.new_zeros((3, 2),dtype=torch.float64)
        for start, stop, chunk in self._chunks(inputs):
            if cache_keys is None:
                prediction = self.network(chunk)
            else:
                keys = cache_keys[start:stop]
                encoded = [None] * len(keys)
                missing = []
                for i, key in enumerate(keys):
                    if key in self._scene_cache:
                        self.cache_hits += 1
                        encoded[i] = self._scene_cache[key]
                        self._scene_cache.move_to_end(key)
                    else:
                        self.cache_misses += 1
                        missing.append(i)
                if missing:
                    indices = torch.tensor(missing, device=self.device, dtype=torch.long)
                    subset = {key: tensor.index_select(0, indices) for key, tensor in chunk.items()}
                    computed = self.network.encode_scene(subset)
                    for position, i in enumerate(missing):
                        item = tuple(t[position].detach().clone() for t in computed)
                        encoded[i] = item
                        self._scene_cache[keys[i]] = item
                        self._scene_cache.move_to_end(keys[i])
                        if len(self._scene_cache) > 128:
                            self._scene_cache.popitem(last=False)
                merged = tuple(torch.stack([item[j] for item in encoded]) for j in range(3))
                prediction = self.network.forward_with_scene(chunk, merged)
            value += integrate_density(prediction, chunk['valid'], height, width, d_ref)
        return value

    def validate(self):
        rows = []
        for group in self.replay.validation_records():
            truths, predictions, rates = [], [], None
            for record in _views(group):
                truth = integrate_density(record['density'], record['inputs']['valid'],
                                          record['height'], record['width'], record['d_ref'])
                predicted = self.predict(record['inputs'], record['height'], record['width'], record['d_ref'])
                next_rates = torch.as_tensor(record['rates'], dtype=torch.float64)
                if rates is not None and not torch.allclose(rates, next_rates):
                    raise ValueError('rates must match across views of a group')
                rates = next_rates
                truths.append(truth.cpu())
                predictions.append(predicted.cpu())
            if not truths:
                raise ValueError('validation group has no views')
            rows.append(dict(source=group.get('source', 0), step=group.get('step', 0),
                             rates=rates, truth=torch.stack(truths).mean(0),
                             prediction=torch.stack(predictions).mean(0)))
        report = summarize_pixel_values(rows)
        report.update(network_updates=self.updates, validation_groups=len(rows))
        return report

    def state_dict(self):
        return dict(network=self.network.state_dict(), optimizer=self.optimizer.state_dict(),
                    updates=self.updates, scales=self.scales)

    def load_state_dict(self, state):
        self._scene_cache.clear()
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates = int(state['updates'])
        self.scales = None if state['scales'] is None else {k: v.detach().cpu().clone() for k, v in state['scales'].items()}
