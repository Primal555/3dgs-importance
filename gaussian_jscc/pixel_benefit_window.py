"""Window-local mixed reconstruction and complete-block interventions."""

import math
import time
import torch
from torch.nn import functional as F

from .allocation_benefit_features import keep_rates
from .allocation_benefit_replay import cpu_copy
from .allocation_distortion import ImageDistortion
from .budgeted_allocation_kernel import raw_batches, rng_state, set_rng
from .optimization import preserved_rng
from .pixel_benefit_replay import sample_sources
from .pixel_benefit import integrate_density
from .pixel_local_renderer import capture


def build_group(*args, **kwargs):
    from .pixel_benefit_features import build_group as build
    return build(*args, **kwargs)


@torch.no_grad()
def decode_group(model, features, q, row, column, actual, semantic_y, noise, snr):
    """Decode one whole block in four states with the original noise slots."""
    out, tiers = decode_groups(model, features, q, torch.tensor([row], device=q.device),
                               torch.tensor([column], device=q.device), actual, semantic_y, noise, snr)
    return out[0], tiers[0]


@torch.no_grad()
def decode_groups(model, features, q, rows, columns, actual, semantic_y, noise, snr):
    """Batch independent interventions, including repeated original block rows."""
    rows, columns = rows.to(q.device), columns.to(q.device)
    f, original = features[rows], q[rows]
    index = torch.arange(len(rows), device=q.device)
    old = original[index, columns]
    alternate_mask = original > 0
    alternate_mask = alternate_mask.clone()
    alternate_mask[index, columns] = ~alternate_mask[index, columns]
    changed_y = model.semantic_encoder(f, f[..., :3], alternate_mask)
    base_y = semantic_y[rows]
    output = actual[rows, None].expand(-1, 4, -1, -1).clone()
    tier_maps = original[:, None].expand(-1, 4, -1).clone()
    tier_maps[index[:, None], torch.arange(4, device=q.device)[None, :], columns[:, None]] = torch.arange(4, device=q.device)
    for shift in (1, 2, 3):
        target = (old + shift) % 4
        alt = original.clone()
        alt[index, columns] = target
        changed = (target == 0) != (old == 0)
        y = torch.where(changed[:, None, None], changed_y, base_y)
        value = model.forward_tier_batches(f, f[..., :3], F.one_hot(alt, 4).to(f), snr, 'awgn',
                                          paired_noise=True, _semantic_y=y, _channel_noise=noise[rows])[0]
        output[index, target] = value
    return output, tier_maps


class PixelWindow:
    def __init__(self, model, scene, table, scheduler, reference, args):
        self.model, self.scene, self.table = model, scene, table
        self.scheduler, self.reference, self.args = scheduler, reference, args
        self.difficulty = torch.ones(len(table.logits), dtype=torch.float64)
        n = len(table.logits)
        self.positions = torch.empty(n, 4, dtype=torch.long)
        render_sources, offset = [], 0
        for batch, source_ids in enumerate(scene.group_ids):
            ids = source_ids.cpu().long()
            rows, cols = (ids >= 0).nonzero(as_tuple=True)
            sources = ids[rows, cols]
            self.positions[sources] = torch.stack((torch.full_like(rows, batch), rows, cols,
                                                  torch.arange(offset, offset + len(sources))), -1)
            render_sources.append(sources)
            offset += len(sources)
        self.render_sources = torch.cat(render_sources)
        if len(self.render_sources) != n or not torch.equal(self.render_sources.sort().values, torch.arange(n)):
            raise ValueError('group IDs must contain every source exactly once')

    def state_dict(self):
        return {'difficulty': self.difficulty.clone()}

    def load_state_dict(self, state):
        difficulty = state['difficulty'].cpu().double()
        if difficulty.shape != self.difficulty.shape or not torch.isfinite(difficulty).all() or (difficulty < 0).any():
            raise ValueError('invalid pixel difficulty state')
        self.difficulty = difficulty.clone()

    def prepare_camera(self, tile, target):
        pass

    def capture_inputs(self, mixed):
        return mixed

    def build_view(self, tile, changed, projected, target, *, source, block_ids, raw, with_labels):
        return build_group(tile, changed, projected, target, self.reference['lambda_s'],
                           self.reference['d_ref'], with_labels=with_labels)

    @staticmethod
    def truth(view):
        if 'values' in view:
            return view['values']
        return integrate_density(view['density'], view['inputs']['valid'],
                                 view['height'], view['width'], view['d_ref'])

    @torch.no_grad()
    def collect(self, step, cameras, learner, replay, pretrain):
        model, scene, table, scheduler = self.model, self.scene, self.table, self.scheduler
        if hasattr(learner,'begin_window'): learner.begin_window()
        device = next(model.parameters()).device
        n = len(table.logits)
        snr = self.args.snr
        q_source = torch.empty(n, dtype=torch.long)
        entries, parts = [], []
        timings = dict(actual_decode_seconds=0., projection_seconds=0., candidate_decode_seconds=0.,
                       teacher_seconds=0., feature_seconds=0., network_seconds=0.)
        for batch, (f_cpu, ids_cpu) in enumerate(zip(scene.groups, scene.group_ids)):
            f, ids = f_cpu.to(device), ids_cpu.to(device)
            valid = ids >= 0
            q = table.sample(ids.clamp_min(0), snr)[0].masked_fill(~valid, 0)
            q_source[ids[valid].cpu()] = q[valid].cpu()
            started = time.perf_counter()
            y = model.semantic_encoder(f, f[..., :3], q > 0)
            state = rng_state(device)
            with preserved_rng(device):
                set_rng(state, device)
                noise = torch.randn((*q.shape, 2 * model.cfg.rates[-1]), device=device, dtype=f.dtype)
                noise *= math.sqrt(10 ** (-float(snr) / 10) / 2)
            actual = model.forward_tier_batches(f, f[..., :3], F.one_hot(q, 4).to(f),
                snr, 'awgn', paired_noise=True, _semantic_y=y)[0]
            raw = raw_batches(actual, scene.geometry, model)
            timings['actual_decode_seconds'] += time.perf_counter() - started
            parts.append(torch.cat((raw[valid], (q[valid] > 0).to(raw)[:, None]), -1))
            entries.append((f, ids, q, y, actual, noise))
        mixed = self.capture_inputs(torch.cat(parts))
        del parts
        captures, targets, components = [], [], []
        distortion = ImageDistortion('mse_ssim' if self.reference['lambda_s'] else 'mse', self.reference['lambda_s'])
        for camera in cameras:
            started = time.perf_counter()
            tile = capture(mixed[:, :-1], camera, scene.degree, mixed[:, -1], scene.white_background,
                           backend='cuda' if device.type == 'cuda' else 'cpu')
            timings['projection_seconds'] += time.perf_counter() - started
            target = scene.reference.get(camera, device)
            started = time.perf_counter()
            self.prepare_camera(tile, target)
            timings['feature_seconds'] += time.perf_counter() - started
            captures.append(tile); targets.append(target)
            components.append(distortion.components(tile.image, target))
        if not cameras:
            raise ValueError('at least one camera required')
        impact = torch.zeros(n, dtype=torch.float64)
        projected_impact = torch.stack([tile.projected[:, 11].clamp_min(0).square() *
                                        tile.projected[:, 6].clamp_min(0) for tile in captures]).sum(0)
        impact[self.render_sources] = projected_impact.double().cpu()
        if pretrain:
            candidates = self.render_sources.tolist()
        else:
            candidates = []
            for columns in scheduler.response_rounds(step):
                for (_, ids, _, _, _, _), cols in zip(entries, columns):
                    cols = cols.to(device)
                    rows = (cols >= 0).nonzero(as_tuple=True)[0]
                    candidates.extend(ids[rows, cols[rows]].cpu().tolist())
            candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise ValueError('candidate pool is empty')
        candidate_ids=torch.tensor(candidates,dtype=torch.long)
        validation=candidate_ids[scheduler.validation[candidate_ids]].tolist()
        training=candidate_ids[~scheduler.validation[candidate_ids]].tolist()
        limit = int(self.args.label_groups)
        room = max(0, replay.validation_groups - len(replay.validation)) if not replay.validation_frozen else 0
        val_count = min(room, len(validation), max(1, limit // 5)) if limit else 0
        order = torch.randperm(len(validation), generator=scheduler.generator)[:val_count].tolist()
        val_draws = [validation[i] for i in order]
        train_count = limit - len(val_draws)
        draws = sample_sources(training, impact[training], self.difficulty[training], train_count,
                               scheduler.generator) if training and train_count else []
        labels = {s for s in val_draws} | {int(d['source']) for d in draws}
        evaluated = list(dict.fromkeys(candidates if not pretrain else
                                       val_draws + [int(draw['source']) for draw in draws]))
        keep = torch.zeros(n, 2, dtype=torch.float64, device=device)
        selected, costs, built = [], [], {}
        evaluated_tiles=cache_hits=0
        side = self.reference['side_proxy']
        rate_cache = [keep_rates(q, ids >= 0, model.cfg.rates, xyz=side['xyz_per_retained'],
                                 bits=side['bits_per_use']) * self.args.beta / self.reference['c_ref']
                      for _, ids, q, _, _, _ in entries]
        def decoded_batches():
            pool = torch.tensor(evaluated, dtype=torch.long)
            batch_size = int(getattr(self.args, 'candidate_batch', 16))
            if batch_size <= 0:
                raise ValueError('candidate_batch must be positive')
            for batch, (f, ids, q, y, actual, noise) in enumerate(entries):
                sources = pool[self.positions[pool, 0] == batch]
                sources=sources[torch.argsort(self.positions[sources,1],stable=True)]
                for start in range(0, len(sources), batch_size):
                    group = sources[start:start + batch_size]
                    loc = self.positions[group]
                    rows, cols = loc[:, 1].to(device), loc[:, 2].to(device)
                    started = time.perf_counter()
                    alternatives, tier_maps = decode_groups(model, f, q, rows, cols, actual, y, noise, snr)
                    shape = alternatives.shape
                    raw = raw_batches(alternatives.reshape(-1, *shape[2:]), scene.geometry, model)
                    raw = raw.reshape(*shape[:2], *raw.shape[1:])
                    timings['candidate_decode_seconds'] += time.perf_counter() - started
                    for i, source in enumerate(group.tolist()):
                        yield source, batch, int(loc[i, 1]), int(loc[i, 2]), raw[i], tier_maps[i]
        for source, batch, row, col, raw, tiers in decoded_batches():
            f, ids, q, y, actual, noise = entries[batch]
            valid = ids[row] >= 0
            block_ids = scene.group_ids[batch][row]
            block_ids = block_ids[block_ids >= 0].cpu().long()
            changed = self.positions[block_ids, 3].to(device)
            rates = rate_cache[batch][row, col]
            views, predictions = [], []
            for tile, target in zip(captures, targets):
                started = time.perf_counter()
                projected = torch.stack([tile.project(raw[tier, valid], (tiers[tier, valid] > 0).to(raw))
                                         for tier in range(4)])
                timings['projection_seconds'] += time.perf_counter() - started
                started = time.perf_counter()
                view = self.build_view(tile, changed, projected, target, source=source,
                                       block_ids=block_ids, raw=raw[:,valid], with_labels=source in labels)
                timings['feature_seconds'] += view.get('feature_seconds', time.perf_counter() - started)
                timings['teacher_seconds'] += view.get('teacher_seconds', 0.)
                evaluated_tiles+=view['tile_count'] if 'tile_count' in view else len(view['inputs']['valid'])
                cache_hits+=view.get('cache_hits',0)
                if getattr(learner, 'scales', None) is not None:
                    started = time.perf_counter()
                    kwargs={'cache_keys':view['prediction_cache_keys']} if 'prediction_cache_keys' in view and hasattr(learner,'begin_window') else {}
                    predictions.append(learner.predict(view['inputs'], view['height'], view['width'], view['d_ref'],**kwargs))
                    timings['network_seconds'] += time.perf_counter() - started
                view.pop('prediction_cache_keys',None)
                if source in labels:
                    views.append(cpu_copy(view))
                del view
            if source in labels:
                built[source] = dict(source=source, step=step, validation=bool(scheduler.validation[source]),
                                     views=views, rates=rates.detach().cpu())
                if predictions and not bool(scheduler.validation[source]):
                    truth = torch.stack([self.truth(view) for view in views]).mean(0)
                    scale=learner.scales['value'] if isinstance(learner.scales,dict) else torch.ones(2)
                    error = float(((torch.stack(predictions).mean(0).cpu()-truth)/scale).square().mean())
                    self.difficulty[source] = 0.9 * self.difficulty[source] + 0.1 * error
            if predictions:
                values = torch.stack(predictions).mean(0).sum(-1).double()
                cost = values + rates
                conditional = table.logits[source].softmax(-1).to(device)
                keep[source, 1] = (cost * conditional).sum()
                selected.append(source); costs.append(cost)
        for source in val_draws:
            replay.add(dict(built[source], importance_weight=1.), True)
        for draw in draws:
            source = int(draw['source'])
            replay.add(dict(built[source], importance_weight=draw['importance_weight'],
                            proposal_probability=draw['proposal_probability']), False)
        scheduler.record(torch.tensor(evaluated, dtype=torch.long), step)
        mean = torch.stack(components).mean(0)
        return dict(keep_costs=keep, tier_ids=torch.tensor(selected, dtype=torch.long),
                    tier_costs=torch.stack(costs) if costs else torch.empty(0, 3, device=device, dtype=torch.float64),
                    q=q_source, metrics=dict(timings, source_mse=float(mean[0]),
                        source_structure_error=float(mean[1]), source_distortion=float(distortion.combine(mean)),
                        evaluated_groups=len(evaluated), labeled_groups=len(labels), camera_captures=len(captures),
                        evaluated_tiles=evaluated_tiles,evaluated_pixels=evaluated_tiles*256,
                        context_cache_hits=cache_hits,network_cache_hits=getattr(learner,'cache_hits',0),
                        timing_scope='host wall time including synchronization points'))
