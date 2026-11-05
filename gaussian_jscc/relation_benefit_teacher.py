"""Stable context-conditioned, paired-noise supervision in original PLY order."""
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from .budgeted_allocation_kernel import raw_batches
from .pixel_benefit_features import exact_values
from .pixel_benefit_window import decode_groups
from .pixel_local_renderer import capture


def _digest(items):
    digest = hashlib.sha256()
    for value in items:
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        else:
            digest.update(repr(value).encode())
        digest.update(b'\0')
    return digest.hexdigest()


class RelationTeacher:
    """Cache immutable contexts and labels for a fixed independent trial bank.

    Features, semantics, decoded blocks and noise slots stay on CPU. Each trial
    retains one local renderer capture per camera on the codec device. Further
    collections decode only previously unseen sources, in candidate batches.
    """

    def __init__(self, model, scene, cameras, reference, *, snr=10., trials=2,
                 seed=42, candidate_batch=64, background_tiers=None,
                 context_id=None, context_name=None):
        self.model, self.scene = model, scene
        self.cameras, self.reference = tuple(cameras), dict(reference)
        self.snr, self.trials, self.seed = float(snr), int(trials), int(seed)
        self.candidate_batch = int(candidate_batch)
        self.device = next(model.parameters()).device
        self._assert_frozen()
        if not self.cameras or self.trials < 1 or self.candidate_batch < 1:
            raise ValueError('positive trials, candidate batch and nonempty cameras required')
        if not math.isfinite(self.snr):
            raise ValueError('finite teacher SNR required')
        if (not math.isfinite(float(reference['d_ref'])) or float(reference['d_ref']) <= 0
                or not math.isfinite(float(reference['lambda_s'])) or float(reference['lambda_s']) < 0):
            raise ValueError('positive d_ref and nonnegative lambda_s required')
        if getattr(scene.reference, 'target', 'source') != 'source':
            raise ValueError('relation teacher requires a source PLY rendering reference')
        self.n = len(scene.raw)
        tiers = torch.full((self.n,), 2, dtype=torch.long) if background_tiers is None else torch.as_tensor(background_tiers).detach().cpu()
        if (tiers.shape != (self.n,) or tiers.dtype == torch.bool or tiers.is_floating_point()
                or ((tiers < 0) | (tiers > 3)).any()):
            raise ValueError('background_tiers must be original-order integer q0--q3 values')
        if context_id is not None and (type(context_id) is not int or context_id < 0):
            raise ValueError('context_id must be a nonnegative integer')
        self.background_tiers = tiers.long().clone()
        self.context_id = context_id
        self.context_name = str(context_name) if context_name is not None else None
        order = torch.as_tensor(scene.order).cpu().long()
        if order.shape != (self.n,) or not torch.equal(order.sort().values, torch.arange(self.n)):
            raise ValueError('scene.order must map every Morton row to one original source')
        self.source_to_morton = torch.argsort(order)
        self.background_masks = (self.background_tiers[order] > 0).float()
        self.positions = torch.full((self.n, 4), -1, dtype=torch.long)
        present = []
        if len(scene.groups) != len(scene.group_ids):
            raise ValueError('scene groups and source IDs must align')
        for batch, (features, ids) in enumerate(zip(scene.groups, scene.group_ids)):
            ids = ids.cpu().long()
            if ids.shape != features.shape[:2] or ((ids < -1) | (ids >= self.n)).any():
                raise ValueError('invalid source IDs in codec groups')
            rows, cols = (ids >= 0).nonzero(as_tuple=True)
            sources = ids[rows, cols]
            self.positions[sources] = torch.stack((torch.full_like(rows, batch), rows, cols,
                                                  self.source_to_morton[sources]), -1)
            present.append(sources)
        if not present or not torch.equal(torch.cat(present).sort().values, torch.arange(self.n)):
            raise ValueError('codec groups must contain every original source exactly once')
        self.entries, self.captures, self.targets = [], [], []
        self._backgrounds, self._labels = [], {}
        self._prepared = False
        source_raw = torch.empty_like(scene.raw.cpu())
        source_raw[order] = scene.raw.cpu()
        camera_fields = ('image_name', 'FoVx', 'FoVy', 'image_width', 'image_height',
                         'world_view_transform', 'full_proj_transform', 'camera_center')
        from .transport import model_id
        self._identity = dict(version=1, codec_sha256=model_id(model),
            source_order='original_ply', render_order='morton',
            source_name=str(getattr(scene, 'spec', {}).get('name', 'unnamed')),
            source_sha256=_digest((source_raw,)), source_fingerprint=str(getattr(scene, 'fingerprint', '')),
            layout_sha256=_digest((order, *scene.group_ids)),
            geometry_sha256=_digest((scene.geometry.lower, scene.geometry.span, scene.geometry.bits)),
            sh_degree=int(scene.degree), white_background=bool(scene.white_background),
            camera_sha256=_digest(getattr(c, field, None) for c in self.cameras for field in camera_fields),
            view_names=[str(getattr(c, 'image_name', 'unnamed')) for c in self.cameras],
            view_weights=[1./len(self.cameras)]*len(self.cameras),
            reference_target='source', background_tier=(int(tiers[0]) if self.n and (tiers == tiers[0]).all() else None),
            background_tiers_sha256=_digest((self.background_tiers,)),
            context_id=self.context_id, context_name=self.context_name, snr=self.snr, seed=self.seed,
            noise_protocol='fixed_pair_trials', noise_trials=self.trials,
            noise_seed_rule='(seed+104729*(trial+1)+1000003*(batch+1)) mod (2**63-1)',
            components=['delta_mse', 'lambda_s_delta_structure'], relative_tier=0,
            lambda_s=float(reference['lambda_s']), d_ref=float(reference['d_ref']))

    def _assert_frozen(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('relation teacher requires a frozen eval-mode codec')

    def identity(self):
        return json.loads(json.dumps(self._identity, allow_nan=False))

    def release_device(self):
        """Discard device captures; fixed decoded CPU backgrounds remain reusable."""
        self.captures = []
        self.targets = []
        self._prepared = False

    @torch.no_grad()
    def _prepare(self):
        if self._prepared:
            return
        self._assert_frozen()
        self.targets = [self.scene.reference.get(camera, self.device).detach() for camera in self.cameras]
        for trial in range(self.trials):
            if len(self.entries) > trial:
                background = self._backgrounds[trial].to(self.device)
                masks = self.background_masks.to(self.device)
                self.captures.append([capture(background, camera, self.scene.degree, masks,
                    self.scene.white_background, backend=self.device.type) for camera in self.cameras])
                continue
            entries = []
            background = torch.empty_like(self.scene.raw, device=self.device)
            masks = self.background_masks.to(self.device)
            for batch, (f_cpu, ids_cpu) in enumerate(zip(self.scene.groups, self.scene.group_ids)):
                f, ids = f_cpu.to(self.device), ids_cpu.to(self.device)
                valid = ids >= 0
                q = self.background_tiers[ids_cpu.cpu().long().clamp_min(0)].to(self.device).masked_fill(~valid, 0)
                semantic = self.model.semantic_encoder(f, f[..., :3], q > 0)
                noise_seed = (self.seed + 104729*(trial+1) + 1000003*(batch+1)) % (2**63-1)
                generator = torch.Generator(device=self.device).manual_seed(noise_seed)
                noise = torch.randn((*q.shape, 2*self.model.cfg.rates[-1]), generator=generator,
                                    dtype=f.dtype, device=self.device)
                noise *= math.sqrt(10**(-self.snr/10)/2)
                actual = self.model.forward_tier_batches(f, f[..., :3], F.one_hot(q, 4).to(f),
                    self.snr, 'awgn', paired_noise=True, _semantic_y=semantic, _channel_noise=noise)[0]
                raw = raw_batches(actual, self.scene.geometry, self.model)
                # Original group IDs are converted explicitly to the renderer's Morton rows.
                slots = self.source_to_morton[ids[valid].cpu()].to(self.device)
                background[slots] = raw[valid]
                entries.append(tuple(x.detach().cpu() for x in (f, ids, q, semantic, actual, noise)))
                del f, ids, q, semantic, actual, noise, raw
            self.entries.append(entries)
            self._backgrounds.append(background.detach().cpu())
            self.captures.append([capture(background, camera, self.scene.degree, masks,
                self.scene.white_background, backend=self.device.type) for camera in self.cameras])
        self._prepared = True

    def _sources(self, sources):
        values = torch.as_tensor(sources, dtype=torch.long).cpu()
        if values.ndim != 1 or ((values < 0) | (values >= self.n)).any():
            raise ValueError('source IDs must be a one-dimensional list in original PLY range')
        return list(dict.fromkeys(values.tolist()))

    @torch.no_grad()
    def forward_candidates(self, sources, trial=0):
        """Return (sources, decoded four-state blocks, tiers) for independent checks."""
        sources = self._sources(sources)
        if not 0 <= int(trial) < self.trials:
            raise ValueError('trial index outside fixed noise bank')
        self._prepare()
        results = []
        pool = torch.tensor(sources, dtype=torch.long)
        for batch, entry in enumerate(self.entries[int(trial)]):
            selected = pool[self.positions[pool, 0] == batch]
            for first in range(0, len(selected), self.candidate_batch):
                group = selected[first:first+self.candidate_batch]
                loc = self.positions[group]
                rows, columns = loc[:, 1], loc[:, 2].to(self.device)
                f, _, q, semantic, actual, noise = (x[rows].to(self.device) for x in entry)
                local_rows = torch.arange(len(group), device=self.device)
                alternatives, tiers = decode_groups(self.model, f, q, local_rows, columns,
                                                    actual, semantic, noise, self.snr)
                results.append((group, alternatives, tiers))
        return results

    @torch.no_grad()
    def collect(self, sources, step=0):
        sources = self._sources(sources)
        self._assert_frozen()
        unseen = [source for source in sources if source not in self._labels]
        if unseen:
            self._prepare()
            totals = {source: torch.zeros(len(self.cameras), 3, 2, dtype=torch.float64) for source in unseen}
            for trial in range(self.trials):
                for group, alternatives, tiers in self.forward_candidates(unseen, trial):
                    shape = alternatives.shape
                    raw = raw_batches(alternatives.reshape(-1, *shape[2:]), self.scene.geometry, self.model)
                    raw = raw.reshape(*shape[:2], *raw.shape[1:])
                    for index, source in enumerate(group.tolist()):
                        batch, row, _, _ = self.positions[source].tolist()
                        ids = self.scene.group_ids[batch][row].cpu().long()
                        valid = ids >= 0
                        changed = self.source_to_morton[ids[valid]].to(self.device)
                        candidate_raw = raw[index, :, valid.to(self.device)]
                        masks = (tiers[index, :, valid.to(self.device)] > 0).to(raw)
                        for view, (tile, target) in enumerate(zip(self.captures[trial], self.targets)):
                            projected = torch.stack([tile.project(candidate_raw[q], masks[q]) for q in range(4)])
                            totals[source][view] += exact_values(tile, changed, projected, target,
                                self.reference['lambda_s'], self.reference['d_ref']).cpu().double()
            for source, total in totals.items():
                self._labels[source] = total / self.trials
        rows = [dict(source=source, step=int(step), values=self._labels[source].clone(),
                     noise_trials=self.trials) for source in sources]
        if self.context_id is not None:
            for row in rows:row.update(context=self.context_id, context_name=self.context_name)
        return rows

    @torch.no_grad()
    def verify(self):
        """Certify two real CUDA blocks against the original full rasterizer."""
        self._assert_frozen()
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('relation teacher preflight requires actual CUDA')
        from .pixel_benefit_preflight import assert_close, verify_group, verify_expansion_probe
        from .rendering import render
        self._prepare()
        camera, tile = self.cameras[0], self.captures[0][0]
        baseline = render(self._backgrounds[0].to(self.device), camera, self.scene.degree,
                          self.scene.white_background, self.background_masks.to(self.device))
        baseline_check = assert_close('relation background capture versus original rasterizer', tile.image,
                                     baseline, atol=3e-5, rtol=3e-5)
        del baseline
        # Choose two populated independent blocks with visible contributions.
        scores = (tile.projected[:, 11].square()*tile.projected[:, 6]).cpu()
        blocks = []
        for ids in self.scene.group_ids:
            ids = ids.cpu().long()
            point_scores = scores[self.source_to_morton[ids.clamp_min(0)]].masked_fill(ids < 0, -1)
            maximum, columns = point_scores.max(-1)
            sources = ids[torch.arange(len(ids)), columns]
            blocks.extend(zip(maximum.tolist(), sources.tolist()))
        selected = [source for score, source in sorted(blocks, reverse=True) if score > 0][:2]
        if len(selected) < 2:
            raise RuntimeError('relation preflight needs two visible populated codec blocks')
        reports = []
        for source in selected:
            batch, row, col, _ = self.positions[source].tolist()
            f, ids, q, semantic, actual, noise = (x.to(self.device) for x in self.entries[0][batch])
            _, alternatives, tiers = self.forward_candidates([source], 0)[0]
            codec_checks = []
            for tier in range(4):
                changed_q = q.clone()
                changed_q[row, col] = tier
                independent = self.model.forward_tier_batches(f, f[..., :3], F.one_hot(changed_q, 4).to(f),
                    self.snr, 'awgn', paired_noise=True, _channel_noise=noise)[0]
                codec_checks.append(dict(tier=tier, **assert_close('relation block versus full paired-noise batch',
                    alternatives[0, tier], independent[row], atol=2e-5, rtol=2e-5)))
            valid = ids[row] >= 0
            changed = self.source_to_morton[ids[row, valid].cpu()].to(self.device)
            candidate_raw = raw_batches(alternatives[0], self.scene.geometry, self.model)[:, valid]
            masks = (tiers[0, :, valid] > 0).to(candidate_raw)
            projected = torch.stack([tile.project(candidate_raw[tier], masks[tier]) for tier in range(4)])
            def full_renderer(tier):
                background = self._backgrounds[0].to(self.device).clone()
                background[changed] = candidate_raw[tier]
                existence = self.background_masks.to(background).clone()
                existence[changed] = masks[tier]
                return render(background, camera, self.scene.degree, self.scene.white_background, existence)
            report = verify_group(tile, changed, projected, self.targets[0], full_renderer,
                lambda_s=self.reference['lambda_s'], d_ref=self.reference['d_ref'])
            local_values = exact_values(tile, changed, projected, self.targets[0],
                                       self.reference['lambda_s'], self.reference['d_ref'])
            report['relation_exact_values'] = assert_close('relation float64 labels versus full rasterizer',
                local_values, torch.tensor(report['full_image_difference'], device=self.device, dtype=torch.float64),
                atol=5e-11/float(self.reference['d_ref']), rtol=2e-5)
            report.update(source=source, batch=batch, block=row, codec_checks=codec_checks)
            reports.append(report)
        expansion = verify_expansion_probe(self.device, self.scene.white_background, render)
        return dict(version=1, status='passed', backend='cuda', scope='two_groups_trial0_view0',
                    identity=self.identity(), baseline=baseline_check, groups=reports, expansion=expansion,
                    full_render_count=1+4*len(reports)+expansion['full_render_count'])


class RelationContextTeacher:
    """Fixed background bank with at most one context's captures on the device.

    Context teachers share noise seeds so differences do not arise from a new
    noise realization. Decoded CPU backgrounds and small labels survive switches.
    """

    def __init__(self, model, scene, cameras, reference, contexts, **kwargs):
        names = [str(context['name']) for context in contexts]
        if names != ['q1','q2','q3','mixed']:
            raise ValueError('fixed background contexts must be ordered q1, q2, q3, mixed')
        self.teachers = [RelationTeacher(model, scene, cameras, reference,
            background_tiers=context['tiers'], context_id=index, context_name=context['name'], **kwargs)
            for index,context in enumerate(contexts)]
        if any(not bool((teacher.background_tiers==tier).all())
               for tier,teacher in enumerate(self.teachers[:3],1)):
            raise ValueError('named q1/q2/q3 requires the matching uniform background')
        self.active_context = None

    def identity(self):
        return dict(version=2, background_protocol='four_fixed_contexts_v1',
                    contexts=[teacher.identity() for teacher in self.teachers],
                    context_weights=[1./len(self.teachers)]*len(self.teachers))

    @property
    def cached_groups(self):
        return sum(len(teacher._labels) for teacher in self.teachers)

    def _activate(self,context):
        if type(context) is not int or not 0 <= context < len(self.teachers):
            raise ValueError('context index outside fixed background bank')
        if context != self.active_context:
            if self.active_context is not None:
                self.teachers[self.active_context].release_device()
            self.active_context = context
        return self.teachers[context]

    def collect(self,sources,step=0,context=0):
        return self._activate(context).collect(sources,step)

    def release_device(self):
        for teacher in self.teachers:teacher.release_device()
        self.active_context = None

    def verify(self):
        reports=[]
        for context in range(len(self.teachers)):
            reports.append(self._activate(context).verify())
        self.release_device()
        return dict(version=2,status='passed',backend='cuda',identity=self.identity(),
                    contexts=reports,full_render_count=sum(r['full_render_count'] for r in reports))
