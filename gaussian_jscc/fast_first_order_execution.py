"""Batched counterfactual execution with original blocks and paired AWGN."""
import torch
from torch.nn import functional as F
from .codec import channel
from .optimization import preserved_rng
from .pointnet_context import allocation_geometry_reuse
from .budgeted_allocation_kernel import raw_batches, set_rng, timing


class FastFirstOrderExecution:
    def __init__(self, candidate_batch=128):
        if type(candidate_batch) is not int or candidate_batch < 1:
            raise ValueError('candidate batch must be a positive integer')
        self.candidate_batch = candidate_batch
        self.reset()

    def reset(self):
        self.candidate_calls = self.finite_calls = self.geometry_builds = self.geometry_hits = 0

    @staticmethod
    def paired_noise(model, f, state, snr, kind):
        if kind != 'awgn': raise ValueError('fast first-order execution requires AWGN')
        with preserved_rng(f.device):
            set_rng(state, f.device)
            zeros = f.new_zeros((*f.shape[:2], 2*model.cfg.rates[-1]))
            return channel(zeros.reshape(-1, 2), snr, kind).reshape_as(zeros)

    @torch.no_grad()
    def candidates(self, model, f, q, columns, snr, kind, geometry, *, semantic_y, actual, actual_rng, timings):
        variants, descriptors, result = [], [], [[] for _ in columns]
        rows = torch.arange(len(q), device=q.device)
        for i, cols in enumerate(columns):
            cols = cols.to(q.device); active = cols >= 0
            if not active.any(): continue
            current = q[rows, cols.clamp_min(0)]
            for shift in (1, 2):
                tier = (current-1+shift)%3+1
                alt = q.clone(); alt[rows[active], cols[active]] = tier[active]
                variants.append(alt); descriptors.append((i, tier.masked_fill(~active, -1), active))
        if not variants: return result
        noise = self.paired_noise(model, f, actual_rng, snr, kind)
        baseline = raw_batches(actual, geometry, model)
        per_call = max(1, self.candidate_batch//len(q))
        with allocation_geometry_reuse(model, len(q)) as cache:
            for start in range(0, len(variants), per_call):
                group = variants[start:start+per_call]; repeat = len(group)
                inputs = f.repeat(repeat, 1, 1); alt = torch.cat(group)
                with timing(timings, 'candidate_decode_seconds'):
                    pred = model.forward_tier_batches(inputs, inputs[..., :3], F.one_hot(alt, 4).to(f), snr, kind,
                        paired_noise=True, _semantic_y=semantic_y.repeat(repeat, 1, 1),
                        _channel_noise=noise.repeat(repeat, 1, 1))[0]
                self.candidate_calls += 1
                for j, raw in enumerate(raw_batches(pred, geometry, model).split(len(q))):
                    i, tier, active = descriptors[start+j]
                    delta = (raw-baseline)*(q > 0)[..., None]*active[:, None, None]
                    result[i].append((tier, delta))
            self.geometry_builds += cache.builds; self.geometry_hits += cache.hits
        return result

    @torch.no_grad()
    def intervention(self, model, f, q, row, col, new, y, state, snr, kind):
        """Reconstruct the affected original block, including all its neighbors."""
        alt = q[row:row+1].clone(); old = int(alt[0, col]); alt[0, col] = new
        noise = self.paired_noise(model, f, state, snr, kind)[row:row+1]
        inputs = f[row:row+1]
        options = {'_semantic_y': y[row:row+1]} if old > 0 and new > 0 else {}
        pred = model.forward_tier_batches(inputs, inputs[..., :3], F.one_hot(alt, 4).to(f), snr, kind,
            paired_noise=True, _channel_noise=noise, **options)[0]
        self.finite_calls += 1
        return pred

    def diagnostics(self):
        return dict(execution_mode='batched_first_order_v1', candidate_batch=self.candidate_batch,
                    candidate_codec_calls=self.candidate_calls, finite_block_codec_calls=self.finite_calls,
                    candidate_geometry_builds=self.geometry_builds, candidate_geometry_hits=self.geometry_hits)

    @torch.no_grad()
    def verify(self, model, scene, snr, out):
        """Small paired execution check on the actual device, before allocation."""
        from .budgeted_allocation_kernel import mixed_point_candidates, rng_state
        from .modular_experiment import json_write
        device = next(model.parameters()).device
        if max(len(group) for group in scene.groups) > self.candidate_batch:
            raise ValueError('candidate batch must cover at least one original codec batch')
        with preserved_rng(device):
            f = scene.groups[0][:2].to(device)
            ids = scene.group_ids[0][:2].to(device)
            q = (torch.arange(f.shape[1], device=device)[None, :].expand(len(f), -1)%3+1).clone()
            q[ids < 0] = 0
            columns = []
            for row in range(len(q)):
                valid = (ids[row] >= 0).nonzero().flatten()
                columns.append(int(valid[0]) if len(valid) else -1)
                if len(valid) > 1: q[row, valid[-1]] = 0
            cols = torch.tensor(columns, device=device)
            y = model.semantic_encoder(f, f[..., :3], q > 0)
            state = rng_state(device)
            actual = model.forward_tier_batches(f, f[..., :3], F.one_hot(q, 4).to(f), snr, 'awgn',
                                               paired_noise=True, _semantic_y=y)[0]
            expected = list(mixed_point_candidates(model, f, q, cols, snr, 'awgn', scene.geometry,
                semantic_y=y, actual=actual, actual_rng=state, timings=None))
            got = self.candidates(model, f, q, [cols], snr, 'awgn', scene.geometry,
                semantic_y=y, actual=actual, actual_rng=state, timings=None)[0]
            maximum = 0.
            for (tiers, delta), (new_tiers, new_delta) in zip(expected, got):
                torch.testing.assert_close(tiers, new_tiers)
                torch.testing.assert_close(delta, new_delta, rtol=2e-4, atol=2e-5)
                maximum = max(maximum, float((delta-new_delta).abs().max()))
            # Both directions of keep-set change must regenerate the semantic y.
            row = next(i for i, c in enumerate(columns) if c >= 0)
            targets = [(columns[row], 0)]
            closed = ((q[row] == 0) & (ids[row] >= 0)).nonzero().flatten()
            if len(closed): targets.append((int(closed[0]), 2))
            for col, new in targets:
                alt = q.clone(); alt[row, col] = new
                with preserved_rng(device):
                    set_rng(state, device)
                    baseline = model.forward_tier_batches(f, f[..., :3], F.one_hot(alt, 4).to(f),
                                                         snr, 'awgn', paired_noise=True)[0][row:row+1]
                single = self.intervention(model, f, q, row, col, new, y, state, snr, 'awgn')
                torch.testing.assert_close(single, baseline, rtol=2e-4, atol=2e-5)
                maximum = max(maximum, float((single-baseline).abs().max()))
        json_write(out/'execution_check.json', dict(status='passed', device=str(device),
            max_absolute_error=maximum, rtol=2e-4, atol=2e-5,
            scope='sampled positive alternatives and keep toggles; original paired noise'))
        print(f'Execution check passed on {device}: max absolute difference={maximum:.3g}', flush=True)
        self.reset()
