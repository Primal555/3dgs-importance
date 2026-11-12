"""Dense keep credit with paired, probability-weighted finite residuals."""
import time
import torch
from torch.nn import functional as F
from .student_policy_feedback import UniformFeedback
from .budgeted_allocation_kernel import raw_batches


def sampling_probabilities(keep_probability, influence, uniform_share=.5):
    """Half global coverage, half visible influence near the hard gate boundary."""
    p = keep_probability.detach().cpu().double()
    impact = influence.detach().cpu().double().abs()
    if p.ndim != 1 or impact.shape != p.shape or not len(p):
        raise ValueError('invalid keep sampling inputs')
    if not torch.isfinite(p).all() or not torch.isfinite(impact).all() or ((p < 0)|(p > 1)).any():
        raise ValueError('nonfinite/out-of-range keep sampling inputs')
    if not 0 < uniform_share <= 1: raise ValueError('global sampling support must be positive')
    priority = (4*p*(1-p))*impact.sqrt()
    priority = priority/priority.sum() if priority.sum() > 0 else torch.full_like(p, 1/len(p))
    return uniform_share/len(p)+(1-uniform_share)*priority


def corrected_signal(approximate, exact, ids, multiplicity, probabilities, draws, mode):
    """Correct a SUM objective. Duplicate draws share a render but keep their mass."""
    base = approximate.detach().clone()
    ids = ids.to(base.device)
    if mode == 'replacement':
        base[ids] = exact.to(base)
    elif mode == 'weighted_residual':
        weights = multiplicity.to(base)/(draws*probabilities.to(base))
        residual = exact.to(base)-base[ids]
        base[ids] += weights.reshape((-1,)+(1,)*(base.ndim-1))*residual
    else:
        raise ValueError('unknown keep correction mode')
    return base


class KeepResidualFeedback(UniformFeedback):
    def __init__(self, ids, rounds, finite_samples, seed, *, mode, keep_draws=64,
                 uniform_share=.5, holdout_draws=16):
        super().__init__(ids, rounds, finite_samples, seed)
        if keep_draws < 1 or holdout_draws < 0: raise ValueError('invalid keep label budget')
        self.mode, self.keep_draws = mode, keep_draws
        self.uniform_share, self.holdout_draws = uniform_share, holdout_draws
        self.keep_rng = torch.Generator().manual_seed(seed+101)
        self.holdout_rng = torch.Generator().manual_seed(seed+102)
        self.check_this_step = False
        self.execution_verified = False
        self.lookup = torch.empty(len(self.counts), 4, dtype=torch.long)
        offset = 0
        for batch, block_ids in enumerate(ids):
            valid = block_ids >= 0
            row_starts = valid.sum(1).cumsum(0)-valid.sum(1)
            rows, cols = valid.nonzero(as_tuple=True)
            self.lookup[block_ids[valid]] = torch.stack((torch.full_like(rows, batch), rows, cols,
                                                        offset+row_starts[rows]), 1)
            offset += int(valid.sum())
        self.last_correction = {}; self.holdout = []

    def select_rounds(self, q, influence, probabilities, step):
        self.keep_sampling = sampling_probabilities(1-probabilities[:, 0], influence, self.uniform_share)
        return super().select_rounds(q, influence, probabilities, step)

    @torch.no_grad()
    def correct_components(self, model, features, id_batches, cache, geometry, task, scene, scheduler,
                           tier_ids, locations, components, rates, keep_parts, keep_tiers, table,
                           d_ref, price, c_ref, side_proxy, snr, kind, *, execution=None, **kwargs):
        from .structured_allocation_finite import correct_components
        if execution is None: raise ValueError('paired fast execution required')
        # The original sampler budget is retained: 32 gives 24 positive checks.
        records, stats, exact, keep_exact = correct_components(model, features, id_batches, cache,
            geometry, task, scene, self, tier_ids, locations, components, rates, keep_parts, keep_tiers,
            table, d_ref, price, c_ref, side_proxy, snr, kind, external_keep=True, execution=execution)
        started = time.perf_counter()
        draws = torch.multinomial(self.keep_sampling, self.keep_draws, replacement=True, generator=self.keep_rng)
        selected, counts = draws.unique(sorted=True, return_counts=True)
        # Held-out actions are uniform over uncorrected points and never supervise this update.
        available = torch.ones(len(self.counts), dtype=torch.bool); available[selected] = False
        remaining = available.nonzero().flatten()
        holdout = torch.empty(0, dtype=torch.long)
        if self.check_this_step and self.holdout_draws and len(remaining):
            holdout = remaining[torch.randint(len(remaining), (self.holdout_draws,), generator=self.holdout_rng)]
        all_ids = torch.cat((selected, holdout.unique(sorted=True)))
        labels = self._measure(model, features, id_batches, cache, geometry, task, scene, all_ids,
                               keep_tiers, execution, snr, kind, d_ref)
        approx = keep_parts[selected.to(keep_parts.device)].detach().clone()
        measured = torch.stack([labels[int(i)]['parts'] for i in selected]).to(keep_parts)
        probability = self.keep_sampling[selected]
        corrected = corrected_signal(keep_parts, measured, selected, counts, probability, self.keep_draws, self.mode)
        keep_parts.copy_(corrected); keep_exact[selected] = True
        unit_side = side_proxy['xyz_per_retained']+side_proxy.get('scale_per_retained', 0.)
        raw_rates = torch.tensor(model.cfg.rates, dtype=torch.float64)
        weights = counts.double()/(self.keep_draws*probability)
        self.last_correction = dict(ids=selected, multiplicity=counts, probability=probability,
            approximate=self._combine(approx, task.distortion.lambda_s).cpu(),
            exact=self._combine(measured, task.distortion.lambda_s).cpu(), weights=weights)
        per_draw_residual = ((self.last_correction['exact']-self.last_correction['approximate'])/probability).repeat_interleave(counts)
        for j, point in enumerate(selected.tolist()):
            label = labels[point]; sign = label['sign']
            rate = price*(raw_rates[keep_tiers[point]]+unit_side)/c_ref
            predicted = float(self._combine(approx[j], task.distortion.lambda_s)+rate)*sign
            truth = float(self._combine(measured[j], task.distortion.lambda_s)+rate)*sign
            records.append(dict(source_row=point, feedback_kind='keep', old_tier=label['old'],
                new_tier=label['new'], first_order_delta_rd=predicted, measured_delta_rd=truth,
                rate_delta_cost=float(rate)*sign, sign_agreement=predicted*truth > 0,
                selection_probability=float(probability[j]), multiplicity=int(counts[j]),
                residual_weight=float(weights[j]) if self.mode == 'weighted_residual' else 1.,
                applied_to_training=True, correction_mode=self.mode,
                cost_scope='same current hard-tier + amortized side marginal used by dense keep training'))
            self.finite_counts[point] += 1
        self.holdout = []
        for point in holdout.tolist():
            label = labels[point]
            rate = price*float(raw_rates[keep_tiers[point]]+unit_side)/c_ref
            self.holdout.append(dict(source_row=point, exact_keep_cost=float(
                self._combine(label['parts'], task.distortion.lambda_s))+rate,
                approximate_keep_cost=float(self._combine(keep_parts[point], task.distortion.lambda_s))+rate,
                population=len(remaining)))
        stats.update(keep_correction_mode=self.mode, keep_draws=self.keep_draws,
            keep_unique_renders=len(selected), holdout_unique_renders=len(all_ids)-len(selected),
            holdout_draws=len(holdout), keep_sampling_min=float(self.keep_sampling.min()),
            residual_weight_max=float(weights.max()), residual_weight_mean=float(weights.mean()),
            keep_residual_rmse=float((self.last_correction['exact']-self.last_correction['approximate']).square().mean().sqrt()),
            residual_sum_estimate=float(per_draw_residual.mean()),
            residual_sum_standard_error=float(per_draw_residual.std()/self.keep_draws**.5) if self.keep_draws > 1 else None,
            finite_keep_corrections=len(selected), finite_corrections=len(records),
            keep_exact_seconds=time.perf_counter()-started,
            keep_batch_execution_verified=self.execution_verified)
        return records, stats, exact, keep_exact

    @torch.no_grad()
    def _measure(self, model, features, id_batches, cache, geometry, task, scene, points,
                 keep_tiers, execution, snr, kind, d_ref):
        """Re-decode independent original blocks in batches, reuse paired noise and base scene."""
        device = scene.device; changed = scene.detach().clone(); result = {}; noises = {}
        for start in range(0, len(points), execution.candidate_batch):
            subset = points[start:start+execution.candidate_batch]
            inputs, alternatives, noise_rows, descriptors = [], [], [], []
            for point in subset.tolist():
                batch, row, col, offset = self.lookup[point].tolist()
                q, _, y, _, rng = cache[batch]
                f = features[batch].to(device)
                if batch not in noises: noises[batch] = execution.paired_noise(model, f, rng, snr, kind)
                old = int(q[row, col]); new = 0 if old else int(keep_tiers[point])
                alt = q[row:row+1].clone(); alt[0, col] = new
                inputs.append(f[row:row+1]); alternatives.append(alt); noise_rows.append(noises[batch][row:row+1])
                descriptors.append((point, batch, row, col, offset, old, new))
            f = torch.cat(inputs); alt = torch.cat(alternatives)
            pred = model.forward_tier_batches(f, f[..., :3], F.one_hot(alt, 4).to(f), snr, kind,
                                              paired_noise=True, _channel_noise=torch.cat(noise_rows))[0]
            execution.finite_calls += 1
            if not self.execution_verified:
                maximum = 0.
                checks = [0]
                checks += [i for i, item in enumerate(descriptors) if (item[5] == 0) != (descriptors[0][5] == 0)][:1]
                for j in checks:
                    _, batch, row, col, _, _, new = descriptors[j]
                    q, _, y, _, rng = cache[batch]
                    expected = execution.intervention(model, features[batch].to(device), q, row, col, new, y, rng, snr, kind)
                    torch.testing.assert_close(pred[j:j+1], expected, atol=2e-5, rtol=2e-4)
                    maximum = max(maximum, float((pred[j:j+1]-expected).abs().max()))
                self.execution_verified = True
                self.execution_max_error = maximum
            raws = raw_batches(pred, geometry, model)
            for j, (point, batch, row, col, offset, old, new) in enumerate(descriptors):
                valid = (id_batches[batch][row] >= 0).to(device); count = int(valid.sum())
                changed[offset:offset+count] = torch.cat((raws[j, valid], (alt[j, valid] > 0).to(raws)[:, None]), -1)
                sign = 1 if old == 0 else -1
                parts = task.delta_components(changed)/d_ref*sign
                result[point] = dict(parts=parts.detach().cpu(), old=old, new=new, sign=sign)
                changed[offset:offset+count] = scene.detach()[offset:offset+count]
        return result

    def state_dict(self):
        return dict(super().state_dict(), keep_residual=dict(mode=self.mode, keep_draws=self.keep_draws,
            uniform_share=self.uniform_share, holdout_draws=self.holdout_draws,
            keep_rng=self.keep_rng.get_state(), holdout_rng=self.holdout_rng.get_state()))

    def load_state_dict(self, state):
        super().load_state_dict(state)
        saved = state.get('keep_residual')
        if saved is None: return  # A fork starts from the original UniformFeedback checkpoint.
        for name in ('mode', 'keep_draws', 'uniform_share', 'holdout_draws'):
            if getattr(self, name) != saved[name]: raise ValueError('keep residual resume settings changed')
        self.keep_rng.set_state(saved['keep_rng']); self.holdout_rng.set_state(saved['holdout_rng'])
