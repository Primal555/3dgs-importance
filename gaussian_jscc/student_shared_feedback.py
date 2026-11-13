"""Shared paired interventions and unbiased keep/tier residual estimators."""
import time
import torch
from torch.nn import functional as F
from .student_keep_feedback import KeepResidualFeedback
from .student_policy_feedback import UniformFeedback
from .budgeted_allocation_kernel import raw_batches


def stratified_draws(strata, priority, draws, generator):
    """Estimate a population SUM: each draw in stratum h has weight N_h/n_h."""
    strata = strata.cpu().long(); priority = priority.cpu().double()
    levels, inverse, sizes = strata.unique(sorted=True, return_inverse=True, return_counts=True)
    if draws < len(levels): raise ValueError('global draws must cover every nonempty stratum')
    score = torch.zeros(len(levels), dtype=torch.float64).scatter_add_(0, inverse, priority)
    if not torch.isfinite(score).all() or (score < 0).any(): raise ValueError('invalid sampling priority')
    if score.sum() == 0: score = sizes.double()
    # A count-proportional component prevents one stratum consuming the budget.
    share = .5*sizes/sizes.sum()+.5*score/score.sum()
    allocation = share*(draws-len(levels)); counts = allocation.floor().long()+1
    left = draws-int(counts.sum())
    counts[(allocation-allocation.floor()).argsort(descending=True, stable=True)[:left]] += 1
    samples, weights = [], []
    for h, n in enumerate(counts.tolist()):
        members = (inverse == h).nonzero().flatten()
        samples.append(members[torch.randint(len(members), (n,), generator=generator)])
        weights.append(torch.full((n,), len(members)/n, dtype=torch.float64))
    return torch.cat(samples), torch.cat(weights), dict(levels=levels.tolist(),
        population=sizes.tolist(), draws=counts.tolist())


def combine_keep_weights(global_ids, global_weights, joint_ids, joint_weights, retained, alpha):
    """Global estimate covers q0; two convexly combined estimates cover retained points."""
    if not 0 <= alpha <= 1: raise ValueError('invalid estimator mixture')
    ids, inverse = torch.cat((global_ids, joint_ids)).unique(sorted=True, return_inverse=True)
    global_weights, joint_weights = global_weights.double(), joint_weights.double()
    scale = torch.where(retained[global_ids], global_weights.new_tensor(alpha), global_weights.new_tensor(1.))
    scaled = global_weights*scale
    weights = torch.zeros(len(ids), dtype=torch.float64)
    weights.scatter_add_(0, inverse, torch.cat((scaled, (1-alpha)*joint_weights)))
    return ids, weights


class SharedResidualFeedback(KeepResidualFeedback):
    def __init__(self, ids, rounds, finite_samples, seed, *, mode, keep_draws=128,
                 uniform_share=.5, holdout_draws=16, full_groups=24):
        super().__init__(ids, rounds, finite_samples, seed, mode='weighted_residual',
            keep_draws=keep_draws, uniform_share=uniform_share, holdout_draws=holdout_draws)
        if full_groups < 1 or keep_draws-full_groups < 16:
            raise ValueError('reserve at least 16 global keep draws in addition to full groups')
        self.full_groups = full_groups
        self.joint_rng = torch.Generator().manual_seed(seed+307)
        self.residual_rms = torch.ones(16, dtype=torch.float64)
        self.residual_seen = torch.zeros(16, dtype=torch.bool)
        self.last_positive_residual = None; self.extra_stats = {}
        self.verified_modes = set()

    def select_rounds(self, q, influence, probabilities, step):
        p = 1-probabilities[:, 0].detach().cpu().double()
        self.retained = q.cpu() > 0
        self.influence = influence.cpu().double().abs()
        # Source blocks follow the existing scene ordering. Four ranges provide spatial coverage.
        block = self.lookup[:, 3]
        region = (4*block//max(1, int(block.max())+1)).clamp_max(3)
        boundary = ((p-.5).abs() < .25).long()
        self.strata = self.retained.long()*8+boundary*4+region
        self.keep_sampling = torch.full_like(p, 1/len(p))  # diagnostics compatibility only
        current = (4*p*(1-p))*self.influence.sqrt()
        observed = self.residual_rms[self.residual_seen]
        fallback = observed.median() if len(observed) else torch.tensor(1., dtype=torch.float64)
        scale = torch.where(self.residual_seen[self.strata], self.residual_rms[self.strata], fallback)
        # Historical errors affect only sample allocation, never the current target values.
        self.keep_priority = current*scale
        conditional = probabilities[:, 1:].detach().cpu().double()
        conditional = conditional/conditional.sum(-1, keepdim=True).clamp_min(1e-300)
        self.tier_priority = (1-conditional.square().sum(-1))*self.influence.sqrt()
        return UniformFeedback.select_rounds(self, q, influence, probabilities, step)

    @torch.no_grad()
    def correct_components(self, model, features, id_batches, cache, geometry, task, scene, scheduler,
                           tier_ids, locations, components, rates, keep_parts, keep_tiers, table,
                           d_ref, price, c_ref, side_proxy, snr, kind, *, execution=None, **kwargs):
        if execution is None: raise ValueError('paired original-block execution required')
        started = time.perf_counter(); n = len(self.counts)
        group_draws = self.full_groups if len(tier_ids) else 0
        global_draws = self.keep_draws-group_draws
        global_ids, global_weights, strata_record = stratified_draws(
            self.strata, self.keep_priority, global_draws, self.keep_rng)
        positions = torch.empty(0, dtype=torch.long); counts = torch.empty(0, dtype=torch.long)
        cond_prob = torch.empty(0, dtype=torch.float64)
        if group_draws:
            priority = self.tier_priority[tier_ids]
            priority = priority/priority.sum() if priority.sum() else torch.full_like(priority, 1/len(priority))
            distribution = self.uniform_share/len(tier_ids)+(1-self.uniform_share)*priority
            draws = torch.multinomial(distribution, group_draws, replacement=True, generator=self.joint_rng)
            positions, counts = draws.unique(sorted=True, return_counts=True)
            cond_prob = distribution[positions]
        joint_ids = tier_ids[positions].cpu()
        conditional_weights = counts.double()/(max(1, group_draws)*cond_prob)
        # The outer candidate inclusion applies exactly once in each population estimator.
        joint_weights = conditional_weights*self.inverse_inclusion[joint_ids]
        alpha = global_draws/self.keep_draws if group_draws else 1.
        selected, weights = combine_keep_weights(global_ids, global_weights, joint_ids, joint_weights,
                                                  self.retained, alpha)
        available = torch.ones(n, dtype=torch.bool); available[selected] = False
        remaining = available.nonzero().flatten()
        holdout = torch.empty(0, dtype=torch.long)
        if self.check_this_step and self.holdout_draws and len(remaining):
            holdout = remaining[torch.randint(len(remaining), (self.holdout_draws,), generator=self.holdout_rng)]
        requests = [(i, 0 if self.retained[i] else int(keep_tiers[i]))
                    for i in torch.cat((selected, holdout)).tolist()]
        requests.extend((i, k) for i in joint_ids.tolist() for k in range(4))
        measured, measure_stats = self._measure_states(model, features, id_batches, cache, geometry,
            task, scene, requests, execution, snr, kind, d_ref)
        approximate_keep = keep_parts[selected.to(keep_parts.device)].clone()
        signs = torch.where(self.retained[selected], -1., 1.).double()
        true_keep = torch.stack([measured[i, 0 if self.retained[i] else int(keep_tiers[i])]
                                 for i in selected.tolist()])*signs[:, None]
        keep_residual = true_keep.to(keep_parts)-approximate_keep
        keep_parts[selected.to(keep_parts.device)] += keep_residual*weights.to(keep_parts)[:, None]
        combine = lambda x: self._combine(x, task.distortion.lambda_s)
        self.last_correction = dict(ids=selected, weights=weights,
            approximate=combine(approximate_keep).cpu(), exact=combine(true_keep).cpu())
        exact = torch.zeros(len(tier_ids), 3, dtype=torch.bool)
        keep_exact = torch.zeros(n, dtype=torch.bool); keep_exact[selected] = True
        positive_residual = torch.zeros_like(components)
        records = []
        unit_side = side_proxy['xyz_per_retained']+side_proxy.get('scale_per_retained', 0.)
        raw_rates = torch.tensor(model.cfg.rates, dtype=torch.float64)
        for j, i in enumerate(selected.tolist()):
            old = int(cache[int(self.lookup[i, 0])][0][int(self.lookup[i, 1]), int(self.lookup[i, 2])])
            new = 0 if old else int(keep_tiers[i]); sign = float(signs[j])
            rate = price*(float(raw_rates[keep_tiers[i]])+unit_side)/c_ref*sign
            records.append(dict(source_row=i, feedback_kind='keep', old_tier=old, new_tier=new,
                first_order_delta_rd=float(combine(approximate_keep[j]))*sign+rate,
                measured_delta_rd=float(combine(true_keep[j]))*sign+rate, rate_delta_cost=rate,
                residual_weight=float(weights[j]), applied_to_training=True,
                cost_scope='current hard-tier and amortized digital-side marginal'))
        for j, position in enumerate(positions.tolist()):
            i = int(tier_ids[position]); old = int(keep_tiers[i])
            truth = torch.stack([measured[i, k] for k in (1, 2, 3)]).to(components)
            approximate = components[position].clone()
            positive_residual[position] = (truth-approximate)*conditional_weights[j].to(components)
            exact[position] = True
            self.finite_counts[i] += 2
            for k in (1, 2, 3):
                if k == old: continue
                records.append(dict(source_row=i, feedback_kind='positive', old_tier=old, new_tier=k,
                    first_order_delta_rd=float(combine(approximate[k-1])+rates[position, k-1]),
                    measured_delta_rd=float(combine(truth[k-1])+rates[position, k-1]),
                    rate_delta_cost=float(rates[position, k-1]), conditional_probability=float(cond_prob[j]),
                    multiplicity=int(counts[j]), inverse_candidate_inclusion=float(self.inverse_inclusion[i]),
                    conditional_residual_weight=float(conditional_weights[j]), applied_to_training=True,
                    cost_scope='payload and block-scale marginal shared with candidate estimator'))
        components.add_(positive_residual)
        self.last_positive_residual = combine(positive_residual-positive_residual.mean(1, keepdim=True)).cpu()
        self.finite_counts[selected] += 1
        for record in records:
            record['sign_agreement'] = (record['first_order_delta_rd'] < 0) == (record['measured_delta_rd'] < 0)
        self.holdout = []
        for i in holdout.tolist():
            sign = -1 if self.retained[i] else 1
            true = combine(measured[i, 0 if self.retained[i] else int(keep_tiers[i])])*sign
            rate = price*(float(raw_rates[keep_tiers[i]])+unit_side)/c_ref
            self.holdout.append(dict(source_row=i, exact_keep_cost=float(true)+rate,
                approximate_keep_cost=float(combine(keep_parts[i]))+rate, population=len(remaining)))
        # Per-stratum dispersion is only a proposal statistic for subsequent windows.
        for level in self.strata[selected].unique().tolist():
            error = combine(keep_residual).cpu()[self.strata[selected] == level]
            rms = error.square().mean().sqrt().clamp_min(1e-12)
            self.residual_rms[level] = .8*self.residual_rms[level]+.2*rms if self.residual_seen[level] else rms
            self.residual_seen[level] = True
        positive_rows = [r for r in records if r['feedback_kind'] == 'positive']
        keep_rows = [r for r in records if r['feedback_kind'] == 'keep']
        def action_metrics(rows):
            return dict(checked_actions=len(rows), sign_errors=sum(not r['sign_agreement'] for r in rows),
                mean_wrong_action_cost=(sum(abs(r['measured_delta_rd']) for r in rows if not r['sign_agreement'])/len(rows)
                                        if rows else None),
                max_wrong_action_cost=max((abs(r['measured_delta_rd']) for r in rows if not r['sign_agreement']), default=0.),
                scope='unique current checked actions; same marginal-cost proxy; descriptive, not population-weighted')
        self.extra_stats = dict(shared_teacher=measure_stats, global_keep_draws=global_draws,
            full_group_draws=group_draws, full_group_unique_points=len(joint_ids),
            keep_strata=strata_record, retained_global_mixture=alpha,
            positive_sign_errors=sum(not r['sign_agreement'] for r in positive_rows),
            positive_checked_actions=len(positive_rows),
            positive_false_beneficial_cost=sum(r['measured_delta_rd'] for r in positive_rows
                if r['first_order_delta_rd'] < 0 < r['measured_delta_rd']),
            positive_actions=action_metrics(positive_rows), keep_actions=action_metrics(keep_rows),
            keep_residual_component_rmse=keep_residual.square().mean(0).sqrt().cpu().tolist(),
            gradient_target='sum of current-context single-action gradients; two-stage inclusion weighting',
            shared_label_dependence='four-state differences share one background; not independent samples')
        stats = dict(keep_draws=self.keep_draws, keep_unique_renders=len(selected),
            holdout_unique_renders=len(set(holdout.tolist())), holdout_draws=len(holdout),
            keep_exact_seconds=time.perf_counter()-started,
            keep_residual_rmse=float(combine(keep_residual).square().mean().sqrt()),
            residual_weight_max=float(weights.max()), residual_weight_mean=float(weights.mean()),
            residual_sum_estimate=float((combine(keep_residual).cpu()*weights).sum()),
            residual_sum_standard_error=None, keep_sampling_min=None,
            finite_positive_corrections=len(positive_rows), finite_keep_corrections=len(selected),
            finite_corrections=len(records), keep_batch_execution_verified=self.execution_verified)
        return records, stats, exact, keep_exact

    @torch.no_grad()
    def _measure_states(self, model, features, id_batches, cache, geometry, task, scene, requests,
                        execution, snr, kind, d_ref):
        """One current-context cache; q0 toggles re-encode the affected original block."""
        result = {}; unique = sorted(set(requests)); device = scene.device
        changed = scene.detach().clone(); inputs_cache = {}; noises = {}; todo = {False: [], True: []}
        for i, k in unique:
            batch, row, col, _ = self.lookup[i].tolist(); old = int(cache[batch][0][row, col])
            if old == k: result[i, k] = torch.zeros(2, dtype=torch.float64); continue
            todo[(old > 0) == (k > 0)].append((i, k))
        calls = 0
        for same_active, pairs in todo.items():
            for start in range(0, len(pairs), execution.candidate_batch):
                group = pairs[start:start+execution.candidate_batch]
                xs, qs, ys, ns, descriptors = [], [], [], [], []
                for i, k in group:
                    batch, row, col, offset = self.lookup[i].tolist()
                    q, _, y, _, rng = cache[batch]
                    if batch not in inputs_cache:
                        inputs_cache[batch] = features[batch].to(device)
                        noises[batch] = execution.paired_noise(model, inputs_cache[batch], rng, snr, kind)
                    alt = q[row:row+1].clone(); old = int(alt[0, col]); alt[0, col] = k
                    xs.append(inputs_cache[batch][row:row+1]); qs.append(alt)
                    ys.append(y[row:row+1]); ns.append(noises[batch][row:row+1])
                    descriptors.append((i, k, batch, row, col, offset, old))
                x = torch.cat(xs); alt = torch.cat(qs)
                options = dict(_semantic_y=torch.cat(ys)) if same_active else {}
                pred = model.forward_tier_batches(x, x[..., :3], F.one_hot(alt, 4).to(x), snr, kind,
                    paired_noise=True, _channel_noise=torch.cat(ns), **options)[0]
                execution.finite_calls += 1; calls += 1
                raw = raw_batches(pred, geometry, model)
                for j, (i, k, batch, row, col, offset, old) in enumerate(descriptors):
                    tag = 'positive' if same_active else ('restore' if old == 0 else 'delete')
                    if tag not in self.verified_modes:
                        q, _, y, _, rng = cache[batch]
                        expected = execution.intervention(model, inputs_cache[batch], q, row, col, k, y, rng, snr, kind)
                        torch.testing.assert_close(pred[j:j+1], expected, atol=2e-5, rtol=2e-4)
                        self.verified_modes.add(tag)
                    valid = (id_batches[batch][row] >= 0).to(device); count = int(valid.sum())
                    changed[offset:offset+count] = torch.cat((raw[j, valid], (alt[j, valid]>0).to(raw)[:, None]), -1)
                    result[i, k] = (task.delta_components(changed)/d_ref).detach().cpu()
                    changed[offset:offset+count] = scene.detach()[offset:offset+count]
        self.execution_verified = bool(self.verified_modes)
        return result, dict(requests=len(requests), unique_states=len(unique),
            baseline_states_reused=len(unique)-sum(map(len, todo.values())),
            alternative_renders=sum(map(len, todo.values())), codec_batches=calls,
            duplicate_requests_reused=len(requests)-len(unique), verified_modes=sorted(self.verified_modes))

    def state_dict(self):
        return dict(super().state_dict(), shared_residual=dict(full_groups=self.full_groups,
            joint_rng=self.joint_rng.get_state(), residual_rms=self.residual_rms.clone(),
            residual_seen=self.residual_seen.clone()))

    def load_state_dict(self, state):
        super().load_state_dict(state)
        saved = state.get('shared_residual')
        if saved is not None:
            if saved['full_groups'] != self.full_groups: raise ValueError('shared group budget changed')
            self.joint_rng.set_state(saved['joint_rng'].cpu())
            self.residual_rms.copy_(saved['residual_rms']); self.residual_seen.copy_(saved['residual_seen'])


def shared_losses(heads, keep_costs, tier_ids, tier_costs, inverse, sampler):
    from .student_keep_controls import losses_for
    losses = losses_for(heads, keep_costs, tier_ids, tier_costs, inverse, sampler)
    p = heads[1][tier_ids.to(heads[1].device)].softmax(-1)
    residual = (p*sampler.last_positive_residual.to(p)*inverse.to(p)[:, None]).sum()
    losses['positive_dense'] = losses.pop('positive')-residual
    losses['positive_residual'] = residual
    return losses
