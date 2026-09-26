"""Training utilities for learned_joint, including honest discrete mask actions."""
from contextlib import contextmanager
import time
import torch
from torch.nn import functional as F
from .training import full_scene_step, codec_batch


def policy_objective(log_probabilities, costs):
    """Independent samples, leave-one-out baseline; no batch-mean bias.

    Each log probability is SUMMED over primitive decisions, not averaged.
    Baselines and costs are detached. >=2 independent masks are required.
    """
    costs = torch.stack([torch.as_tensor(c).detach() for c in costs])
    if len(costs) < 2:
        raise ValueError('score-function estimation requires >=2 independent samples')
    baseline = (costs.sum()-costs)/(len(costs)-1)
    return ((costs-baseline) * torch.stack(log_probabilities)).mean()


def discrete_joint_step(model, mask, feature_batches, id_batches, geometry, snr, kind,
                        distortion_fn, beta=.001, auxiliary_weight=0., samples=2, mode='replay',
                        rate_meter=None, train_codec=True, paired_noise=False):
    """Codec gets conditional backprop; categorical masks get REINFORCE.

    distortion_fn(decoded_rows, retained_original_ids) returns a render task
    distortion. It MUST penalize omission through the complete scene reference,
    not merely compare attributes of the remaining rows.
    Auxiliary losses regularize codec parameters only; dropping points cannot
    lower the mask's reward by deleting their auxiliary losses.
    Payload rate has an exact expectation. With a rate_meter, measured discrete
    XYZ/tier-stream cost enters the score-function reward, NOT a detached-only
    logging term. Both are averaged over SOURCE primitives.
    beta is a Lagrange penalty, NOT a guarantee of a hard total-symbol cap.
    """
    if model.cfg.architecture != 'learned_joint' or samples < 2:
        raise ValueError('requires learned_joint and >=2 independent mask samples')
    device = next(model.parameters()).device
    # Independent mask draws, common full-slot channel noise conditional on the
    # step. RNG isolation prevents channel draws from affecting mask sampling.
    noise_seed = int(torch.randint(2**31-1,()).item()) if paired_noise else None
    @contextmanager
    def noise_context():
        if not paired_noise:
            yield
            return
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            state = torch.Generator(device=device).manual_seed(noise_seed).get_state()
            if device.type=='cuda':
                torch.cuda.set_rng_state(state,device)
            else:
                torch.set_rng_state(state)
            yield

    def paired_forward(f,q):
        return codec_batch(model,f,q,snr,kind,geometry,0,return_metrics=True,
                           compute_auxiliary=auxiliary_weight!=0,paired_noise=True)
    log_probs, costs, all_stats, expectations, compositions, side_costs = [], [], [], [], [], []
    cost_timings, task_timings = [], []
    source_count = sum(int((ids >= 0).sum()) for ids in id_batches)
    for _ in range(samples):
        batches, retained, sample_logs, rates = [], [], [], []
        for features, ids in zip(feature_batches, id_batches):
            ids = ids.to(device)
            valid = ids >= 0
            scores = mask.scores(ids.clamp_min(0), snr)
            distribution = torch.distributions.Categorical(logits=scores)
            q = distribution.sample()
            sample_logs.append(distribution.log_prob(q)[valid].sum())
            rates.append((distribution.probs[valid]*scores.new_tensor(model.cfg.rates)).sum())
            q = torch.where(valid, q, 0)
            batches.append((features, q))
            retained.append(ids[q > 0])
        ids = torch.cat(retained)
        counts = torch.zeros(len(model.cfg.rates),device=device)
        for (_,q), original_ids in zip(batches,id_batches):
            counts += torch.bincount(q[original_ids.to(device)>=0],minlength=len(model.cfg.rates))
        compositions.append(counts)
        log_probs.append(torch.stack(sample_logs).sum())
        expectations.append(torch.stack(rates).sum()/source_count)
        flat_q = torch.cat([q[gi.to(q.device)>=0].cpu() for (_,q),gi in zip(batches,id_batches)])
        cost_started = time.perf_counter()
        measured = rate_meter.details(flat_q) if rate_meter else {}
        side_costs.append(measured.get('allocation_side_uses_per_source_gaussian', 0.))
        cost_timings.append((time.perf_counter()-cost_started, measured.get('position_compression_seconds', 0.),
                             measured.get('tier_map_compression_seconds', 0.)))
        task_started = time.perf_counter()
        if len(ids) and not train_codec:
            with noise_context():
                scene = decode_batches(model, feature_batches, [q for _,q in batches], snr, kind, geometry,paired_noise=paired_noise)
            with torch.no_grad():
                cost = distortion_fn(scene, ids)
            stats = {'retained_gaussians': len(ids), 'render_loss': float(cost)/samples}
        elif len(ids):
            def distortion(scene):
                return distortion_fn(scene, ids)/samples
            if hasattr(distortion_fn, 'backward_scene'):
                def backward_scene(scene):
                    cost = distortion_fn.backward_scene(scene)
                    scene.grad.div_(samples)
                    return cost/samples
                distortion.backward_scene = backward_scene
            with noise_context():
                _, stats = full_scene_step(model, batches, geometry, snr, kind, distortion,
                                           attr_weight=auxiliary_weight/samples, mode=mode,
                                           batch_forward=paired_forward if paired_noise else None)
            cost = next(model.parameters()).new_tensor(stats['render_loss']*samples)
        else:
            # No fabricated q0 XYZ enters the renderer. Empty-scene task loss
            # still supplies a valid score-function cost, with no codec graph.
            with torch.no_grad():
                empty = next(model.parameters()).new_empty((0, 3+model.cfg.attr_dim))
                cost = distortion_fn(empty, ids)
            stats = {'retained_gaussians': 0, 'render_loss': float(cost)/samples}
        costs.append(cost)
        all_stats.append(stats)
        task_timings.append(time.perf_counter()-task_started)
    normalizer = rate_meter.normalizer if rate_meter else model.cfg.rates[-1]
    rewards = [c + beta*s/normalizer for c,s in zip(costs,side_costs)]
    policy = policy_objective(log_probs, rewards)
    mean_rate = torch.stack(expectations).mean()
    rate_loss = beta*mean_rate/normalizer
    (policy+rate_loss).backward()
    total_rate_loss = rate_loss.detach()+beta*sum(side_costs)/samples/normalizer
    mean_aux = sum(v.get('aux_loss', 0) for v in all_stats)/samples
    return torch.stack(costs).mean()+total_rate_loss+auxiliary_weight*mean_aux, {
        'mask_estimator': 'independent-sample leave-one-out REINFORCE',
        'policy_surrogate': float(policy.detach()), 'sample_task_costs': [float(c) for c in costs],
        'expected_symbols_per_gaussian': float(mean_rate.detach()),
        'rate_loss': float(total_rate_loss), 'mask_samples': samples,
        'rate_accounting_seconds': sum(t[0] for t in cost_timings),
        'position_compression_seconds': sum(t[1] for t in cost_timings),
        'tier_map_compression_seconds': sum(t[2] for t in cost_timings),
        'codec_task_seconds': sum(task_timings),
        'sample_side_uses_per_gaussian': side_costs, 'rate_normalizer': normalizer,
        'sample_policy_costs': [float(r) for r in rewards], 'codec_updated': train_codec,
        'paired_mask_channel_noise': paired_noise,
        'expected_payload_plus_sampled_side_uses_per_gaussian': float(mean_rate.detach())+sum(side_costs)/samples,
        'retained_counts': [v['retained_gaussians'] for v in all_stats],
        'sampled_tier_counts': torch.stack(compositions).mean(0).cpu().tolist(),
        'scene_gradient_norms_by_sample': [v.get('scene_gradient_norms') for v in all_stats],
        'render_loss': float(torch.stack(costs).mean()),
        'aux_loss': mean_aux}


@torch.no_grad()
def decode_batches(model, feature_batches, q_batches, snr, kind, geometry, paired_noise=False):
    device = next(model.parameters()).device
    return torch.cat([codec_batch(model, f.to(device), q.to(device), snr, kind, geometry, 0,
                                 compute_auxiliary=False, paired_noise=paired_noise)[0]
                      for f, q in zip(feature_batches, q_batches)])


def layout_schedule(rates):
    """One update per positive prefix, then one per-point mixed update."""
    return tuple(range(1, len(rates))) + (None,)


def hard_layout(ids, uniform=None, drop=.05, tier_count=4):
    if tier_count < 2 or (uniform is not None and not 1 <= uniform < tier_count):
        raise ValueError('uniform tier must be positive and present in the rate table')
    valid = ids >= 0
    q = torch.full_like(ids, uniform) if uniform is not None else torch.randint(1, tier_count, ids.shape, device=ids.device)
    if uniform is None and drop:
        q[torch.rand(ids.shape, device=ids.device) < drop] = 0
    return torch.where(valid, q, 0)
