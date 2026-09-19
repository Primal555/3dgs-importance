"""Training utilities for learned_joint, including honest discrete mask actions."""
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
                        distortion_fn, beta=.001, auxiliary_weight=0., samples=2, mode='direct'):
    """Codec gets conditional backprop; categorical masks get REINFORCE.

    distortion_fn(decoded_rows, retained_original_ids) returns a render task
    distortion. It MUST penalize omission through the complete scene reference,
    not merely compare attributes of the remaining rows.
    Auxiliary losses regularize codec parameters only; dropping points cannot
    lower the mask's reward by deleting their auxiliary losses.
    Rate is the exact expected payload cost, averaged over SOURCE primitives.
    beta is a Lagrange penalty, NOT a guarantee of a hard total-symbol cap.
    """
    if model.cfg.architecture != 'learned_joint' or samples < 2:
        raise ValueError('requires learned_joint and >=2 independent mask samples')
    device = next(model.parameters()).device
    log_probs, costs, all_stats, expectations, compositions = [], [], [], [], []
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
        counts = torch.zeros(4,device=device)
        for (_,q), original_ids in zip(batches,id_batches):
            counts += torch.bincount(q[original_ids.to(device)>=0],minlength=4)
        compositions.append(counts)
        log_probs.append(torch.stack(sample_logs).sum())
        expectations.append(torch.stack(rates).sum()/source_count)
        if len(ids):
            def distortion(scene):
                return distortion_fn(scene, ids)/samples
            if hasattr(distortion_fn, 'backward_scene'):
                def backward_scene(scene):
                    cost = distortion_fn.backward_scene(scene)
                    scene.grad.div_(samples)
                    return cost/samples
                distortion.backward_scene = backward_scene
            _, stats = full_scene_step(model, batches, geometry, snr, kind, distortion,
                                       attr_weight=auxiliary_weight/samples, mode=mode)
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
    policy = policy_objective(log_probs, costs)
    mean_rate = torch.stack(expectations).mean()
    rate_loss = beta*mean_rate/model.cfg.rates[-1]
    (policy+rate_loss).backward()
    mean_aux = sum(v.get('aux_loss', 0) for v in all_stats)/samples
    return torch.stack(costs).mean()+rate_loss.detach()+auxiliary_weight*mean_aux, {
        'mask_estimator': 'independent-sample leave-one-out REINFORCE',
        'policy_surrogate': float(policy.detach()), 'sample_task_costs': [float(c) for c in costs],
        'expected_symbols_per_gaussian': float(mean_rate.detach()),
        'rate_loss': float(rate_loss.detach()), 'mask_samples': samples,
        'retained_counts': [v['retained_gaussians'] for v in all_stats],
        'sampled_tier_counts': torch.stack(compositions).mean(0).cpu().tolist(),
        'scene_gradient_norms_by_sample': [v.get('scene_gradient_norms') for v in all_stats],
        'render_loss': float(torch.stack(costs).mean()),
        'aux_loss': mean_aux}


@torch.no_grad()
def decode_batches(model, feature_batches, q_batches, snr, kind, geometry):
    device = next(model.parameters()).device
    return torch.cat([codec_batch(model, f.to(device), q.to(device), snr, kind, geometry, 0,
                                 compute_auxiliary=False)[0]
                      for f, q in zip(feature_batches, q_batches)])


def hard_layout(ids, uniform=None, drop=.05):
    valid = ids >= 0
    q = torch.full_like(ids, uniform) if uniform is not None else torch.randint(1, 4, ids.shape, device=ids.device)
    if uniform is None and drop:
        q[torch.rand(ids.shape, device=ids.device) < drop] = 0
    return torch.where(valid, q, 0)
