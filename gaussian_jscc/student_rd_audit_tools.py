"""Paired hard-map evaluation and shared-policy update diagnostics."""
import itertools
import math
import torch
from torch.nn import functional as F
from .student_policy_model import hard_tiers
from .optimization import preserved_rng
from .modular_checkpoint import restore_rng
from .budgeted_allocation_kernel import raw_batches


def split_action_maps(before, after, ids):
    before, after = [x.detach().cpu() for x in before], [x.detach().cpu() for x in after]
    old, new = hard_tiers(*before), hard_tiers(*after)
    keep = torch.where(new > 0, before[1].argmax(1)+1, 0)
    positive = torch.where(old > 0, after[1].argmax(1)+1, 0)
    evaluated = torch.zeros(len(old), dtype=torch.bool); evaluated[ids.cpu()] = True
    return dict(baseline=old, full=new, keep_only=keep, positive_only=positive,
                positive_evaluated=torch.where(evaluated, positive, old),
                positive_unseen=torch.where(~evaluated, positive, old))


def transition_stats(before, after, ids):
    before, after = [x.detach().cpu() for x in before], [x.detach().cpu() for x in after]
    old, new = hard_tiers(*before), hard_tiers(*after)
    evaluated = torch.zeros(len(old), dtype=torch.bool); evaluated[ids.cpu()] = True
    positive_changed = (old > 0) & (new > 0) & (old != new)
    positive_head_changed = (old > 0) & (before[1].argmax(1) != after[1].argmax(1))
    keep_margin = (before[0].softmax(1)[:, 1]-.5).abs()
    sorted_p = before[1].softmax(1).topk(2, dim=1).values
    tier_margin = sorted_p[:, 0]-sorted_p[:, 1]
    def mean(values, mask): return float(values[mask].mean()) if mask.any() else None
    return dict(changed_points=int((old != new).sum()), changed_fraction=float((old != new).float().mean()),
        deleted=int(((old > 0) & (new == 0)).sum()), restored=int(((old == 0) & (new > 0)).sum()),
        positive_evaluated_changes=int((positive_changed & evaluated).sum()),
        positive_unseen_changes=int((positive_changed & ~evaluated).sum()),
        positive_change_scope='full map; retained in both before and after',
        positive_only_map_evaluated_changes=int((positive_head_changed & evaluated).sum()),
        positive_only_map_unseen_changes=int((positive_head_changed & ~evaluated).sum()),
        positive_evaluated_population=int(((old > 0) & evaluated).sum()),
        positive_unseen_population=int(((old > 0) & ~evaluated).sum()),
        transition_counts=torch.bincount(4*old+new, minlength=16).reshape(4, 4).tolist(),
        changed_keep_initial_margin=mean(keep_margin, (old > 0) != (new > 0)),
        changed_positive_initial_margin=mean(tier_margin, positive_changed),
        keep_probability_change_mean=float((after[0].softmax(1)-before[0].softmax(1)).abs().mean()),
        tier_probability_change_mean=float((after[1].softmax(1)-before[1].softmax(1)).abs().mean()))


def take_step(student, optimizer, losses, clip):
    """Inspect branch gradients, then execute the ordinary summed-loss Adam step."""
    named = list(student.named_parameters()); parameters = [p for _, p in named]
    grads = {}
    for key, loss in losses.items():
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        grads[key] = [torch.zeros_like(p, device='cpu') if g is None else g.detach().cpu()
                      for p, g in zip(parameters, values)]
    groups = dict(all=list(range(len(named))),
        shared=[i for i, (n, _) in enumerate(named) if not n.startswith(('keep_head.', 'tier_head.'))],
        keep_head=[i for i, (n, _) in enumerate(named) if n.startswith('keep_head.')],
        tier_head=[i for i, (n, _) in enumerate(named) if n.startswith('tier_head.')])
    summary = {}
    def dot(a, b, indices): return sum(float((a[i].double()*b[i].double()).sum()) for i in indices)
    for group, indices in groups.items():
        norms = {key: math.sqrt(dot(g, g, indices)) for key, g in grads.items()}
        cosine = {f'{a}:{b}': dot(grads[a], grads[b], indices)/(norms[a]*norms[b])
            if norms[a]*norms[b] > 0 else None for a, b in itertools.combinations(grads, 2)}
        summary[group] = dict(norm=norms, cosine=cosine)
    before = [p.detach().cpu().clone() for p in parameters]
    optimizer.zero_grad(set_to_none=True); sum(losses.values()).backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    delta = [p.detach().cpu()-old for p, old in zip(parameters, before)]
    return dict(gradients=summary, grad_norm_before_clip=float(norm),
        gradient_dot_parameter_step={key: dot(g, delta, groups['all']) for key, g in grads.items()},
        parameter_step_norm=math.sqrt(dot(delta, delta, groups['all'])))


@torch.no_grad()
def paired_decode(model, scene, q, snr, rng):
    """Replay kernel decode order; q0 shadow calls preserve RNG and need not run."""
    device = next(model.parameters()).device; parts = []; q = q.to(device)
    with preserved_rng(device):
        restore_rng(rng)
        for features, source_ids in zip(scene.groups, scene.group_ids):
            f, ids = features.to(device), source_ids.to(device)
            valid = ids >= 0
            choices = q[ids.clamp_min(0)].masked_fill(~valid, 0)
            y = model.semantic_encoder(f, f[..., :3], choices > 0)
            actual = model.forward_tier_batches(f, f[..., :3], F.one_hot(choices, 4).to(f), snr, 'awgn',
                                                paired_noise=True, _semantic_y=y)[0]
            raw = raw_batches(actual, scene.geometry, model)
            parts.append(torch.cat((raw[valid], (choices[valid] > 0).to(raw)[:, None]), 1))
    return torch.cat(parts)


@torch.no_grad()
def paired_evaluate(model, scene, q, cameras, snr, rng, distortion, meter, reference):
    from .rendering import render
    decoded = paired_decode(model, scene, q, snr, rng)
    cost = meter.details(q); result = {}
    for group, views in cameras.items():
        rows = []
        for camera in views:
            image = render(decoded[:, :-1], camera, scene.degree, scene.white_background, decoded[:, -1])
            source = scene.reference.get(camera, decoded.device)
            components = distortion.components(image, source)
            rows.append(dict(view=str(camera.image_name), mse=float(components[0]),
                structure=float(components[1]), distortion=float(distortion.combine(components)),
                psnr=float(-10*components[0].clamp_min(1e-12).log10())))
        d = sum(r['distortion'] for r in rows)/len(rows)
        result[group] = dict(distortion=d, normalized_distortion=d/reference['d_ref'],
            normalized_cost=cost['total_channel_uses']/reference['c_ref'],
            source_psnr=sum(r['psnr'] for r in rows)/len(rows), per_view=rows,
            rd=d/reference['d_ref']+reference['beta']*cost['total_channel_uses']/reference['c_ref'])
    return dict(evaluation=result, cost=cost)


def discrete_prediction(before, after, keep_costs, ids, costs, inverse):
    """Current-gate tangent plus inclusion-weighted positive action differences."""
    old, new = before.cpu(), after.cpu(); ids = ids.cpu()
    signal = (keep_costs[:, 1]-keep_costs[:, 0]).detach().double().cpu()
    keep = float(((new.gt(0).double()-old.gt(0).double())*signal).sum())
    active = (old[ids] > 0) & (new[ids] > 0)
    rows = torch.arange(len(ids))[active]; c = costs.detach().double().cpu()
    tier = float(((c[rows, new[ids][active]-1]-c[rows, old[ids][active]-1])*inverse[active].cpu()).sum())
    return dict(keep=keep, positive=tier, total=keep+tier,
        scope='single-action gate/tier estimate with inverse-inclusion weighting; interactions unmodeled')
