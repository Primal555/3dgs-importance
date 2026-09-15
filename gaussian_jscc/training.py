"""Batched full-scene training with bounded codec activation memory.

Replay applies the chain rule at the reconstructed-scene tensor. It does not
subsample Gaussians, cache stale predictions, or change the render objective.
Only one batch's codec activations is live during replay backward.
"""

import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import to_raw
from .losses import reconstruction_loss


class PhaseTimer:
    def __init__(self, device, enabled):
        self.device, self.enabled = device, enabled
        self.values = {}
        self.last = self.now()

    def now(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def mark(self, name):
        current = self.now()
        if self.enabled:
            self.values[name] = current - self.last
        self.last = current


def codec_batch(model, features, q, snr, kind, geometry, seed_weight, return_metrics=False):
    choices = F.one_hot(q, 4).to(features.dtype)
    pred, seed, _ = model.forward_tier_batches(features, features[..., :3], choices, snr, kind)
    keep = q > 0
    pred, seed, target = pred[keep], seed[keep], features[keep]
    if len(pred) == 0:
        result = (to_raw(pred, geometry, model), pred.sum())
        return (*result, {}) if return_metrics else result
    auxiliary, terms = reconstruction_loss(pred, target, geometry, model, seed, seed_weight, return_terms=True)
    result = (to_raw(pred, geometry, model), auxiliary)
    return (*result, {key: value.mean() for key, value in terms.items()}) if return_metrics else result


def full_scene_step(model, batches, geometry, snr, kind, distortion_fn,
                    attr_weight=.1, seed_weight=.2, mode="replay", profile=False,
                    batch_forward=None, gradient_observer=None):
    """Compute AND backpropagate one loss; caller clips/steps the optimizer.

    batches is a nonempty list of (padded_features, int64_tiers), each containing
    independent spatial blocks. CPU-resident batches are transferred on demand.
    q0 is permitted for padding/fixed drop maps, not as a learned mask here.
    distortion_fn receives the COMPLETE decoded scene in retained source order.
    An optional replay-only observer receives weighted component objectives on
    each recomputed batch. It may use autograd.grad(retain_graph=True), but must
    not mutate .grad, parameters, or RNG. Summed observations are full-scene
    parameter gradients (the render objective here is its exact VJP surrogate).
    """
    if mode not in ("replay", "checkpoint"):
        raise ValueError("render backward must be replay or checkpoint")
    if gradient_observer is not None and mode != "replay":
        raise ValueError("gradient observer requires replay")
    device = next(model.parameters()).device
    timer = PhaseTimer(device, profile)
    if profile and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows, auxiliary, states, counts = [], [], [], []
    metric_sums = {}

    def forward(f, q):
        if batch_forward is not None:
            return batch_forward(f, q)
        return codec_batch(model, f, q, snr, kind, geometry, seed_weight, return_metrics=True)

    for features, q in batches:
        features, q = features.to(device), q.to(device)
        if mode == "replay":
            # Capture the device RNG immediately before sampling channel noise.
            states.append(torch.cuda.get_rng_state(device) if device.type == "cuda"
                          else torch.get_rng_state())
            with torch.no_grad():
                raw, aux, metrics = forward(features, q)
        else:
            raw, aux, metrics = checkpoint(forward, features, q, use_reentrant=False, preserve_rng_state=True)
        rows.append(raw)
        counts.append(len(raw))
        auxiliary.append(aux * len(raw))
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0) + value.detach() * len(raw)
    total_count = sum(counts)
    if total_count == 0:
        raise ValueError("cannot render an all-dropped scene")
    scene = torch.cat(rows)
    del rows
    if mode == "replay":
        scene.requires_grad_(True)
    aux_loss = torch.stack(auxiliary).sum() / total_count
    timer.mark("codec_forward_seconds")
    distortion = distortion_fn(scene)
    loss = distortion + attr_weight * aux_loss
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite render loss")
    timer.mark("render_forward_seconds")

    if mode == "checkpoint":
        loss.backward()
        timer.mark("combined_backward_seconds")
    else:
        distortion.backward()
        upstream = scene.grad.detach()
        timer.mark("render_backward_seconds")
        # Restore the caller's RNG after replay: recomputation must not consume
        # another set of channel draws or change subsequent tier sampling.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] \
            if device.type == "cuda" else []
        offset = 0
        with torch.random.fork_rng(devices=devices):
            for (features, q), state, count in zip(batches, states, counts):
                if device.type == "cuda":
                    torch.cuda.set_rng_state(state, device)
                else:
                    torch.set_rng_state(state)
                raw, aux, metrics = forward(features.to(device), q.to(device))
                # VJP through the codec, plus the original weighted auxiliary.
                render_objective = (raw * upstream[offset:offset + count]).sum()
                weighted_aux = attr_weight * aux * count / total_count
                if gradient_observer is not None:
                    weighted_geometry = (metrics["geometry"] * model.cfg.geometry_weight
                                         * attr_weight * count / total_count)
                    gradient_observer({"geometry": weighted_geometry,
                                       "attributes": weighted_aux - weighted_geometry,
                                       "render": render_objective})
                objective = render_objective + weighted_aux
                objective.backward()
                offset += count
        timer.mark("codec_replay_backward_seconds")
    values = {"render_loss": float(distortion.detach()), "aux_loss": float(aux_loss.detach()),
              "retained_gaussians": total_count, "codec_batches": len(batches), **timer.values}
    values.update({f"{key}_loss": float(value / total_count) for key, value in metric_sums.items()})
    if profile and device.type == "cuda":
        values["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
        values["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2 ** 20
    return loss.detach(), values


def joint_scene_step(model, mask, batches, geometry, snr, kind, distortion_fn,
                     temperature=1., beta=.01, attr_weight=.1, mode="replay", profile=False):
    """Batched hard-Gumbel codec+mask backward with bounded activation memory.

    batches contain padded (features, ORIGINAL row ids); padding id is -1.
    The replay boundary includes both raw Gaussian parameters and four choices,
    so the renderer's inactive-mask gradient reaches dropped-row logits too.
    RNG snapshots cover BOTH Gumbel selection and channel noise.
    """
    device = next(model.parameters()).device

    def forward(features, ids):
        valid = ids >= 0
        scores = mask.scores(ids.clamp_min(0), snr)
        choices = F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1)
        padding = torch.zeros_like(choices)
        padding[..., 0] = 1
        choices = torch.where(valid[..., None], choices, padding)
        pred, _, active = model.forward_tier_batches(features, features[..., :3], choices, snr, kind)
        pred, target, active = pred[valid], features[valid], active[valid]
        aux, terms = reconstruction_loss(pred, target, geometry, model, active=active, return_terms=True)
        return (torch.cat((to_raw(pred, geometry, model), choices[valid]), -1), aux,
                {key: value.mean() for key, value in terms.items()})

    counts = None

    def distortion(scene):
        nonlocal counts
        counts = scene[:, -4:].detach().sum(0)
        return distortion_fn(scene[:, :-4], scene[:, -3:].sum(-1))

    loss, values = full_scene_step(model, batches, geometry, snr, kind, distortion,
                                   attr_weight=attr_weight, mode=mode, profile=profile,
                                   batch_forward=forward)
    rate_timer = PhaseTimer(device, profile)
    # Separate inexpensive graph: expectation over all source rows, never over
    # a learned retained count. No second codec forward is needed for rate.
    total = sum(int((ids >= 0).sum()) for _, ids in batches)
    rates = next(model.parameters()).new_tensor(model.cfg.rates)
    expectation = []
    for _, ids in batches:
        ids = ids.to(device)
        expectation.append((mask.scores(ids[ids >= 0], snr).softmax(-1) * rates).sum())
    mean_rate = torch.stack(expectation).sum() / total
    rate_penalty = beta * mean_rate / model.cfg.rates[-1]
    rate_penalty.backward()
    rate_timer.mark("rate_backward_seconds")
    values.update(source_gaussians=total, retained_gaussians=int(counts[1:].sum()),
                  sampled_tier_counts=counts.cpu().tolist(),
                  expected_symbols_per_gaussian=float(mean_rate.detach()),
                  rate_loss=float(rate_penalty.detach()), temperature=temperature,
                  **rate_timer.values)
    if profile and device.type == "cuda":
        values["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
        values["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2 ** 20
    return loss + rate_penalty.detach(), values
