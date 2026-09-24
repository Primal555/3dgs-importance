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


def codec_batch(model, features, q, snr, kind, geometry, seed_weight, return_metrics=False,
                compute_auxiliary=True):
    choices = F.one_hot(q, 4).to(features.dtype)
    pred, seed, _ = model.forward_tier_batches(features, features[..., :3], choices, snr, kind)
    keep = q > 0
    pred, seed, target = pred[keep], seed[keep], features[keep]
    if len(pred) == 0:
        result = (to_raw(pred, geometry, model), pred.sum())
        return (*result, {'geometry': pred.sum()}) if return_metrics else result
    if compute_auxiliary:
        auxiliary, terms = reconstruction_loss(pred, target, geometry, model, seed, seed_weight, return_terms=True)
    else:
        # Render-only training must not even evaluate the historical parameter
        # objective (zero times an invalid auxiliary could still produce NaN).
        auxiliary, terms = pred.sum() * 0, {}
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
        return codec_batch(model, f, q, snr, kind, geometry, seed_weight, return_metrics=True,
                           compute_auxiliary=attr_weight != 0 or gradient_observer is not None)

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
    streamed = mode == 'replay' and hasattr(distortion_fn, 'backward_scene')
    # A multiview task can accumulate dL/d(scene) one camera at a time, freeing
    # each rasterizer graph before the next camera. The codec is replayed once.
    distortion = distortion_fn.backward_scene(scene) if streamed else distortion_fn(scene)
    loss = distortion + attr_weight * aux_loss
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite render loss")
    timer.mark("render_forward_seconds")

    if mode == "checkpoint":
        loss.backward()
        timer.mark("combined_backward_seconds")
    else:
        if not streamed:
            distortion.backward()
        upstream = scene.grad.detach()
        if not torch.isfinite(upstream).all():
            raise RuntimeError('nonfinite rendered-scene gradient; no optimizer step performed')
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
    if mode == 'replay':
        values['scene_gradient_norms'] = {'xyz': float(upstream[:, :3].norm()),
                                         'attributes': float(upstream[:, 3:].norm())}
    values.update({f"{key}_loss": float(value / total_count) for key, value in metric_sums.items()})
    if profile and device.type == "cuda":
        values["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
        values["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2 ** 20
    return loss.detach(), values
