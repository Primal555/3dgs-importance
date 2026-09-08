"""Batched full-scene training with bounded codec activation memory.

Replay applies the chain rule at the reconstructed-scene tensor. It does not
subsample Gaussians, cache stale predictions, or change the render objective.
Only one batch's codec activations is live during replay backward.
"""

import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import attribute_loss, to_raw


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


def codec_batch(model, features, q, snr, kind, geometry, seed_weight):
    choices = F.one_hot(q, 4).to(features.dtype)
    pred, seed, _ = model.forward_tier_batches(features, features[..., :3], choices, snr, kind)
    keep = q > 0
    pred, seed, target = pred[keep], seed[keep], features[keep]
    if len(pred) == 0:
        return to_raw(pred, geometry, model), pred.sum()
    auxiliary = attribute_loss(pred, target) + seed_weight * F.smooth_l1_loss(seed, target[:, :3])
    return to_raw(pred, geometry, model), auxiliary


def full_scene_step(model, batches, geometry, snr, kind, distortion_fn,
                    attr_weight=.1, seed_weight=.2, mode="replay", profile=False):
    """Compute AND backpropagate one loss; caller clips/steps the optimizer.

    batches is a nonempty list of (padded_features, int64_tiers), each containing
    independent spatial blocks. CPU-resident batches are transferred on demand.
    q0 is permitted for padding/fixed drop maps, not as a learned mask here.
    distortion_fn receives the COMPLETE decoded scene in retained source order.
    """
    if mode not in ("replay", "checkpoint"):
        raise ValueError("render backward must be replay or checkpoint")
    device = next(model.parameters()).device
    timer = PhaseTimer(device, profile)
    if profile and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows, auxiliary, states, counts = [], [], [], []

    def forward(f, q):
        return codec_batch(model, f, q, snr, kind, geometry, seed_weight)

    for features, q in batches:
        features, q = features.to(device), q.to(device)
        if mode == "replay":
            # Capture the device RNG immediately before sampling channel noise.
            states.append(torch.cuda.get_rng_state(device) if device.type == "cuda"
                          else torch.get_rng_state())
            with torch.no_grad():
                raw, aux = forward(features, q)
        else:
            raw, aux = checkpoint(forward, features, q, use_reentrant=False, preserve_rng_state=True)
        rows.append(raw)
        counts.append(len(raw))
        auxiliary.append(aux * len(raw))
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
                raw, aux = forward(features.to(device), q.to(device))
                # VJP through the codec, plus the original weighted auxiliary.
                objective = (raw * upstream[offset:offset + count]).sum()
                objective = objective + attr_weight * aux * count / total_count
                objective.backward()
                offset += count
        timer.mark("codec_replay_backward_seconds")
    values = {"render_loss": float(distortion.detach()), "aux_loss": float(aux_loss.detach()),
              "retained_gaussians": total_count, "codec_batches": len(batches), **timer.values}
    if profile and device.type == "cuda":
        values["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
        values["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2 ** 20
    return loss.detach(), values
