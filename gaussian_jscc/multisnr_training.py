"""Matched SNR sampling and cached-latent updates for the shared codec."""
import math
import random

import torch
from torch.nn import functional as F

from .data import to_raw
from .modular_training import configure_phase, latent_reconstruction_loss, latent_diagnostics
from .position_delivery import delivered_positions
from .training import full_scene_step

LAYOUTS = ('1', '2', '3', 'mixed')


def schedule(visit, seed, low, high, bins=5):
    """A shuffled Cartesian cycle gives every layout every SNR stratum.

    Counter-based sampling is independent of scene loading, validation, color
    updates and global RNG. The same scene visit resumes with identical inputs.
    """
    if visit < 1 or bins < 1 or not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError('positive visit/bins and finite increasing SNR range required')
    cycle, index = divmod(visit-1, bins*len(LAYOUTS))
    rng = random.Random(seed + 1000003*cycle)
    cells = [(b, q) for b in range(bins) for q in LAYOUTS]
    rng.shuffle(cells)
    b, layout = cells[index]
    jitter = random.Random(seed + 7000003*visit).random()
    return low + (b+jitter)*(high-low)/bins, layout, b


def tiers(active, layout, seed):
    q = (torch.randint(1, 4, active.shape, generator=torch.Generator().manual_seed(seed))
         if layout == 'mixed' else torch.full(active.shape, int(layout), dtype=torch.long))
    return q.to(active.device).masked_fill(~active, 0)


def optimizer_for(model, phase, args):
    lr = args.channel_lr if phase == 'channel_bootstrap' else args.joint_channel_lr
    lrs = dict(channel_encoder=lr, channel_decoder=lr, semantic_decoder=args.decoder_lr)
    optimizer = configure_phase(model, phase, lrs, freeze_semantic=phase == 'channel_bootstrap')
    return optimizer, lrs


def cached_render_step(model, batches, geometry, snr, task, latent_weight):
    """Packed CPU rows contain original features followed by immutable y.

    Full-scene replay preserves the AWGN draw between its forward and VJP.
    Semantic encoding is omitted in both passes; transmitted symbols and the
    semantic decoder still use current trainable parameters.
    """
    model.train(); model.semantic_encoder.eval(); model.set_scene_geometry(geometry)
    width = model.cfg.latent_dim

    def forward(packed, q):
        features, target = packed[..., :-width], packed[..., -width:]
        y, recovered, _ = model.forward_channel_batches(features, features[..., :3],
            F.one_hot(q, 4).to(features), snr, 'awgn', paired_noise=True, _semantic_y=target)
        auxiliary, metrics = latent_reconstruction_loss(y, recovered, q)
        side = delivered_positions(features[..., :3], q, model.cfg)
        pred = model.decode_representation(recovered, q, side)
        return to_raw(pred[q>0], geometry, model), auxiliary, metrics

    loss, stats = full_scene_step(model, batches, geometry, snr, 'awgn', task,
        attr_weight=latent_weight, mode='replay', profile=True, batch_forward=forward)
    metrics = {k[:-5]: stats.pop(k) for k in list(stats) if k.startswith('latent_') and k.endswith('_loss')}
    stats.update(latent_diagnostics(metrics), **task.stats,
        objective='source_ply_rgb_mse+latent_mse', channel_latent_weight=latent_weight)
    return loss, stats
