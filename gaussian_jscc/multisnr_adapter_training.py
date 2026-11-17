"""Matched three-SNR gradient accumulation over one rotating camera window."""
import hashlib
import random

import torch
from torch.nn import functional as F

from .channel_supervision_controls import color_directions
from .local_response import local_response_loss
from .modular_training import latent_reconstruction_loss
from .multisnr_adaptation import noiseless_tier
from .multisnr_training import cached_render_step
from .optimization import preserved_rng
from .position_delivery import delivered_positions
from .render_objective import MultiViewRenderTask
from .successive_q3_refinement import CachedQ3Batches
from .successive_mixed_refinement import CachedMixedBatches


def visit_schedule(seed, scene, visit):
    if visit < 1:
        raise ValueError('positive scene visit required')
    def key(name, index):
        text = f'{seed}:{scene}:{name}:{index}'.encode()
        return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little') % (2**63-1)
    cycle, slot = divmod(visit-1, 4)
    layouts = ['1', '2', '3', 'mixed']
    random.Random(key('layout', cycle)).shuffle(layouts)
    rng = random.Random(key('snr', visit))
    snrs = [5.+(i+rng.random())*10./3 for i in range(3)]
    return dict(layout=layouts[slot], snrs=snrs, noise_seed=key('noise', visit),
                batch_seed=key('batch', visit))


def batches_for(scene, targets, layout, seed, visit):
    return (CachedMixedBatches(scene, targets, seed, visit) if layout == 'mixed'
            else CachedQ3Batches(scene, targets, int(layout)))


def train_step(model, scene, targets, cameras, args, visit):
    """Accumulate the mean of three full-scene objectives, then one local auxiliary.

    The same standard normal draws are replayed at each SNR, scaled by the real
    channel. Each condition frees its render/codec graphs before the next one.
    No parameter update occurs inside this function.
    """
    device = next(model.parameters()).device
    schedule = visit_schedule(args.seed, scene.name, visit)
    batches = batches_for(scene, targets, schedule['layout'], args.seed, visit)
    rows = []
    total = 0.
    for snr in schedule['snrs']:
        task = MultiViewRenderTask(cameras, scene.reference, scene.degree, scene.white_background)
        with preserved_rng(device):
            torch.manual_seed(schedule['noise_seed'])
            loss, stats = cached_render_step(model, batches, scene.geometry, snr, task, args.latent_weight)
        rows.append(dict(snr_db=snr, render_mse=stats['render_loss'], latent_mse=stats['latent_mse'],
                         loss=float(loss), views_per_step=len(cameras)))
        total += float(loss)/3
    # cached_render_step has already backpropagated each complete objective.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(3)
    result = dict(schedule, conditions=rows, render_mse=sum(r['render_mse'] for r in rows)/3,
                  latent_mse=sum(r['latent_mse'] for r in rows)/3,
                  full_scene_recoveries=3, rendered_views=3*len(cameras))
    q = noiseless_tier(visit, args.noiseless_every)
    if q is not None and args.noiseless_weight:
        index = random.Random(schedule['batch_seed']).randrange(len(targets))
        features = scene.groups[index].to(device); y = targets[index].to(device)
        active = scene.group_ids[index].to(device) >= 0
        selected = active.long()*q
        centers = torch.stack([torch.as_tensor(c.camera_center) for c in cameras]).to(device)
        directions, _ = color_directions(features[..., :3][active], scene.geometry, centers,
                                         args.response_directions, schedule['batch_seed'])
        values = []
        for snr in schedule['snrs']:
            _, recovered, _ = model.forward_channel_batches(features, features[..., :3],
                F.one_hot(selected, 4).to(features), snr, 'none', paired_noise=True, _semantic_y=y)
            latent, _ = latent_reconstruction_loss(y, recovered, selected)
            pred = model.decode_representation(recovered, selected,
                    delivered_positions(features[..., :3], selected, model.cfg))
            response, _ = local_response_loss(pred[active], features[active], scene.geometry,
                                              model, directions=directions)
            auxiliary = args.latent_weight*latent + args.response_weight*response
            (args.noiseless_weight/3*auxiliary).backward()
            values.append(float(auxiliary.detach()))
        total += args.noiseless_weight*sum(values)/3
        result.update(noiseless_tier=q, noiseless_batch=index, noiseless_objectives=values)
    result['loss'] = total
    return result
