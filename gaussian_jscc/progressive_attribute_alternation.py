"""Ordering and sample offsets for two passes over the original post-training budgets."""
import torch

from .progressive_attribute_channel import EIGHT_RATES
from .position_delivery import delivered_positions
from .progressive_attribute_evaluation import layout_for
from .progressive_attribute_experiment import clean_response_loss


def alternating_schedule(awgn, finite, scenes):
    """Counts are per scene; reference_step addresses the original uninterrupted trainer."""
    if awgn % 8 or finite % 8:
        raise ValueError('each half must contain equal numbers of visits to all four layouts')
    phases = []
    end = 0

    def add(name, lane, visits, reference):
        nonlocal end
        phases.append(dict(name=name, lane=lane, start_step=end, end_step=end+visits*scenes,
                           reference_step=reference*scenes))
        end += visits*scenes

    tier_half = awgn//8
    joint_half = (awgn+finite)//2
    for half in range(2):
        add(f'joint_{half+1}', 'joint', joint_half, half*joint_half)
        for tier in (1, 2, 3):
            reference = (tier-1)*2*tier_half + half*tier_half
            add(f'q{tier}_pass_{half+1}', f'q{tier}', tier_half, reference)
    add('mixed', 'mixed', 2*tier_half, 6*tier_half)
    return phases


def refinement_noise_states(train, visits_per_tier, device):
    """Start each tier at its noise position in the sequential refinement control.

    transmit draws a full fixed-width Gaussian tensor per padded batch, at every
    positive layout. A complete sweep over training scenes thus advances Philox
    by the same offset each time. CPU generators are advanced by actual draws.
    """
    generator = torch.Generator(device=device)
    initial = torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state()
    generator.set_state(initial)
    shapes = [(*f.shape[:-1], EIGHT_RATES[-1], 2) for scene in train for f in scene.groups]

    def draw_sweep():
        for shape in shapes:
            torch.randn(shape, generator=generator, device=device, dtype=train[0].groups[0].dtype)

    if device.type == 'cuda':
        offset = generator.get_offset()
        draw_sweep()
        stride = generator.get_offset()-offset
        generator.set_offset(offset)
    states = {'joint': initial}
    for lane in ('q1', 'q2', 'q3', 'mixed'):
        states[lane] = generator.get_state()
        if lane != 'mixed':
            if device.type == 'cuda':
                generator.set_offset(generator.get_offset()+visits_per_tier*stride)
            else:
                for _ in range(visits_per_tier):
                    draw_sweep()
    return states


@torch.no_grad()
def validation_response(model, teacher, scene, cached, settings):
    """Compare the same Morton blocks, layouts, directions and noise at every phase boundary."""
    device = next(model.parameters()).device
    model.eval()
    model.set_scene_geometry(scene.geometry)
    teacher.set_scene_geometry(scene.geometry)
    centers = torch.stack([c.camera_center for c in scene.val_cameras]).to(device)
    seed = settings.seed+1000
    blocks = [(i, j) for i, batch in enumerate(cached) for j in range(len(batch))]
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(seed))[:64].tolist()
    selected = [blocks[k] for k in order]
    result = {}
    for layout in (1, 2, 3, 'mixed'):
        layouts = layout_for(scene, layout, settings.seed+700001, layers=3)
        result[str(layout)] = {}
        for path in ('awgn', 'finite_noiseless'):
            generator = torch.Generator(device=device).manual_seed(seed)
            total, count = 0., 0
            for i, j in selected:
                data = cached[i][j:j+1].to(device)
                features, y = data[..., :59], data[..., 59:]
                q = layouts[i][j:j+1].to(device)
                active = q > 0
                pred, _, _ = model.forward_cached(features, y, q, settings.snr,
                    noiseless=path == 'finite_noiseless', generator=generator)
                target = teacher.decode(y, active, delivered_positions(features[..., :3], q, model.cfg))
                error = clean_response_loss(model, scene, pred[active], target[active], centers, seed)
                n = int(active.sum())
                total += float(error)*n
                count += n
            result[str(layout)][path] = total/count
    return dict(seed=seed, blocks=selected, directions='4 validation camera centers + 4 sphere',
                active_gaussians=count, local_response_mse=result)
