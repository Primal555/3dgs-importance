"""Fixed held-out blocks and paired scene renders for separated codec training."""
from pathlib import Path
import math
import torch
from torch.nn.utils.rnn import pad_sequence
from .codec import channel, prefix_mask
from .data import to_scene
from .optimization import preserved_rng
from .render_validation import append_json
from .spatial_response import spatial_response_loss


def paths(model, features, active, snr, kind, communication=True):
    core = model.learned
    y = core.representation_encoder(features, active)
    clean = core.representation_decoder(y, active)
    if not communication:
        return clean, None, y, None
    q = active.long()*3
    symbols = core.channel_encoder(y, q, snr)
    mask = prefix_mask(q.flatten(), model.cfg.rates).reshape_as(symbols)
    received = symbols.new_zeros(symbols.shape).masked_scatter(
        mask, channel(symbols[mask].reshape(-1, 2), snr, kind).flatten())
    recovered = core.channel_decoder(received, q, snr)
    return clean, core.representation_decoder(recovered, active), y, recovered


def objective(pred, source, geometry, model, args, directions, return_components=False):
    axis_mode = getattr(args, 'position_objective', 'scene-scale') == 'teacher-axis'
    result = spatial_response_loss(pred, source, geometry, model, args.local_response_views,
                                   directions=directions, fine_weight=args.spatial_fine_weight,
                                   return_components=axis_mode or return_components)
    if axis_mode:
        from .axis_position import teacher_axis_position
        _, stats, components = result
        position, axis_stats = teacher_axis_position(pred, source, geometry, model, args.axis_floor_world)
        # REPLACE the old position term; no addition of the scene-scale objective.
        components['position'] = position/3
        loss = sum(components.values()).to(pred.dtype)
        stats['legacy_scene_position_diagnostic'] = stats['spatial_position_response']
        for key in list(stats):
            if key.startswith(('spatial_coarse_', 'spatial_fine_', 'spatial_scale_')):
                stats['legacy_diagnostic_'+key] = stats.pop(key)
        stats.update(axis_stats, spatial_position_response=float(position.detach()),
                     spatial_geometry_response=float(position.detach()),
                     position_objective_contribution=float((position/3).detach()))
        return (loss, stats, components) if return_components else (loss, stats)
    return result


@torch.no_grad()
def validate_blocks(model, blocks, indices, geometry, args, step, phase, out):
    device = next(model.parameters()).device
    was_training = model.training
    sums = {name: {} for name in ('clean', 'communication')}
    count = 0
    latent_mse = 0.
    try:
        with preserved_rng(device):
            torch.manual_seed(args.seed+9100)
            model.eval()
            directions = torch.randn(args.local_response_views, 3, device=device)
            for index in indices:
                f = blocks[index].to(device)[None]
                active = torch.ones(f.shape[:2], dtype=torch.bool, device=device)
                clean, comm, y, recovered = paths(model, f, active, args.snr, args.channel)
                n = f.shape[1]
                count += n
                latent_mse += float((y-recovered).square().mean())*n
                for name, pred in (('clean', clean), ('communication', comm)):
                    loss, stats = objective(pred[active], f[active], geometry, model, args, directions)
                    delta = pred[active][:, :3]-f[active][:, :3]
                    delta = delta.double()*geometry.span.to(device).double()
                    stats.update(loss=float(loss), xyz_world_mse=float(delta.square().mean()))
                    # Mean block diagnostics are explicitly not pooled quantiles.
                    for key, value in stats.items():
                        sums[name][key] = sums[name].get(key, 0.)+float(value)*n
    finally:
        model.train(was_training)
    result = {'step': step, 'phase': phase, 'heldout_points': count,
              'heldout_blocks': len(indices), 'channel': args.channel, 'snr': args.snr,
              'latent_mse': latent_mse/count, 'tier': 3,
              'clean_is_internal_not_transmitted': True,
              'communication_symbols_per_gaussian': model.cfg.rates[3],
              'quantile_note': 'shape/position quantiles are point-weighted means of per-block quantiles'}
    for name in sums:
        result[name] = {key: value/count for key, value in sums[name].items()}
        result[name]['xyz_pooled_world_rmse'] = math.sqrt(result[name].pop('xyz_world_mse'))
    append_json(Path(out)/'validation.jsonl', result)
    print(f'validation {step}: clean loss={result["clean"]["loss"]:.5g}, '
          f'communication loss={result["communication"]["loss"]:.5g}; '
          f'XYZ RMSE={result["clean"]["xyz_pooled_world_rmse"]:.5g} / '
          f'{result["communication"]["xyz_pooled_world_rmse"]:.5g}', flush=True)
    return result


def save_images(directory, photo, reference, clean, communication):
    from PIL import Image, ImageDraw
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {'photo': photo, 'source': reference, 'clean': clean, 'communication': communication}
    images = []
    for name, value in tensors.items():
        pixels = (value.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype('uint8')
        image = Image.fromarray(pixels)
        image.save(directory/f'{name}.png')
        images.append(image)
    width, height = images[0].size
    panel = Image.new('RGB', (width*4, height+24), 'white')
    draw = ImageDraw.Draw(panel)
    for index, (name, image) in enumerate(zip(tensors, images)):
        panel.paste(image, (index*width, 24))
        draw.text((index*width+4, 4), name, fill='black')
    panel.save(directory/'comparison.png')


@torch.no_grad()
def validate_images(model, blocks, raw, geometry, cameras, reference, args, step, phase, out):
    from .rendering import render
    from .render_objective import image_metrics
    device = next(model.parameters()).device
    was_training = model.training
    images = {'clean': [], 'communication': []}
    try:
        with preserved_rng(device):
            model.eval()
            # Decode/render one entire scene at a time; keep comparison images on CPU.
            for route in images:
                torch.manual_seed(args.seed+19200)
                parts = []
                for start in range(0, len(blocks), args.blocks_per_batch):
                    group = blocks[start:start+args.blocks_per_batch]
                    f = pad_sequence(group, batch_first=True).to(device)
                    lengths = torch.tensor([len(b) for b in group], device=device)
                    active = torch.arange(f.shape[1], device=device)[None] < lengths[:, None]
                    if route == 'clean':
                        pred = model.reconstruct_clean(f, active)
                    else:
                        core = model.learned
                        q = active.long()*3
                        symbols = core.encode(f, f[..., :3], q, args.snr)
                        mask = prefix_mask(q.flatten(), model.cfg.rates).reshape_as(symbols)
                        received = symbols.new_zeros(symbols.shape).masked_scatter(mask,
                            channel(symbols[mask].reshape(-1, 2), args.snr, args.channel).flatten())
                        pred = core.decode(received, q, args.snr)
                    parts.append(to_scene(pred[active], geometry, model).cpu())
                scene = torch.cat(parts).to(device)
                del parts
                for camera in cameras:
                    images[route].append(render(scene, camera, model.cfg.sh_degree, args.white_background).cpu())
                del scene
            views = []
            for index, camera in enumerate(cameras):
                target = reference.get(camera, device).cpu()
                photo = camera.original_image[:3].cpu()
                clean, comm = images['clean'][index], images['communication'][index]
                entry = {'view': str(getattr(camera, 'image_name', index)),
                         'reference_photo': image_metrics(target, photo)}
                for route, image in (('clean', clean), ('communication', comm)):
                    entry[route] = {'source': image_metrics(image, target), 'photo': image_metrics(image, photo)}
                views.append(entry)
                save_images(Path(out)/'images'/f'{step:06d}'/f'view_{index:02d}', photo, target, clean, comm)
    finally:
        model.train(was_training)
    result = {'step': step, 'phase': phase, 'views': views, 'channel': args.channel,
              'snr': args.snr, 'scope': 'fixed camera validation; full scene includes fitted and held-out blocks'}
    for route in images:
        result[route] = {f'{reference_name}_{key}': sum(v[route][reference_name][key] for v in views)/len(views)
                         for reference_name in ('source', 'photo') for key in ('mse', 'psnr', 'ssim', 'l1')}
    result['communication_psnr_drop'] = result['clean']['source_psnr']-result['communication']['source_psnr']
    append_json(Path(out)/'render_validation.jsonl', result)
    print(f'render {step}: source PSNR clean={result["clean"]["source_psnr"]:.3f}, '
          f'communication={result["communication"]["source_psnr"]:.3f}; '
          f'photo PSNR={result["clean"]["photo_psnr"]:.3f}/{result["communication"]["photo_psnr"]:.3f}', flush=True)
    return result
