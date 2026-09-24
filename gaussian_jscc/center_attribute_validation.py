"""Clean reconstruction and center-only/12-bit diagnostic comparisons."""
from pathlib import Path
import math
import torch
from .data import to_scene
from .center_attribute_codec import center_loss
from .attribute_objective import attribute_objective
from .render_validation import append_json
from .render_objective import image_metrics


@torch.no_grad()
def validate_centers(model, blocks, indices, geometry, args):
    device = next(model.parameters()).device
    total, squared, distance, count = 0., 0., [], 0
    block_rows = []
    for index in indices:
        f = blocks[index].to(device)[None]
        active = torch.ones(f.shape[:2], device=device, dtype=torch.bool)
        xyz = model.learned.centers(f[..., :3], active)[0]
        n = f.shape[1]
        total += float(center_loss(xyz, f[0, :, :3], geometry, args.center_smoothing))*n
        delta = (xyz.double()-f[0, :, :3].double())*geometry.span.to(device).double()
        squared += float(delta.square().sum())
        block_rows.append({'block': int(index), 'points': n, 'sse': float(delta.square().sum()),
                           'world_rmse': float(delta.square().mean().sqrt())})
        distance.append(delta.norm(dim=-1).cpu())
        count += n
    distances = torch.cat(distance)
    return {'center_loss': total/count, 'world_rmse': math.sqrt(squared/(count*3)),
            'distance_p50_world': float(distances.median()),
            'distance_p95_world': float(torch.quantile(distances, .95)), 'points': count,
            'blocks': block_rows,
            'largest_two_blocks_sse_fraction': sum(sorted([r['sse'] for r in block_rows], reverse=True)[:2])/max(squared, 1e-30)}


@torch.no_grad()
def validate_attributes(model, blocks, indices, geometry, args):
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(args.seed+73019)
    directions = torch.randn(args.attribute_views, 3, generator=generator).to(device)
    totals, count = {}, 0
    for index in indices:
        f = blocks[index].to(device)[None]
        pred = model.reconstruct_clean(f)[0]
        loss, stats, _ = attribute_objective(pred, f[0], geometry, model,
                                            args.attribute_views, directions)
        n = f.shape[1]
        for key, value in {'loss': float(loss), **stats}.items():
            totals[key] = totals.get(key, 0.)+value*n
        count += n
    return {**{key: value/count for key, value in totals.items()},
            'points': count, 'blocks': len(indices), 'direction_seed': args.seed+73019}


def save_panel(directory, images):
    from PIL import Image, ImageDraw
    directory.mkdir(parents=True, exist_ok=True)
    converted = []
    for name, tensor in images.items():
        pixels = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).round().astype('uint8')
        image = Image.fromarray(pixels)
        image.save(directory/f'{name}.png')
        converted.append((name, image))
    width, height = converted[0][1].size
    panel = Image.new('RGB', (width*len(converted), height+24), 'white')
    draw = ImageDraw.Draw(panel)
    for i, (name, image) in enumerate(converted):
        panel.paste(image, (width*i, 24))
        draw.text((width*i+4, 4), name, fill='black')
    panel.save(directory/'comparison.png')


@torch.no_grad()
def validate_render(model, batches, raw, geometry, cameras, reference, args, step, out, images=False):
    from .rendering import render
    device = next(model.parameters()).device
    parts = []
    for features, q in batches:
        f, active = features.to(device), q.to(device) > 0
        pred = model.reconstruct_clean(f, active)
        parts.append(to_scene(pred[active], geometry, model).cpu())
    full = torch.cat(parts)
    del parts
    # Source attributes here are diagnostics ONLY, never optimization input.
    center_only = torch.cat((full[:, :3], raw[:, 3:]), -1)
    unit = geometry.normalize(raw[:, :3])
    qxyz = geometry.denormalize((unit*4095).round()/4095)
    quantized12 = torch.cat((qxyz, raw[:, 3:]), -1)
    variants = {'full': full, 'center_only': center_only, 'quantized12': quantized12}
    view_images = [{
        'photo': camera.original_image[:3].cpu(), 'source': reference.get(camera, device).cpu()
    } for camera in cameras]
    for name, scene in variants.items():
        scene = scene.to(device)
        for i, camera in enumerate(cameras):
            view_images[i][name] = render(scene, camera, model.cfg.sh_degree, args.white_background).cpu()
        del scene
    views = []
    for index, (camera, values) in enumerate(zip(cameras, view_images)):
        row = {'view': str(getattr(camera, 'image_name', index)),
               'reference_photo': image_metrics(values['source'], values['photo'])}
        for name in variants:
            row[name] = {target: image_metrics(values[name], values[target]) for target in ('source', 'photo')}
        views.append(row)
        if images:
            save_panel(Path(out)/'images'/f'{step:06d}'/f'view_{index:02d}', values)
    result = {'views': views, 'diagnostic_only': ['center_only', 'quantized12'],
              'no_communication': True}
    for name in variants:
        result[name] = {f'{target}_{metric}': sum(v[name][target][metric] for v in views)/len(views)
                        for target in ('source', 'photo') for metric in ('mse', 'psnr', 'ssim', 'l1')}
    result['center_gap_db'] = result['quantized12']['source_psnr']-result['center_only']['source_psnr']
    return result


def validate(model, blocks, heldout, batches, raw, geometry, cameras, reference, args, state, out, images=False, fitted_probe=()):
    from .optimization import preserved_rng
    was_training = model.training
    try:
        with preserved_rng(next(model.parameters()).device):
            model.eval()
            centers = validate_centers(model, blocks, heldout, geometry, args)
            fitted_centers = validate_centers(model, blocks, fitted_probe, geometry, args) if fitted_probe else None
            attributes = validate_attributes(model, blocks, heldout, geometry, args) if state['phase'] != 'center' else None
            render_due = state['phase'] != 'attribute' or images or state['phase_step'] == 0
            rendered = validate_render(model, batches, raw, geometry, cameras, reference, args,
                                       state['step'], out, images) if cameras and render_due else None
    finally:
        model.train(was_training)
    result = {'step': state['step'], 'phase': state['phase'], 'phase_step': state['phase_step'],
              'centers': centers, 'fitted_centers': fitted_centers, 'attributes': attributes, 'render': rendered,
              'block_scope': 'heldout in A/B; C rendering trains all Gaussians; render views remain heldout'}
    append_json(Path(out)/'validation.jsonl', result)
    text = f'validation {state["step"]} {state["phase"]}: center world RMSE={centers["world_rmse"]:.6g}'
    if attributes:
        text += f'; attribute loss={attributes["loss"]:.6g}'
    if rendered:
        text += (f'; full PSNR={rendered["full"]["source_psnr"]:.3f}, '
                 f'center-only={rendered["center_only"]["source_psnr"]:.3f}, '
                 f'12bit={rendered["quantized12"]["source_psnr"]:.3f}, gap={rendered["center_gap_db"]:.3f}dB')
    print(text, flush=True)
    return result


def plot_run(out):
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    charts = out/'charts'
    charts.mkdir(exist_ok=True)
    read = lambda name: [json.loads(line) for line in (out/name).read_text(encoding='utf-8').splitlines() if line.strip()]
    validation = read('validation.jsonl')
    losses = read('loss.jsonl') if (out/'loss.jsonl').exists() else []
    phases = [p for p in ('center', 'attribute', 'joint') if any(r['phase'] == p for r in losses)]
    if phases:
        fig, axes = plt.subplots(2, len(phases), squeeze=False, figsize=(5*len(phases), 7))
        for col, phase in enumerate(phases):
            rows = [r for r in losses if r['phase'] == phase]
            x = [r['step'] for r in rows]
            axes[0, col].plot(x, [r['loss'] for r in rows])
            label = {'center': 'world-center distance', 'attribute': 'local attribute loss', 'joint': 'image MSE'}[phase]
            axes[0, col].set_title(phase+' / '+label)
            if phase == 'attribute':
                for term in ('shape', 'appearance'):
                    axes[0, col].plot(x, [r.get('terms', {}).get(term, float('nan')) for r in rows], label=term+' weighted')
                axes[0, col].legend(fontsize=7)
            for module in rows[0]['module_grad_norms']:
                axes[1, col].plot(x, [r['module_grad_norms'][module] for r in rows], label=module)
            axes[1, col].set_yscale('symlog', linthresh=.01)
            axes[1, col].legend(fontsize=7)
            for axis in axes[:, col]:
                axis.grid(alpha=.2)
                axis.set_xlabel('Global step')
        fig.tight_layout()
        fig.savefig(charts/'training_by_phase.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for phase in ('center', 'attribute', 'joint'):
        rows = [r for r in validation if r['phase'] == phase]
        axes[0].plot([r['step'] for r in rows], [r['centers']['world_rmse'] for r in rows], label=phase)
    axes[0].set_title('Center world RMSE (blocks trained only in C)')
    rendered = [r for r in validation if r['render']]
    for col, metric in ((1, 'source_psnr'), (2, 'source_ssim')):
        for route in ('full', 'center_only', 'quantized12'):
            axes[col].plot([r['step'] for r in rendered], [r['render'][route][metric] for r in rendered], label=route)
        axes[col].set_title(metric+' / fixed heldout views')
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=.2)
        axis.set_xlabel('Global step')
    fig.tight_layout()
    fig.savefig(charts/'validation.png', dpi=150)
    plt.close(fig)
    guarded = [r for r in losses if r.get('stats', {}).get('step_guard')]
    if guarded:
        x = [r['step'] for r in guarded]
        guards = [r['stats']['step_guard'] for r in guarded]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(x, [g['scale'] for g in guards], linewidth=.6)
        axes[0, 0].set_title('Joint accepted step scale (0 = rejected)')
        for key in ('loss_before', 'loss_after'):
            axes[0, 1].plot(x, [g[key] for g in guards], linewidth=.6, label=key)
        axes[0, 1].set_title('Same-training-batch distance loss')
        axes[0, 1].legend()
        for key, label in (('accepted', 'rejected'), ('momentum_restarted', 'momentum restart')):
            total, values = 0, []
            for i, g in enumerate(guards):
                total += (not g[key]) if key == 'accepted' else g[key]
                values.append(total/(i+1))
            axes[1, 0].plot(x, values, label=label)
        axes[1, 0].set_title('Cumulative fraction')
        axes[1, 0].legend()
        axes[1, 1].plot(x, [g['attempts'][0]['directional_derivative'] for g in guards], linewidth=.6)
        axes[1, 1].axhline(0, color='black', linewidth=.5)
        axes[1, 1].set_title('Original Adam direction: negative = downhill')
        for axis in axes.flat:
            axis.set_xlabel('Training step')
            axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(charts/'center_step_guard.png', dpi=150)
        plt.close(fig)
    soft = [r for r in losses if r.get('stats', {}).get('soft_update')]
    if soft:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        x = [r['step'] for r in soft]
        for name in soft[0]['stats']['soft_update']['groups']:
            entries = [r['stats']['soft_update']['groups'][name] for r in soft]
            axes[0, 0].plot(x, [e['lr'] for e in entries], label=name)
            axes[0, 1].plot(x, [e['scale'] for e in entries], linewidth=.6, label=name)
            for key in ('relative_proposal', 'relative_actual_update'):
                axes[1, 0].plot(x, [e[key] for e in entries], linewidth=.6, label=name+'/'+key)
            for key in ('limited', 'zero_update'):
                count, fraction = 0, []
                for i, e in enumerate(entries):
                    count += e[key]
                    fraction.append(count/(i+1))
                axes[1, 1].plot(x, fraction, label=name+'/'+key)
        for axis, title in zip(axes.flat, ('Scheduled LR', 'Proposal scale (no loss approval)',
                                         'Update norm / parameter norm', 'Cumulative fraction')):
            axis.set_title(title)
            axis.set_xlabel('Center step')
            axis.grid(alpha=.2)
            axis.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(charts/'center_soft_updates.png', dpi=150)
        plt.close(fig)
    effort = [r for r in losses if r.get('stats', {}).get('cumulative_center_work')]
    if effort:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        points = {r['step']: r['stats']['cumulative_center_work']['processed_points'] for r in effort}
        points[effort[0]['step']-1] = 0
        axes[0].plot([points[r['step']] for r in effort], [r['loss'] for r in effort], linewidth=.6)
        axes[0].set_title('Center loss vs sampled points (including repeats)')
        checked = [r for r in validation if r['phase'] == 'center' and r['step'] in points and r['render']]
        axes[1].plot([points[r['step']] for r in checked],
                     [r['render']['center_only']['source_psnr'] for r in checked], marker='o')
        axes[1].set_title('Center-only PSNR vs sampled points')
        for axis in axes:
            axis.set_xlabel('Cumulative sampled points (not unique points)')
            axis.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(charts/'center_training_effort.png', dpi=150)
        plt.close(fig)
    rows = [r for r in validation if r.get('attributes')]
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for phase in ('attribute', 'joint'):
            selected = [r for r in rows if r['phase'] == phase]
            x = [r['step'] for r in selected]
            axes[0].plot(x, [r['attributes']['loss'] for r in selected], label=phase)
            for term in ('attribute_logcov_mse', 'attribute_response_mse'):
                axes[1].plot(x, [r['attributes'][term] for r in selected], label=phase+'/'+term)
        axes[0].set_title('Fixed-block attribute validation loss')
        axes[1].set_title('Unweighted components (different scales)')
        for axis in axes:
            axis.legend(fontsize=6)
            axis.grid(alpha=.2)
            axis.set_xlabel('Global step; blocks heldout only in A/B')
        fig.tight_layout()
        fig.savefig(charts/'attribute_validation.png', dpi=150)
        plt.close(fig)
