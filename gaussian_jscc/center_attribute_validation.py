"""Clean reconstruction and center-only/12-bit diagnostic comparisons."""
from pathlib import Path
import math
import torch
from .data import to_scene
from .center_attribute_codec import center_loss
from .render_validation import append_json
from .render_objective import image_metrics


@torch.no_grad()
def validate_centers(model, blocks, indices, geometry, args):
    device = next(model.parameters()).device
    total, squared, distance, count = 0., 0., [], 0
    for index in indices:
        f = blocks[index].to(device)[None]
        active = torch.ones(f.shape[:2], device=device, dtype=torch.bool)
        xyz = model.learned.centers(f[..., :3], active)[0]
        n = f.shape[1]
        total += float(center_loss(xyz, f[0, :, :3], geometry, args.center_smoothing))*n
        delta = (xyz.double()-f[0, :, :3].double())*geometry.span.to(device).double()
        squared += float(delta.square().sum())
        distance.append(delta.norm(dim=-1).cpu())
        count += n
    distances = torch.cat(distance)
    return {'center_loss': total/count, 'world_rmse': math.sqrt(squared/(count*3)),
            'distance_p50_world': float(distances.median()),
            'distance_p95_world': float(torch.quantile(distances, .95)), 'points': count}


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


def validate(model, blocks, heldout, batches, raw, geometry, cameras, reference, args, state, out, images=False):
    from .optimization import preserved_rng
    was_training = model.training
    try:
        with preserved_rng(next(model.parameters()).device):
            model.eval()
            centers = validate_centers(model, blocks, heldout, geometry, args)
            rendered = validate_render(model, batches, raw, geometry, cameras, reference, args,
                                       state['step'], out, images) if cameras else None
    finally:
        model.train(was_training)
    result = {'step': state['step'], 'phase': state['phase'], 'phase_step': state['phase_step'],
              'centers': centers, 'render': rendered,
              'block_scope': 'heldout in center phase only; attribute/joint rendering uses all scene Gaussians'}
    append_json(Path(out)/'validation.jsonl', result)
    text = f'validation {state["step"]} {state["phase"]}: center world RMSE={centers["world_rmse"]:.6g}'
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
            axes[0, col].set_title(phase+(' / world-center distance' if phase == 'center' else ' / image MSE'))
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
    axes[0].set_title('Center world RMSE (blocks trained after phase A)')
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
