"""Semantic-only image/attribute metrics and reproducible latent interventions."""
from pathlib import Path

import torch

from .data import to_raw
from .local_response import camera_directions
from .position_delivery import delivered_positions
from .render_objective import image_metrics, spaced_indices
from .semantic_attribute_codec import GROUPS, error_rows, attribute_objective


@torch.no_grad()
def decode_scene(model, scene):
    device = next(model.parameters()).device
    model.eval(); model.set_scene_geometry(scene.geometry)
    rows = []
    for features, ids in zip(scene.groups, scene.group_ids):
        f, active = features.to(device), (ids >= 0).to(device)
        rows.append(to_raw(model(f, active)[active], scene.geometry, model).cpu())
    result = torch.cat(rows)
    if result.shape != scene.raw.shape or not torch.isfinite(result).all():
        raise RuntimeError('invalid semantic reconstruction')
    return result


@torch.no_grad()
def attribute_metrics(model, scene, decoded, cameras):
    device = next(model.parameters()).device
    centers = torch.stack([c.camera_center.cpu() for c in cameras])
    values = {}
    for start in range(0, len(decoded), 8192):
        target = scene.raw[start:start+8192].to(device)
        pred = decoded[start:start+8192].to(device)
        directions = camera_directions(target[:, :3], centers.to(device))
        terms = error_rows(pred, target, model, directions)
        for k, v in terms.items():
            values.setdefault(k, []).append(v.cpu())
    result = {}
    for k, chunks in values.items():
        x = torch.cat(chunks)
        result[k] = {'mse': float(x.mean()), 'rmse': float(x.mean().sqrt()),
                     'p95_squared_error': float(x.quantile(.95)),
                     'p99_squared_error': float(x.quantile(.99))}
    return result


def save_image(path, image):
    from PIL import Image
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pixels = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().numpy()
    Image.fromarray(pixels).save(path)


@torch.no_grad()
def evaluate(model, scene, cameras, perceptual, out, *, interventions=False, images=False):
    from .rendering import render, RenderReference
    device = next(model.parameters()).device
    decoded = decode_scene(model, scene)
    delivered = torch.cat((decoded[:, :3], scene.raw[:, 3:]), -1)
    if not hasattr(scene, '_semantic_xyz_reference'):
        scene._semantic_xyz_reference = RenderReference(delivered, scene.degree, scene.white_background, 'source')
        scene._semantic_reference_metrics = {}
    elif not torch.equal(scene._semantic_xyz_reference.raw, delivered):
        raise RuntimeError('delivered XYZ/reference attributes changed between arms')
    variants = {'semantic': decoded, 'xyz_only': delivered, 'source': scene.raw}
    if interventions:
        for name, (a, b) in GROUPS.items():
            value = decoded.clone()
            value[:, 3+a:3+b] = scene.raw[:, 3+a:3+b]
            variants['restore_'+name] = value
    selected_images = set(spaced_indices(len(cameras), min(2, len(cameras))))
    rows = []
    # A single reconstructed scene per variant; camera graphs never retained.
    for name, value in variants.items():
        raw = value.to(device)
        by_view = []
        for index, camera in enumerate(cameras):
            source = scene.reference.get(camera, device)
            photo = camera.original_image[:3].to(device)
            image = (source if name == 'source' else scene._semantic_xyz_reference.get(camera, device)
                     if name == 'xyz_only' else render(raw, camera, scene.degree, scene.white_background))
            record = {'view': str(camera.image_name)}
            cache_key = (name, id(camera), perceptual is not None)
            cached = scene._semantic_reference_metrics.get(cache_key) if name in ('source', 'xyz_only') else None
            if cached is not None:
                record.update(cached)
            else:
                for reference, target in [('source', source), ('photo', photo)]:
                    metrics = image_metrics(image, target)
                    if perceptual is not None:
                        metrics['lpips'] = perceptual(image, target)
                    record.update({reference+'_'+k: v for k, v in metrics.items()})
                if name in ('source', 'xyz_only'):
                    scene._semantic_reference_metrics[cache_key] = dict(record)
            # Isolates attribute error with exactly the same delivered positions.
            if name == 'semantic':
                target = scene._semantic_xyz_reference.get(camera, device)
                metrics = image_metrics(image, target)
                if perceptual is not None:
                    metrics['lpips'] = perceptual(image, target)
                record.update({'delivered_source_'+k: v for k, v in metrics.items()})
            if images and index in selected_images:
                save_image(Path(out)/'images'/f'{index:03d}_{name}.png', image)
                if name == 'source':
                    save_image(Path(out)/'images'/f'{index:03d}_photo.png', photo)
            by_view.append(record)
        mean = {k: sum(r[k] for r in by_view)/len(by_view) for k in by_view[0] if k != 'view'}
        rows.append(dict(variant=name, views=by_view, **mean))
        del raw
    return dict(scene=scene.name, view_names=[str(c.image_name) for c in cameras], rows=rows,
                attributes=attribute_metrics(model, scene, decoded, cameras),
                communication_cost=None, scope='continuous semantic reconstruction; no channel experiment')


def directions_for(scene, features, active, cameras):
    points = scene.geometry.denormalize(features[active][:, :3])
    return camera_directions(points, torch.stack([c.camera_center.to(points) for c in cameras]))


def latent_diagnostics(model, scene, scales, seed=42):
    """Fixed spatial blocks; perturbation is evaluation-only, not channel noise."""
    device = next(model.parameters()).device
    model.eval(); model.set_scene_geometry(scene.geometry)
    ids = spaced_indices(len(scene.blocks), min(32, len(scene.blocks)))
    from torch.nn.utils.rnn import pad_sequence
    f = pad_sequence([scene.blocks[i] for i in ids], batch_first=True).to(device)
    active = torch.arange(f.shape[1], device=device)[None] < torch.tensor(
        [len(scene.blocks[i]) for i in ids], device=device)[:, None]
    side = delivered_positions(f[..., :3], active.long(), model.cfg)
    cameras = scene.val_cameras or scene.test_cameras
    cameras = [cameras[i] for i in spaced_indices(len(cameras), min(8, len(cameras)))]
    dirs = directions_for(scene, f, active, cameras)
    with torch.no_grad():
        z = model.encode(f, active)
        base = model.decode(z, active, side)
        target = to_raw(f[active], scene.geometry, model)
        base_raw = to_raw(base[active], scene.geometry, model)
        baseline = error_rows(base_raw, target, model, dirs)
        samples = z[active].double().cpu()
        std = samples.std(0, unbiased=False).to(z)
        stats, perturbations = {}, []
        generator = torch.Generator(device=device).manual_seed(seed)
        for name, (a, b) in model.slices.items():
            x = samples[:, a:b]; x = x-x.mean(0)
            cov = x.T@x/max(1, len(x)-1)
            eigen = torch.linalg.eigvalsh(cov).clamp_min(0)
            mass = eigen/eigen.sum().clamp_min(1e-30)
            stats[name] = dict(dimensions=b-a, variance=x.square().mean(0).tolist(),
                eigenvalues=eigen.tolist(), effective_rank=(float(torch.exp(
                    -(mass*mass.clamp_min(1e-30).log()).sum())) if eigen.sum() > 0 else 0.),
                near_constant_fraction=float((x.square().mean(0) < 1e-8).double().mean()))
            noise = torch.randn(z[..., a:b].shape, device=device, generator=generator)
            for amplitude in (.01, .05):
                disturbed = z.clone()
                disturbed[..., a:b] += amplitude*std[a:b]*noise*active[..., None]
                raw = to_raw(model.decode(disturbed, active, side)[active], scene.geometry, model)
                changed = error_rows(raw, target, model, dirs)
                response = error_rows(raw, base_raw, model, dirs)
                perturbations.append(dict(latent=name, amplitude_std=amplitude,
                    output_response={k: float(v.mean()) for k, v in response.items()},
                    target_mse_change={k: float(changed[k].mean()-baseline[k].mean()) for k in changed}))
    # Small fixed subset for group gradients; no optimizer changes or full-scene graph.
    f, active, side = f[:2], active[:2], side[:2]
    dirs = directions_for(scene, f, active, cameras)
    with torch.enable_grad():
        pred = model(f, active)
        _, terms = attribute_objective(pred[active], f[active], scene.geometry, model, dirs, scales)
        parameters = list(model.semantic_encoder.fuse.parameters())
        grads = {k: torch.cat([g.reshape(-1) for g in torch.autograd.grad(
            value, parameters, retain_graph=True)]) for k, value in terms.items()}
        cosines = {a+'/'+b: float(torch.nn.functional.cosine_similarity(grads[a], grads[b], dim=0))
                   for i, a in enumerate(grads) for b in list(grads)[i+1:]}
    return dict(sample_points=len(samples), latent=stats, perturbations=perturbations,
                shared_encoder_gradient_cosines=cosines,
                gradient_scope='encoder.fuse on two fixed diagnostic blocks; read-only',
                perturbation_scope='fixed diagnostic blocks; target errors and output sensitivity separately')


def write_plots(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for field in ('source_psnr', 'photo_psnr', 'photo_ssim', 'photo_lpips'):
        fig, ax = plt.subplots(figsize=(7, 4))
        names = sorted({(r['arm'], r['scene']) for r in rows if r['scope'] == 'validation'})
        for arm, scene in names:
            selected = sorted([r for r in rows if r['arm'] == arm and r['scene'] == scene and
                               r['scope'] == 'validation' and field in r], key=lambda r: r['step'])
            if selected:
                ax.plot([r['step'] for r in selected], [r[field] for r in selected], label=arm+'/'+scene)
        if ax.lines:
            ax.set(xlabel='Optimizer updates', ylabel=field); ax.legend(fontsize=7); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(out/(field+'.png'), dpi=160)
        plt.close(fig)


def diagnostic_plots(path, record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    keys = list(record['latent'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].bar(keys, [record['latent'][k]['dimensions'] for k in keys], label='Allocated')
    axes[0].bar(keys, [record['latent'][k]['effective_rank'] for k in keys], width=.5, label='Effective rank')
    axes[0].legend(); axes[0].set_ylabel('Dimensions')
    for k in keys:
        axes[1].plot(record['latent'][k]['variance'], label=k)
    axes[1].set(xlabel='Within-group dimension', ylabel='Latent variance', yscale='symlog')
    axes[1].legend(); fig.tight_layout(); fig.savefig(path/'latent_usage.png', dpi=160); plt.close(fig)
    for amplitude in (.01, .05):
        rows = [r for r in record['perturbations'] if r['amplitude_std'] == amplitude]
        fields = list(rows[0]['output_response'])
        values = np.array([[r['output_response'][f] for f in fields] for r in rows])
        # Different attribute units: compare rows WITHIN each column only.
        fig, ax = plt.subplots(figsize=(8, 3.5))
        im = ax.imshow(np.log10(np.maximum(values, 1e-16)), aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(fields)), fields, rotation=15)
        ax.set_yticks(range(len(rows)), [r['latent'] for r in rows])
        for i in range(len(rows)):
            for j in range(len(fields)):
                ax.text(j, i, f'{values[i,j]:.1e}', ha='center', va='center', fontsize=8, color='white')
        ax.set_title(f'Output sensitivity at {amplitude:g} latent std; compare within columns')
        fig.colorbar(im, ax=ax, label='log10 mean squared response')
        fig.tight_layout(); fig.savefig(path/f'sensitivity_{amplitude:g}.png', dpi=160); plt.close(fig)
