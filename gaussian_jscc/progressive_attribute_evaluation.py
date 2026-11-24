"""Paired prefix evaluation, serialized accounting and actual side delivery."""
from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile

import torch

from comparison_baselines.digital_link import LinkConfig
from comparison_baselines.run import _execute
from .data import to_raw
from .digital_side_delivery import transmit_sides
from .modular_experiment import json_write
from .position_delivery import delivered_positions
from .progressive_attribute_packet import export_packet, receive_packet
from .progressive_attribute_supervision import assess_attributes
from .render_objective import image_metrics
from .semantic_attribute_metrics import save_image


def layout_for(scene, layer, seed, *, layers=8):
    # Draw once in Morton order: evaluation layouts survive batching changes.
    g = torch.Generator().manual_seed(seed)
    values = (torch.randint(1, layers+1, (len(scene.raw),), generator=g) if layer == 'mixed'
              else torch.full((len(scene.raw),), int(layer), dtype=torch.long))
    rows, offset = [], 0
    for ids in scene.group_ids:
        valid = ids >= 0; n = int(valid.sum())
        q = torch.zeros_like(ids)
        q[valid] = values[offset:offset+n]; offset += n
        rows.append(q)
    return rows


@torch.no_grad()
def decode_cached(model, scene, cached, layouts, snr, seed, *, noiseless=False, clean=False, neighbor=True,
                  record_symbols=False):
    if record_symbols and (clean or not noiseless):
        raise ValueError('symbol statistics require a uniform finite noiseless prefix')
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval(); model.set_scene_geometry(scene.geometry)
    rows, squared, count, energy, symbols = [], 0., 0, 0., 0
    symbol_sum, symbol_products, symbol_points = 0., 0., 0
    rates = torch.as_tensor(model.layer_config.rates, device=device)
    for data, q in zip(cached, layouts):
        data, q = data.to(device), q.to(device)
        f, y, active = data[..., :59], data[..., 59:], q > 0
        if clean:
            yh = y
            pred = model.semantic.decode(y, active, delivered_positions(f[..., :3], q, model.cfg))
        else:
            pred, yh, wire = model.forward_cached(f, y, q, snr, noiseless=noiseless,
                                                  generator=generator, neighbor=neighbor)
            n = int(rates[q].sum())
            energy += float(wire['transmitted_power'])*n; symbols += n
            if record_symbols:
                z = wire['received'][active].double()
                if z.numel() != 2*n:
                    raise ValueError('symbol statistics require a uniform finite noiseless prefix')
                symbol_sum = symbol_sum + z.sum(dim=0)
                symbol_products = symbol_products + z.T @ z
                symbol_points += len(z)
        squared += float((yh[active]-y[active]).square().sum()); count += int(active.sum())*y.shape[-1]
        rows.append(to_raw(pred[active], scene.geometry, model).cpu())
    diagnostics = dict(latent_mse=squared/max(1, count), transmitted_power=energy/max(1, symbols))
    if record_symbols:
        # Every consecutive pair belongs to one transmitted complex symbol.
        mean = symbol_sum.cpu()/symbol_points
        second_moment = symbol_products.cpu()/symbol_points
        covariance = second_moment - torch.outer(mean, mean)
        spectrum = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
        fractions = spectrum/spectrum.sum().clamp_min(torch.finfo(spectrum.dtype).tiny)
        positive = fractions[fractions > 0]
        diagnostics['symbols'] = dict(
            points=symbol_points,
            mean_power_by_symbol=second_moment.diag().reshape(-1, 2).sum(1).tolist(),
            centered_power_by_symbol=covariance.diag().clamp_min(0).reshape(-1, 2).sum(1).tolist(),
            real_covariance_eigenvalues=spectrum.tolist(),
            real_covariance_effective_rank=float((-(positive*positive.log()).sum()).exp()) if positive.numel() else 0.)
    return torch.cat(rows), diagnostics


@torch.no_grad()
def render_metrics(raw, model, scene, cameras, perceptual, image_dir=None):
    from .rendering import render
    device = next(model.parameters()).device
    raw = raw.to(device)
    if not torch.isfinite(raw).all():
        raise RuntimeError('nonfinite reconstructed Gaussian attributes')
    views = []
    for index, camera in enumerate(cameras):
        image = render(raw, camera, scene.degree, scene.white_background)
        source = scene.reference.get(camera, device)
        photo = camera.original_image[:3].to(device)
        row = dict(view=str(camera.image_name))
        for prefix, target in [('source', source), ('photo', photo)]:
            m = image_metrics(image, target)
            if perceptual is not None:
                m['lpips'] = perceptual(image, target)
            row.update({prefix+'_'+k: v for k, v in m.items()})
        views.append(row)
        if image_dir is not None and index in (0, len(cameras)//2):
            save_image(Path(image_dir)/f'{index:03d}.png', image)
            save_image(Path(image_dir)/f'{index:03d}_source.png', source)
            save_image(Path(image_dir)/f'{index:03d}_photo.png', photo)
    return dict(views=views, **{k: sum(v[k] for v in views)/len(views) for k in views[0] if k != 'view'})


def assess(model, scene, cached, args, out, *, scope, layers, perceptual=None,
           actual=False, core_comparison=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cameras = scene.val_cameras if scope == 'validation' else scene.test_cameras
    rows = []
    identity = model.identity()
    record_symbols = getattr(args, 'record_symbol_diagnostics', False)
    record_attributes = args.grouped_attribute_supervision or record_symbols
    # The semantic reference is identical at every layer endpoint.
    clean_cache = getattr(scene, '_progressive_clean_metrics', {})
    clean_metrics = clean_cache.get(scope)
    if clean_metrics is None:
        clean, _ = decode_cached(model, scene, cached, layout_for(scene, 1, args.seed), args.snr,
                                 args.seed+1000, clean=True)
        clean_metrics = render_metrics(clean, model, scene, cameras, perceptual)
        if record_attributes:
            clean_metrics['attribute_errors'] = assess_attributes(clean, scene.raw, cameras, next(model.parameters()).device)
        clean_cache[scope] = clean_metrics
        scene._progressive_clean_metrics = clean_cache
        del clean
    json_write(out/'clean.json', dict(scene=scene.name, scope=scope, **clean_metrics))
    for layer in layers:
        layouts = layout_for(scene, layer, args.seed+700001, layers=model.layer_config.layer_count)
        symbols = layer if layer == 'mixed' else model.layer_config.rates[layer]
        for path in ('finite_noiseless', 'awgn'):
            trials = 1 if path == 'finite_noiseless' else args.trials
            for trial in range(trials):
                folder = out/str(layer)/path/str(trial)
                folder.mkdir(parents=True, exist_ok=True)
                result = folder/'result.json'
                if result.exists():
                    import json
                    row = json.loads(result.read_text(encoding='utf8'))
                    if row['codec_id'] != identity:
                        raise ValueError('evaluation resume weights changed')
                    rows.append(row); continue
                seed = args.seed+1000+trial
                raw, diagnostics = decode_cached(model, scene, cached, layouts, args.snr,
                    seed, noiseless=path == 'finite_noiseless',
                    record_symbols=record_symbols and path == 'finite_noiseless' and layer != 'mixed')
                with tempfile.TemporaryDirectory(prefix='packet-', dir=folder) as tmp:
                    sender, receiver = Path(tmp)/'sender', Path(tmp)/'receiver'
                    measured, cost = export_packet(model, scene, cached, layouts, args.snr, seed, sender,
                                                  noiseless=path == 'finite_noiseless')
                    packet_raw = receive_packet(model, sender)
                    torch.testing.assert_close(packet_raw, raw, atol=2e-5, rtol=1e-5)
                    row = dict(scene=scene.name, scope=scope, layer=layer, path=path, trial=trial,
                        seed=seed, snr_db=args.snr, codec_id=identity, diagnostics=diagnostics,
                        accounting=cost, packet_measured=measured,
                        packet_max_error=float((packet_raw-raw).abs().max()),
                        **render_metrics(packet_raw, model, scene, cameras, perceptual,
                            folder/'images' if scope == 'test' and trial == 0 and path == 'awgn' else None))
                    if record_attributes:
                        row['attribute_errors'] = assess_attributes(packet_raw, scene.raw, cameras,
                                                                    next(model.parameters()).device)
                    if actual and path == 'awgn':
                        config = LinkConfig(device=args.digital_device)
                        def runner(src, dst, report, link_seed):
                            params = ['transmit', '--source', src, '--destination', dst, '--report', report,
                                      '--snr-db', args.snr, '--seed', link_seed]
                            params += [item for k, v in asdict(config).items()
                                       for item in ('--'+k.replace('_', '-'), v)]
                            _execute(args.digital_python, 'comparison_baselines.digital_link', params, folder/'link.log')
                            import json
                            return json.loads(Path(report).read_text(encoding='utf8'))
                        link = transmit_sides(sender, receiver, folder/'links', config, args.snr,
                                              args.seed+100000+3*trial, runner)
                        if link['side_channel_uses'] != cost['side_channel_uses']:
                            raise RuntimeError('actual side usage differs from serialized cost')
                        delivered = dict(status=link['status'], total_channel_uses=cost['total_channel_uses'],
                            side_link_simulated=True, streams=link['streams'], metrics=None)
                        if link['status'] == 'success':
                            shutil.copyfile(sender/'received.npy', receiver/'received.npy')
                            recovered = receive_packet(model, receiver)
                            torch.testing.assert_close(recovered, packet_raw, atol=0, rtol=0)
                            delivered['metrics'] = {k: v for k, v in row.items()
                                                     if k.startswith(('source_', 'photo_'))}
                            delivered['quality_basis'] = 'received-file decode equals the rendered packet tensor exactly'
                        row['actual_delivery'] = delivered
                    if core_comparison and bool(model.neighbor_enabled):
                        core, _ = decode_cached(model, scene, cached, layouts, args.snr, seed,
                            noiseless=path == 'finite_noiseless', neighbor=False)
                        row['core_metrics'] = render_metrics(core, model, scene, cameras, perceptual)
                json_write(result, row); rows.append(row)
                print(f'{scope} {scene.name} k={symbols} '
                      f'{path} trial={trial}: PSNR={row["source_psnr"]:.4f}, '
                      f'photo={row["photo_psnr"]:.4f}, cost={cost["total_channel_uses"]/1e6:.5f}M', flush=True)
    json_write(out/'summary.json', dict(rows=rows, clean=clean_metrics, scope=scope,
        quality_conditioning='main metrics assume reliable side files; actual_delivery reports successes and failures separately'))
    return rows
