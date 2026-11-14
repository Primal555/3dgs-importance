"""Paired quality matrices and actual side-file delivery for multi-SNR codecs."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from comparison_baselines.side_information_accounting import side_information_cost
from comparison_baselines.side_link_policy import side_link_config
from .cli import seed_all
from .modular_evaluation import decode_path
from .modular_experiment import json_write
from .multisnr_training import tiers, LAYOUTS
from .optimization import preserved_rng
from .render_objective import image_metrics
from .render_validation import save_panel
from .student_digital_cost import DigitalDeploymentCost
from .successive_delivery_packet import packet_header
from .transport import model_id, encode_metadata


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def fixed_map(scene, layout, seed):
    # Original PLY row order is the common boundary for meters and packets.
    return tiers(torch.ones(len(scene.raw), dtype=torch.bool), layout, seed+991)


@torch.no_grad()
def measure(model, scene, raw, cameras, perceptual, image_path=None):
    from .rendering import render
    device = next(model.parameters()).device
    rows = []
    for index, camera in enumerate(cameras):
        image = render(raw, camera, scene.degree, scene.white_background)
        source = scene.reference.get(camera, device)
        photo = camera.original_image[:3].to(device)
        row = dict(view=str(camera.image_name))
        for prefix, target in (('source', source), ('photo', photo)):
            row.update({prefix+'_'+k: v for k,v in image_metrics(image, target).items()})
            if perceptual is not None: row[prefix+'_lpips'] = perceptual(image, target)
        rows.append(row)
        if index == 0 and image_path is not None:
            save_panel(image_path, photo, source, image, 'photo | source PLY | delivered')
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'view'}, rows


def evaluate_matrix(model, scene, args, out, label, *, scope='test', fixed_condition=False, perceptual=None):
    """Each row is atomic: resuming a long evaluation keeps completed trials."""
    device = next(model.parameters()).device
    root = Path(out)/'evaluation'/label/scene.name
    cameras = scene.test_cameras if scope == 'test' else scene.val_cameras
    identity = model_id(model)
    model.eval(); model.set_training_snrs([10.] if fixed_condition else args.snr_range)
    layouts=getattr(args,'eval_layouts',LAYOUTS)
    kinds=getattr(args,'eval_paths',('clean','finite_noiseless','awgn'))
    paths = [('clean', 10., '3', 0)] if 'clean' in kinds else []
    if 'finite_noiseless' in kinds:
        paths += [('finite_noiseless', snr, q, 0) for snr in args.eval_snrs for q in layouts]
    if 'awgn' in kinds:
        paths += [('awgn', snr, q, trial) for snr in args.eval_snrs for q in layouts for trial in range(args.trials)]
    rows = []
    meter = DigitalDeploymentCost(model, scene, identity, args.eval_snrs[0], 2.)
    cost_cache = {}
    with preserved_rng(device):
        # Source PLY fidelity to photographs is an unchanged input-quality reference.
        source_rows = []
        for camera in cameras:
            source = scene.reference.get(camera, device)
            photo = camera.original_image[:3].to(device)
            r = dict(view=str(camera.image_name), **image_metrics(source, photo))
            if perceptual is not None: r['lpips'] = perceptual(source, photo)
            source_rows.append(r)
        json_write(root/'source_photo.json', dict(views=source_rows,
            mean={k: float(np.mean([r[k] for r in source_rows])) for k in source_rows[0] if k != 'view'}))
        for path, snr, layout, trial in paths:
            config=side_link_config(snr,getattr(args,'side_link_policy','fixed_16qam_v1'),
                                    getattr(args,'digital_device','cuda'))
            file = root/'rows'/f'{path}_{snr:g}_{layout}_{trial}.json'
            if file.exists():
                row = read(file)
                if row['codec_id'] != identity: raise RuntimeError('evaluation checkpoint identity changed')
                if path!='clean' and any(row['comparison_accounting']['link'][key]!=getattr(config,key)
                                         for key in ('k','n','bits_per_symbol')):
                    raise RuntimeError('evaluation side-link configuration changed')
                rows.append(row); continue
            q = fixed_map(scene, layout, args.seed)
            qs = [torch.where(ids>=0, q[ids.clamp_min(0)], 0) for ids in scene.group_ids]
            seed_all(args.seed+1000+trial)
            raw, diagnostics = decode_path(model, scene, qs, path, snr)
            image_path = (root/'images'/f'{snr:g}_{layout}.png'
                if scene.name in args.delivery_scenes and label == 'final' and path == 'awgn' and trial == 0
                and snr in (args.eval_snrs[0],10.,args.eval_snrs[-1]) else None)
            metrics, views = measure(model, scene, raw, cameras, perceptual, image_path)
            cost = None
            if path != 'clean':
                key = (path,snr,layout)
                if key not in cost_cache:
                    meter.snr = snr
                    ledger = meter.details(q)
                    header = packet_header(model,scene,q,snr,noiseless=path=='finite_noiseless',codec_id=identity)
                    ledger['metadata_bytes'] = len(encode_metadata(header,q[scene.order].numpy()))
                    cost_cache[key] = side_information_cost(ledger,config)
                cost = cost_cache[key]
            row = dict(codec_id=identity, scene=scene.name, endpoint=label, scope=scope,
                path=path, snr_db=snr, condition_snr_db=model.resolve_condition_snr(snr),
                layout=layout, trial=trial, seed=args.seed+1000+trial, **metrics,
                diagnostics=diagnostics, comparison_accounting=cost,
                total_channel_uses=None if cost is None else cost['total_channel_uses'],
                quality_conditioning='clean bypass' if cost is None else 'reliable_side_assumption', views=views)
            json_write(file, row); rows.append(row)
            print(f'{label}/{scene.name} {path}/{layout} SNR={snr:g} trial={trial}: '
                  f'source={metrics["source_psnr"]:.3f}, photo={metrics["photo_psnr"]:.3f}', flush=True)
    json_write(root/'summary.json', dict(status='complete', codec_id=identity, rows=rows))
    model.set_training_snrs(args.snr_range)
    return rows


def actual_delivery(model, scene, args, out, *, perceptual=None):
    """Run the same no-ARQ digital link as the baselines in its own environment."""
    from comparison_baselines.run import _execute, summarize_trials
    from .digital_side_delivery import transmit_sides
    from .successive_delivery_packet import export_packet, receive_packet, SIDE_FILES
    root = Path(out)/'actual_delivery'/scene.name
    identity = model_id(model); rows = []
    model.eval(); model.set_training_snrs(args.snr_range)
    for snr in args.eval_snrs:
        config=side_link_config(snr,getattr(args,'side_link_policy','fixed_16qam_v1'),args.digital_device)
        for layout in args.delivery_layouts:
            q = fixed_map(scene, layout, args.seed)
            for trial in range(args.trials):
                folder = root/f'{snr:g}'/layout/f'{trial:03d}'
                result = folder/'result.json'
                if result.exists():
                    row = read(result)
                    if row['codec_id'] != identity: raise RuntimeError('delivery checkpoint identity changed')
                    if row['comparison_accounting']['link']!=asdict(config):
                        raise RuntimeError('delivery side-link configuration changed')
                    rows.append(row); continue
                folder.mkdir(parents=True, exist_ok=True)
                print(f'Actual side files: {scene.name}, {snr:g} dB, q={layout}, trial={trial}', flush=True)
                with tempfile.TemporaryDirectory(prefix='packet-', dir=folder) as tmp:
                    sender, receiver = Path(tmp)/'sender', Path(tmp)/'receiver'
                    accounting = export_packet(model, scene, q, snr, args.seed+1000+trial, sender)
                    if not (folder/'execution_check.json').exists():
                        packet_raw, packet_q = receive_packet(model, sender)
                        with preserved_rng(next(model.parameters()).device):
                            seed_all(args.seed+1000+trial)
                            qs = [torch.where(ids>=0, q[ids.clamp_min(0)], 0) for ids in scene.group_ids]
                            ideal, _ = decode_path(model, scene, qs, 'awgn', snr)
                        torch.testing.assert_close(packet_raw, ideal, atol=2e-5, rtol=1e-5)
                        if not torch.equal(packet_q, q[scene.order]): raise RuntimeError('packet row order mismatch')
                        json_write(folder/'execution_check.json', dict(status='passed',
                            maximum_absolute_error=float((packet_raw-ideal).abs().max())))
                        del packet_raw, ideal

                    def runner(src, dest, report, link_seed):
                        params = ['transmit','--source',src,'--destination',dest,'--report',report,
                                  '--snr-db',snr,'--seed',link_seed]
                        params += [x for k,v in asdict(config).items() for x in ('--'+k.replace('_','-'),v)]
                        _execute(args.digital_python, 'comparison_baselines.digital_link', params, folder/'link.log')
                        return read(report)

                    link = transmit_sides(sender, receiver, folder/'links', config, snr,
                                          args.seed+100000+3*trial, runner)
                    cost = side_information_cost(accounting, config)
                    if link['side_channel_uses'] != cost['side_channel_uses']: raise RuntimeError('digital accounting mismatch')
                    cost.update(side_link_simulated=True,
                        reliability=f'actual LDPC{config.k}/{config.n}, {2**config.bits_per_symbol}QAM, AWGN, no ARQ')
                    row = dict(codec_id=identity, method=layout, snr_db=snr, trial=trial,
                        status=link['status'], metrics=None, comparison_accounting=cost,
                        total_channel_uses=cost['total_channel_uses'],
                        quality_conditioning='successful reception of all three side files',
                        side_file_status={k:v['status'] for k,v in link['streams'].items()})
                    if link['status'] == 'success':
                        shutil.copyfile(sender/'received.npy', receiver/'received.npy')
                        for name in SIDE_FILES: (sender/name).unlink()
                        raw, received_q = receive_packet(model, receiver)
                        if not torch.equal(received_q, q[scene.order]): raise RuntimeError('receiver tier map mismatch')
                        row['metrics'], views = measure(model, scene, raw, scene.test_cameras, perceptual)
                        json_write(folder/'per_view.json', views)
                    json_write(result, row); rows.append(row)
    report = dict(status='complete', codec_id=identity, trials=rows, results=summarize_trials(rows),
                  side_link_policy=getattr(args,'side_link_policy','fixed_16qam_v1'))
    json_write(root/'summary.json', report)
    return report
