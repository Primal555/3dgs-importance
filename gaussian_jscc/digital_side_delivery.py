"""Evaluate completed allocations with actual LDPC/QAM side-file delivery."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import numpy as np
import torch
from comparison_baselines.digital_link import LinkConfig, file_channel_cost
from comparison_baselines.run import _execute, summarize_trials
from comparison_baselines.side_information_accounting import side_information_cost
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_evaluation import tier_hash
from .cli import device_for, seed_all
from .data import read_ply
from .model_factory import build_codec
from .modular_experiment import json_write
from .perceptual_metrics import LPIPSEvaluator
from .render_objective import image_metrics, spaced_indices
from .render_validation import save_panel
from .successive_delivery_packet import export_packet, receive_packet, SIDE_FILES
from .successive_multiscene_inputs import make_scene
from .transport import model_id


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def transmit_sides(sender, receiver, reports, config, snr, seed, runner):
    """All three streams are attempted and charged, including failed deliveries."""
    sender, receiver, reports = Path(sender), Path(receiver), Path(reports)
    receiver.mkdir(parents=True, exist_ok=False); reports.mkdir(parents=True, exist_ok=True)
    results = {}
    for index, name in enumerate(SIDE_FILES):
        source, destination = sender/name, receiver/name
        result = runner(source, destination, reports/(name+'.json'), seed+index)
        expected = file_channel_cost(source.stat().st_size, config)
        if (result['config'] != asdict(config) or result['snr_db'] != snr or
                result['seed'] != seed+index or result.get('noiseless') is not False or
                any(result[k] != v for k, v in expected.items()) or result['source_sha256'] != sha(source)):
            raise RuntimeError('digital link result/protocol identity mismatch: '+name)
        if result['status'] == 'success':
            if not destination.is_file() or sha(destination) != sha(source):
                raise RuntimeError('successful link did not publish the received bytes: '+name)
        elif result['status'] != 'failed': raise RuntimeError('unexpected file delivery status')
        results[name] = result
    return dict(status='success' if all(r['status']=='success' for r in results.values()) else 'failed',
        streams=results, side_channel_uses=sum(r['total_channel_uses'] for r in results.values()))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allocation', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--labels', nargs='+', choices=('selected','latest','uniform_q2','uniform_q3'), default=['selected','uniform_q2'])
    p.add_argument('--digital-python', required=True); p.add_argument('--device', default='cuda')
    p.add_argument('--digital-device', default='cuda'); p.add_argument('--manifest')
    p.add_argument('--baseline', type=Path, help='completed digital baseline directory; inherit its exact LinkConfig')
    p.add_argument('--cpu-threads', type=int, default=4)
    return p


def main(args):
    source, out = args.allocation.resolve(), args.out.resolve()
    if source.is_relative_to(out) or out.is_relative_to(source): raise ValueError('source/output must be separate')
    if out.exists(): raise FileExistsError('use a new output directory')
    if len(args.labels)!=len(set(args.labels)): raise ValueError('duplicate labels')
    if args.cpu_threads < 1: raise ValueError('positive cpu threads required')
    complete, plan = read(source/'complete.json'), read(source/'plan.json')
    if complete['status'] != 'complete': raise ValueError('completed allocation required')
    settings = plan['settings']
    config = LinkConfig(device=args.digital_device)
    baseline = None
    if args.baseline:
        baseline = read(args.baseline/'summary.json')
        if baseline['status']!='complete' or baseline['plan'].get('arq') is not False:
            raise ValueError('completed no-ARQ baseline required')
        config = LinkConfig(**dict(baseline['plan']['link'], device=args.digital_device))
    options = argparse.Namespace(**settings)
    options.out = str(out); options.device = args.device
    options.manifest = args.manifest or settings.get('manifest')
    saved, cfg, spec, old, scene_args, source_info = load_inputs(options)
    device = device_for(args.device); torch.set_num_threads(args.cpu_threads)
    model = build_codec(cfg).to(device); model.load_state_dict(saved['state_dict']); del saved
    model.eval().requires_grad_(False); model.set_training_snrs([10.])
    codec_id = model_id(model)
    if codec_id != complete['codec_id']: raise ValueError('allocation/codec identity mismatch')
    raw, degree = read_ply(spec['ply'])
    out.mkdir(parents=True)
    scene = make_scene(spec, raw, degree, model, scene_args, out, old=old); del raw
    cameras = scene.test_cameras
    if baseline is not None:
        b = baseline['plan']
        if (b['scene'] != spec['name'] or b['evaluation']['camera_names'] != [str(c.image_name) for c in cameras] or
                any(b['evaluation'][k] != getattr(scene_args,k) for k in ('resolution','images','white_background'))):
            raise ValueError('baseline/source camera evaluation protocol differs')
    snr, trials, seed = settings['snr'], settings['trials'], settings['seed']
    if trials < 1: raise ValueError('positive trials required')
    eval_plan = dict(version='actual_digital_side_delivery_v1', allocation=str(source),
        allocation_complete_sha256=sha(source/'complete.json'), codec_id=codec_id,
        source=source_info, link=asdict(config), snr_db=snr, trials=trials, labels=args.labels,
        camera_names=[str(c.image_name) for c in cameras], reference='source PLY and original photos',
        arq=False, side_link_simulated=True, quality_conditioning='all three side files delivered successfully',
        receiver_inputs='shared model, received XYZ/metadata/scales files and noisy attribute symbols',
        source_side_copies_used_at_receiver=False, analogue_noise_seed='seed+1000+trial',
        digital_noise_seed='seed+100000+3*trial+stream_index',
        baseline=str(args.baseline) if args.baseline else None)
    json_write(out/'plan.json', eval_plan)
    rows = []; perceptual = LPIPSEvaluator(device) if settings['lpips'] else None
    for label in args.labels:
        if label.startswith('uniform_q'): q = torch.full((len(scene.raw),),int(label[-1]),dtype=torch.long)
        else:
            q = torch.from_numpy(np.load(source/'deployment'/f'{label}_tiers.npy',allow_pickle=False).astype(np.int64))
            if label=='selected' and tier_hash(q)!=complete['tier_sha256']: raise ValueError('selected tier hash mismatch')
        for trial in range(trials):
            folder = out/'trials'/label/f'{trial:03d}'; folder.mkdir(parents=True)
            print(f'{label} trial {trial+1}/{trials}: serializing packet and transmitting three side files at {snr:g}dB',flush=True)
            with tempfile.TemporaryDirectory(prefix='delivery-', dir=folder) as tmp:
                tmp = Path(tmp); sender, receiver = tmp/'sender', tmp/'receiver'
                seed_all(seed+1000+trial)
                accounting = export_packet(model,scene,q,snr,seed+1000+trial,sender)
                if label==args.labels[0] and trial==0:
                    from .modular_evaluation import decode_path
                    from .optimization import preserved_rng
                    packet_raw, _ = receive_packet(model,sender)
                    with preserved_rng(device):
                        seed_all(seed+1000+trial)
                        qs = [torch.where(ids>=0,q[ids.clamp_min(0)],0) for ids in scene.group_ids]
                        ideal = decode_path(model,scene,qs,'awgn',snr)[0] if (q>0).any() else scene.raw[:0].to(device)
                    difference = (packet_raw-ideal).abs()
                    diagnostic = dict(status='checking',device=str(device),
                        xyz_wire_quantizer_exact=True,
                        max_absolute_error=float(difference.max()) if ideal.numel() else 0.,
                        mismatched_elements=int((difference > 2e-5+1e-5*ideal.abs()).sum()),
                        max_error_per_attribute=difference.amax(0).tolist() if ideal.numel() else [])
                    try:
                        torch.testing.assert_close(packet_raw,ideal,atol=2e-5,rtol=1e-5)
                    except AssertionError:
                        diagnostic['status']='failed'
                        json_write(out/'execution_check.json',diagnostic)
                        raise
                    json_write(out/'execution_check.json',dict(status='passed',device=str(device),
                        xyz_wire_quantizer_exact=True,
                        packet_matches_original_batched_codec=True,
                        max_absolute_error=float((packet_raw-ideal).abs().max()) if ideal.numel() else 0.))
                    del packet_raw,ideal
                def runner(src,dest,report,link_seed):
                    params = ['transmit','--source',src,'--destination',dest,'--report',report,
                              '--snr-db',snr,'--seed',link_seed]
                    params += [item for k,v in asdict(config).items() for item in ('--'+k.replace('_','-'),v)]
                    _execute(args.digital_python,'comparison_baselines.digital_link',params,folder/'link.log')
                    return read(report)
                link = transmit_sides(sender,receiver,folder/'links',config,snr,seed+100000+3*trial,runner)
                cost = side_information_cost(accounting,config)
                if link['side_channel_uses']!=cost['side_channel_uses']: raise RuntimeError('side cost mismatch')
                row = dict(method=label,snr_db=snr,trial=trial,status=link['status'],metrics=None,
                    payload_complex_symbols=accounting['payload_complex_symbols'],
                    total_channel_uses=cost['total_channel_uses'],comparison_accounting=dict(cost,
                        side_link_simulated=True,reliability=('actual LDPC/QAM/AWGN; CRC and SHA256 verified'
                            if link['status']=='success' else 'actual LDPC/QAM/AWGN; side-file delivery failed')),
                    side_file_status={k:r['status'] for k,r in link['streams'].items()},
                    tier_sha256=tier_hash(q),images_saved=False)
                if link['status']=='success':
                    # Attribute payload is analogue AWGN; its float container is not digitally retransmitted.
                    shutil.copyfile(sender/'received.npy',receiver/'received.npy')
                    # Remove all transmitted source side files before receiver execution.
                    for name in SIDE_FILES: (sender/name).unlink()
                    decoded, packet_q = receive_packet(model,receiver)
                    observations = []
                    from .rendering import render
                    for index,camera in enumerate(cameras):
                        image = render(decoded,camera,scene.degree,scene.white_background)
                        ref = scene.reference.get(camera,device); photo = camera.original_image[:3].to(device)
                        values = dict(view=str(camera.image_name))
                        for prefix,target in (('source',ref),('photo',photo)):
                            values.update({prefix+'_'+k:v for k,v in image_metrics(image,target).items()})
                            if perceptual is not None: values[prefix+'_lpips']=perceptual(image,target)
                        observations.append(values)
                        if trial==0 and index in spaced_indices(len(cameras),settings['image_views']):
                            save_panel(out/'images'/label/f'view{index:04d}.png',photo,ref,image,label)
                            row['images_saved']=True
                    row['metrics']={k:float(np.mean([v[k] for v in observations])) for k in observations[0] if k!='view'}
                    json_write(folder/'per_view.json',observations)
                    if not torch.equal(packet_q,q[scene.order]): raise RuntimeError('receiver tier order differs')
                json_write(folder/'result.json',row); rows.append(row)
            print(f'{label} trial {trial}: {row["status"]}, cost={row["total_channel_uses"]/1e6:.6f}M',flush=True)
            json_write(out/'summary.json',dict(status='running',plan=eval_plan,trials=rows,results=summarize_trials(rows)))
    if model_id(model)!=codec_id: raise RuntimeError('shared codec changed')
    summary = dict(status='complete',plan=eval_plan,trials=rows,results=summarize_trials(rows),frozen_codec_verified=True)
    json_write(out/'summary.json',summary)
    json_write(out/'complete.json',dict(status='complete',trials=len(rows),successes=sum(r['status']=='success' for r in rows)))
    with tarfile.open(out/'review.tar.gz','w:gz') as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name!='review.tar.gz' and path.suffix in ('.json','.log','.png'):
                archive.add(path,arcname=path.relative_to(out))
    return summary


if __name__=='__main__': main(parser().parse_args())
