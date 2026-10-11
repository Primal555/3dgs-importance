"""Paired deployment comparisons. Real packet bytes, not float I/Q storage bytes."""
import argparse
import csv
import json
import math
from io import BytesIO
from pathlib import Path

import numpy as np
import torch

from .allocation_diagnostics import prior_ranked_tiers
from .allocation import deployment_tiers
from .multiscene import Scene, manifest, parser as training_parser
from .data import read_ply
from .position_delivery import encode_positions
from .render_validation import append_json, validate_render
from .route2 import hard_tiers, load_mask
from .transport import encode_metadata, load_checkpoint, model_id


def shuffled_tiers(q, seed, positive_only=False):
    q = q.clone().cpu()
    indices = torch.where(q>0)[0] if positive_only else torch.arange(len(q))
    permutation = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
    q[indices] = q[indices[permutation]]
    return q


def draw_diagnostics(probabilities, tiers, ten_draw_tiers, draws):
    """Separate deletion changes from positive-vote changes; no causal claims."""
    both_kept = (tiers > 0) & (ten_draw_tiers > 0)
    return {
        'deployment_draws': draws,
        'expected_deleted_gaussians': float(probabilities[:, 0].double().pow(draws).sum()),
        'actual_deleted_gaussians': int((tiers == 0).sum()),
        'changed_deletion_decisions_vs_ten': int(((tiers == 0) != (ten_draw_tiers == 0)).sum()),
        'changed_positive_tiers_vs_ten': int(((tiers != ten_draw_tiers) & both_kept).sum()),
        'mean_single_draw_q0_probability': float(probabilities[:, 0].mean()),
        'single_draw_expected_tier_counts': probabilities.double().sum(0).tolist(),
        'draw_rule_note': 'all draws q0 deletes; otherwise positive mode. Fewer draws also change positive-tier votes.',
    }


def packet_cost(model, geometry, raw_sorted, q_sorted, snr, channel, net_bits_per_use=2.):
    """Exactly the same XYZ+header+tier syntax as transport.transmit.

    The reliable digital net spectral efficiency is an assumption, not a simulated FEC.
    No float-array disk sizes are converted into alleged analogue payload bits.
    """
    if not math.isfinite(net_bits_per_use) or net_bits_per_use <= 0:
        raise ValueError('net_bits_per_use must be positive')
    q = q_sorted.detach().cpu().long()
    xyz = encode_positions(geometry.normalize(raw_sorted[:,:3]), q, model.cfg)
    header = {'version':2, 'config':model.cfg.to_dict(), 'model_id':model_id(model),
              'count':int((q>0).sum()), 'source_count':len(q), 'geometry':geometry.to_dict(),
              'snr_db':float(snr), 'channel':channel}
    metadata = encode_metadata(header,q.numpy())
    payload = int(torch.tensor(model.cfg.rates)[q].sum())
    coordinate_uses = math.ceil(len(xyz)*8/net_bits_per_use)
    metadata_uses = math.ceil(len(metadata)*8/net_bits_per_use)
    total = payload+coordinate_uses+metadata_uses
    capacity = math.log2(1+10**(snr/10))
    return {'payload_complex_symbols':payload,'xyz_bytes':len(xyz),'metadata_bytes':len(metadata),
            'xyz_MB':len(xyz)/1e6,'metadata_MB':len(metadata)/1e6,
            'coordinate_channel_uses':coordinate_uses,'metadata_channel_uses':metadata_uses,
            'total_channel_uses':total,'total_uses_per_source_gaussian':total/len(q),
            'net_bits_per_use_assumption':net_bits_per_use,
            'same_snr_awgn_capacity_bits_per_use':capacity,
            'digital_efficiency_exceeds_same_snr_capacity':net_bits_per_use>capacity,
            'equivalent_digital_MB':total*net_bits_per_use/8/1e6,
            'equivalent_MB_note':'resource-equivalent at assumed digital efficiency; NOT JSCC payload file size',
            'rate_scope':'actual XYZ stream + complete metadata framing + analogue payload; shared model excluded',
            'reliability':'XYZ and metadata assumed reliable; no FEC/packet errors simulated'}


def evaluate(args):
    from .cli import device_for
    from .rendering import load_cameras, RenderReference
    device = device_for(args.device)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'use a new evaluation directory: {out}')
    checkpoint = Path(args.checkpoint)
    model = load_checkpoint(checkpoint,device).eval()
    if model.cfg.prefix_mode != 'progressive' or model.cfg.position_delivery != 'quantized':
        raise ValueError('this experiment requires progressive JSCC with explicit quantized coordinates')
    identity = model_id(model)
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    training = saved.get('training',{})
    from .appearance_metrics import load_regions
    color_regions = load_regions(args.color_rois) if args.color_rois else training.get('color_regions', {})
    if args.uniform_only and (args.require_allocations or args.draw_counts):
        raise ValueError('uniform-only cannot require deployment allocation tables')
    # Use original resolution/layout unless explicitly overridden.
    opts = training_parser().parse_args(['--out',str(out)])
    for key in ('resolution','white_background','images','blocks_per_batch','snr','net_bits_per_use'):
        if key in training:
            setattr(opts,key,training[key])
    if args.resolution is not None:
        opts.resolution = args.resolution
    opts.net_bits_per_use = args.net_bits_per_use
    selected = [s for s in manifest(args.manifest) if args.role=='all' or s['role']==args.role]
    if args.scenes:
        if not set(args.scenes).issubset({s['name'] for s in selected}):
            raise ValueError('--scenes must select existing scenes within the requested role')
        selected = [s for s in selected if s['name'] in args.scenes]
    if args.draw_counts and (len(set(args.draw_counts)) != len(args.draw_counts) or
                            any(not 1 <= n <= 10 for n in args.draw_counts)):
        raise ValueError('--draw-counts must be distinct integers in 1..10')
    if not selected or args.trials<1 or args.test_views<0 or args.image_views<1 or not args.snrs or not all(math.isfinite(s) for s in args.snrs):
        raise ValueError('invalid scene/trial/view/SNR selection')
    trained_names = {s['name'] for s in training.get('codec_training_scenes',training.get('selected_scenes',[])) if s['role']=='train'}
    if any(s['role']=='heldout' and s['name'] in trained_names for s in selected):
        raise ValueError('heldout scene was part of this codec training manifest')
    out.mkdir(parents=True)
    # This compact checkpoint is what would need distributing once, not optimizer state.
    from .transport import save_checkpoint
    with BytesIO() as buffer:
        save_checkpoint(buffer,model,0,{'purpose':'evaluation deployment weights'})
        model_bytes = buffer.getbuffer().nbytes
        if args.output_profile == 'full':
            (out/'shared_codec.pt').write_bytes(buffer.getvalue())
    rows, view_rows = [], []
    for spec in selected:
        raw,degree = read_ply(spec['ply'])
        if degree != model.cfg.sh_degree:
            raise ValueError(f'{spec["name"]}: SH degree mismatch')
        s = Scene(spec,raw,degree,model,opts,out,with_mask=False,cameras=False)
        cameras = load_cameras(spec['source'],opts.resolution,opts.white_background,opts.images,'test')
        if not cameras:
            raise ValueError(f'{s.name}: empty test-camera split')
        if args.test_views:
            cameras = [cameras[i] for i in sorted(set(np.linspace(0,len(cameras)-1,min(args.test_views,len(cameras)),dtype=int)))]
        reference = RenderReference(s.raw,degree,opts.white_background,'source')
        folder = Path(args.allocations_dir) if args.allocations_dir else checkpoint.parent
        allocation_path = folder/f'{s.name}.pt'
        if (args.require_allocations or args.draw_counts) and not allocation_path.is_file():
            raise FileNotFoundError(f'required allocation table missing: {allocation_path}')
        mask = load_mask(allocation_path,raw,model,device) if allocation_path.exists() and not args.uniform_only else None
        del raw
        # Map sampled ONCE at trained SNR; do not silently reallocate for SNR sweep.
        q = hard_tiers(mask,opts.snr,args.deployment_seed) if mask is not None else None
        def batches(tiers):
            return [torch.where(ids>=0,tiers[ids.clamp_min(0)],0) for ids in s.group_ids]
        extras, maps, diagnostics = {}, {}, {}
        if q is not None:
            maps['mask'] = q
            if args.draw_counts:
                with torch.no_grad():
                    probabilities = torch.cat([mask.probabilities(torch.arange(start,
                        min(start+65536, len(mask.logits)), device=device), opts.snr).cpu()
                        for start in range(0, len(mask.logits), 65536)])
                diagnostics['mask'] = draw_diagnostics(probabilities, q, q, 10)
                for draws in args.draw_counts:
                    if draws == 10:
                        continue  # mask is already the ten-draw baseline; do not render it twice.
                    name = f'draws_{draws}'
                    maps[name] = deployment_tiers(probabilities, args.deployment_seed, draws=draws)
                    extras[name] = batches(maps[name])
                    diagnostics[name] = draw_diagnostics(probabilities, maps[name], q, draws)
                del probabilities
            for seed in args.shuffle_seeds:
                for positive in (False,True):
                    name = ('shuffle_positive_' if positive else 'shuffle_all_')+str(seed)
                    maps[name] = shuffled_tiers(q,seed,positive)
                    extras[name] = batches(maps[name])
            if s.prior is not None:
                maps['prior_ranked'] = prior_ranked_tiers(s.prior,q)
                extras['prior_ranked'] = batches(maps['prior_ranked'])
        for snr in args.snrs:
            dest = s.out/f'snr_{snr:g}'
            dest.mkdir()
            result = validate_render(model,s.groups,s.group_ids,s.raw,s.geometry,cameras,reference,
                snr,args.channel,args.trials,args.seed,dest,0,'test',mask=mask,
                mask_tiers=q,extra_layouts=extras,white_background=opts.white_background,
                position_net_bits_per_use=args.net_bits_per_use,position_meter=s.position_meter,
                image_views=args.image_views if args.output_profile=='compact' else None,
                image_layouts=tuple(args.image_layouts) if args.image_layouts else
                    tuple(diagnostics) if diagnostics and args.output_profile=='compact' else
                    ('mask','1',str(len(model.cfg.rates)-1),'mixed') if args.output_profile=='compact' else None,
                save_images=not args.no_images, color_metrics=args.color_metrics,
                color_regions=color_regions.get(s.name, {}))
            full_cost = packet_cost(model,s.geometry,s.raw,torch.full((len(s.raw),),len(model.cfg.rates)-1),
                                    snr,args.channel,args.net_bits_per_use)
            for entry in result['layouts']:
                label = entry['layout']
                if label == 'mixed':
                    # A random layout is a codec diagnostic, not a learned deployment policy.
                    continue
                tiers = maps[label] if label in maps else torch.full((len(s.raw),),int(label),dtype=torch.long)
                cost = packet_cost(model,s.geometry,s.raw,tiers[s.order],snr,args.channel,args.net_bits_per_use)
                row = {k:v for k,v in entry.items() if k!='views'}
                row.update(cost,scene=s.name,role=spec['role'],layout=label,snr=snr,channel=args.channel,
                    codec_id=identity,source_gaussians=len(s.raw),test_views=len(cameras),trials=args.trials,
                    deployment_seed=args.deployment_seed,shared_model_bytes=model_bytes,
                    beta=training.get('beta'),training_seed=training.get('seed'),noise_seed=args.seed,
                    total_channel_uses_first_delivery=cost['total_channel_uses']+math.ceil(model_bytes*8/args.net_bits_per_use),
                    training_snr=opts.snr,stage=training.get('checkpoint_label',checkpoint.parent.name),
                    training_scope=training.get('scope','unknown'),
                    full_tier_total_channel_uses=full_cost['total_channel_uses'],
                    savings_vs_full_tier_percent=100*(1-cost['total_channel_uses']/full_cost['total_channel_uses']),
                    generalization=('unknown training-scene provenance' if not trained_names else
                                    'seen-scene heldout cameras' if s.name in trained_names else 'unseen-scene frozen codec'),
                    allocation_fit=('scene-adapted table, NOT whole-system zero shot' if mask is not None and spec['role']=='heldout'
                                    else 'scene-trained table' if mask is not None else 'no table'),
                    prior_baseline_available=s.prior is not None)
                row.update(diagnostics.get(label, {}))
                if label == 'mask':
                    row.setdefault('deployment_draws', 10)
                row.update(allocation_training_sampling=training.get('allocation_sampling', 'unknown'),
                           frozen_codec_verified=training.get('frozen_codec_verified', False))
                rows.append(row)
                append_json(out/'results.jsonl',row)
                for v in entry['views']:
                    view_rows.append({'scene':s.name,'role':spec['role'],'layout':label,'snr':snr,**v})
            for gain in result.get('prefix_gains',[]):
                append_json(out/'prefix_gains.jsonl',{'scene':s.name,'role':spec['role'],'snr':snr,**gain})
            if mask is not None:
                learned = next(e for e in result['layouts'] if e['layout']=='mask')
                for e in result['layouts']:
                    if e['layout'] in ('mixed','mask'):
                        continue
                    for metric in ('source_mse','source_psnr','source_ssim','photo_psnr','photo_ssim'):
                        differences=[a[metric]-b[metric] for a,b in zip(learned['views'],e['views'])]
                        append_json(out/'paired_comparisons.jsonl',{'scene':s.name,'role':spec['role'],'snr':snr,
                            'comparison':'mask minus '+e['layout'],'metric':metric,
                            'mean_delta':sum(differences)/len(differences),'paired_view_trial_deltas':differences,
                            'note':'paired measurements; not independent-scene statistical significance'})
        del s,mask,reference,cameras
    if model_id(model) != identity:
        raise RuntimeError('evaluation changed codec weights/statistics')
    # CSV companions retain metadata, not just the coordinates used by a figure.
    for filename,table in [('results.csv',rows),('per_view.csv',view_rows)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in table for k in row)))
            writer.writeheader()
            writer.writerows(table)
    from .multiscene_plots import plot_experiments
    plot_experiments(out, formats=('png',) if args.output_profile=='compact' else ('png','pdf'))
    (out/'evaluation.json').write_text(json.dumps({**vars(args),'codec_id':identity,
        'selection':'test cameras never used for optimizer/model selection',
        'paired_noise':'identical full point/slot AWGN per trial across layouts',
        'paired_draws':'3/5-draw exports take prefixes of the same ten uniforms; common tie noise',
        'shuffle_scope':'exact same tier histogram/payload; compressed side-stream cost can change',
        'uncertainty':'trials/permutations are repeated measurements, not independent scenes'},indent=2),encoding='utf-8')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='configs/multiscene_tandt_db.json')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--allocations-dir')
    p.add_argument('--out',required=True)
    p.add_argument('--role',choices=['train','heldout','all'],default='all')
    p.add_argument('--scenes',nargs='+',help='optional named scene subset')
    p.add_argument('--draw-counts',nargs='+',type=int,help='diagnostic deployment rules, e.g. 10 5 3; does not retrain tables')
    p.add_argument('--require-allocations',action='store_true',help='fail if a paired scene allocation table is missing')
    p.add_argument('--no-images',action='store_true',help='metrics only; do not save render panels')
    p.add_argument('--color-metrics',action='store_true',help='also measure RGB bias and channel-difference errors')
    p.add_argument('--color-rois',help='optional predeclared scene/view/region boxes JSON')
    p.add_argument('--uniform-only',action='store_true',help='ignore allocation tables; evaluate codec prefixes only')
    p.add_argument('--image-layouts',nargs='+',help='optional panel labels only; does NOT restrict metric layouts')
    p.add_argument('--device',default='cuda')
    p.add_argument('--channel',choices=['awgn','none'],default='awgn')
    p.add_argument('--snrs',nargs='+',type=float,default=[10.])
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--shuffle-seeds',nargs='*',type=int,default=[101,202,303],help='empty list disables shuffles for targeted controls')
    p.add_argument('--deployment-seed',type=int,default=42)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--test-views',type=int,default=0,help='0: ALL test views; positive: explicitly labelled quick subset')
    p.add_argument('--resolution',type=int)
    p.add_argument('--net-bits-per-use',type=float,default=2.)
    p.add_argument('--output-profile',choices=['compact','full'],default='full')
    p.add_argument('--image-views',type=int,default=4,help='saved panels only; does not restrict metric views')
    return p


if __name__=='__main__':
    evaluate(parser().parse_args())
