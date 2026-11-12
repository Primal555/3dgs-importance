"""Paired positive-tier permutation control for completed successive allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import torch
from .allocation_distortion import ImageDistortion
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_evaluation import evaluate_map, tier_hash
from .cli import device_for
from .data import read_ply
from .model_factory import build_codec
from .modular_allocation_shuffle import shuffle_retained
from .modular_experiment import json_write
from .student_digital_cost import DigitalDeploymentCost
from .successive_multiscene_inputs import make_scene
from .transport import model_id


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def paired_difference(a,b):
    def index(row): return {(r['trial'],r['view']):r for r in row['per_view_trial']}
    left,right=index(a),index(b)
    if left.keys()!=right.keys(): raise ValueError('unmatched test trials/views')
    keys=['source_psnr','source_ssim','photo_psnr','photo_ssim','source_distortion']
    if next(iter(left.values())).get('source_lpips') is not None:
        keys+=['source_lpips','photo_lpips']
    return {k:float(np.mean([left[i][k]-right[i][k] for i in left])) for k in keys}


def main(args):
    source,out=Path(args.allocation).resolve(),Path(args.out).resolve()
    if out==source or source.is_relative_to(out): raise ValueError('shuffle output must not contain source')
    complete,plan=read(source/'complete.json'),read(source/'plan.json')
    if complete['status']!='complete' or not complete['frozen_codec_verified']:
        raise ValueError('completed frozen-codec allocation required')
    q=torch.from_numpy(np.load(source/'deployment/selected_tiers.npy',allow_pickle=False).astype(np.int64))
    if tier_hash(q)!=complete['tier_sha256']: raise ValueError('selected map identity mismatch')
    shuffled=shuffle_retained(q,args.shuffle_seed)
    if not torch.equal(q==0,shuffled==0) or not torch.equal(torch.bincount(q,minlength=4),torch.bincount(shuffled,minlength=4)):
        raise RuntimeError('shuffle changed deletion mask or tier counts')
    record=dict(version='first_order_positive_shuffle_v1',source=str(source),
        source_complete_sha256=hashlib.sha256((source/'complete.json').read_bytes()).hexdigest(),
        source_plan_sha256=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),
        source_tier_sha256=tier_hash(q),shuffle_seed=args.shuffle_seed,shuffled_tier_sha256=tier_hash(shuffled),
        protocol='digital_side_file_v1',optimizer_updates=0)
    if (out/'plan.json').exists():
        if read(out/'plan.json')!=record: raise ValueError('shuffle source/settings changed')
        if (out/'complete.json').exists(): return read(out/'complete.json')
    elif out.exists(): raise FileExistsError('use a new shuffle output')
    options=argparse.Namespace(**plan['settings']);options.out=str(out);options.device=args.device
    saved,cfg,spec,old,scene_args,info=load_inputs(options)
    device=device_for(args.device);torch.set_num_threads(options.cpu_threads)
    model=build_codec(cfg).to(device);model.load_state_dict(saved['state_dict']);del saved
    model.eval().requires_grad_(False);model.set_training_snrs([10.])
    identity=model_id(model)
    if identity!=complete['codec_id']: raise ValueError('source codec identity mismatch')
    out.mkdir(parents=True,exist_ok=True);json_write(out/'plan.json',record)
    raw,degree=read_ply(spec['ply']);scene=make_scene(spec,raw,degree,model,scene_args,out,old=old);del raw
    if len(q)!=len(scene.raw): raise ValueError('source point count mismatch')
    options.allocation_image_objective=ImageDistortion('mse_ssim',complete['lambda_s'])
    meter=DigitalDeploymentCost(model,scene,identity,options.snr,scene_args.net_bits_per_use)
    c_ref=meter.details(torch.full_like(q,3))['total_channel_uses']
    rows={}
    for label,tiers in [('selected',q),('shuffled_positive',shuffled)]:
        metrics=evaluate_map(model,scene,tiers,scene.test_cameras,options,save_images=True,
            folder=out/'evaluation'/label,label=label)
        cost=meter.details(tiers)
        row=dict(metrics,**cost,d_ref=complete['d_ref'],c_ref=c_ref,beta=complete['beta'])
        row['rd_score']=row['source_distortion']/row['d_ref']+row['beta']*row['total_channel_uses']/c_ref
        rows[label]=row
        print(f'{label}: source PSNR={row["source_psnr"]:.4f}, photo PSNR={row["photo_psnr"]:.4f}, '
              f'cost={row["total_channel_uses"]/1e6:.6f}M, RD={row["rd_score"]:.6f}',flush=True)
    a,b=rows['selected'],rows['shuffled_positive']
    if a['payload_complex_symbols']!=b['payload_complex_symbols'] or a['xyz_bytes']!=b['xyz_bytes']:
        raise RuntimeError('shuffle payload/XYZ accounting changed')
    if model_id(model)!=identity: raise RuntimeError('frozen codec changed')
    comparison=dict(paired_selected_minus_shuffled=paired_difference(a,b),
        total_cost_difference=a['total_channel_uses']-b['total_channel_uses'],
        rd_difference=a['rd_score']-b['rd_score'],
        changed_positive_fraction=float((q[q>0]!=shuffled[q>0]).float().mean()) if (q>0).any() else 0.,
        fixed_q0_mask=True,fixed_tier_counts=True,fixed_attribute_symbol_count=True,
        interpretation='paired effect of this permutation; metadata and block scales remeasured; one permutation is not a distribution estimate')
    result=dict(status='complete',**record,codec_id=identity,comparison=comparison,
        views=len(scene.test_cameras),noise_trials=options.trials,side_link_simulated=False)
    json_write(out/'review_summary.json',dict(completion=result,rows=rows,comparison=comparison))
    np.save(out/'shuffled_tiers.npy',shuffled.numpy().astype(np.uint8),allow_pickle=False)
    json_write(out/'complete.json',result)
    with tarfile.open(out/'review.tar.gz.tmp','w:gz') as archive:
        for file in sorted(out.rglob('*')):
            if file.is_file() and file.suffix in ('.json','.png','.npy'):
                archive.add(file,arcname=file.relative_to(out).as_posix(),recursive=False)
    (out/'review.tar.gz.tmp').replace(out/'review.tar.gz')
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--allocation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--shuffle-seed',type=int,default=20261112)
    return p


if __name__=='__main__':main(parser().parse_args())
