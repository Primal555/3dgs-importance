"""Frozen-codec isolation of deletion and positive-tier deployment decisions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from .allocation_distortion import ImageDistortion
from .budgeted_allocation_evaluation import ExactDeploymentCost,evaluate_map,tier_hash
from .budgeted_allocation_inputs import load_inputs
from .cli import device_for,seed_all
from .data import read_ply
from .model_factory import build_codec
from .modular_experiment import json_write
from .relation_benefit_experiment import FAMILY,unified_cost
from .successive_multiscene_inputs import make_scene
from .transport import model_id

VERSION='relation_deployment_controls_v1'
BASELINES=('selected','uniform_q1','uniform_q2')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--run');p.add_argument('--manifest');p.add_argument('--device',default='cuda')
    p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--resume',action='store_true')
    return p


def controlled_maps(q):
    q=torch.as_tensor(q)
    if q.ndim!=1 or q.dtype not in (torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64):
        raise ValueError('original-row integer tier vector required')
    if not q.numel() or ((q<0)|(q>3)).any():raise ValueError('nonempty q0--q3 tier vector required')
    q=q.detach().cpu().long()
    return dict(restore_q0_to_q1=q.masked_fill(q==0,1),
                kept_uniform_q2=torch.where(q>0,2,0))


def _digest(document):
    return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def read_source(source):
    source=Path(source).resolve()
    training=_read(source/'training.json');plan=_read(source/'plan.json')
    complete=_read(source/'complete.json');review=_read(source/'review_summary.json')
    if (complete.get('status')!='complete' or complete.get('family')!=FAMILY
            or not complete.get('frozen_codec_verified') or training.get('version')!=FAMILY
            or plan.get('version')!=FAMILY or training['codec_id']!=complete['codec_id']
            or plan['codec_id']!=complete['codec_id'] or training['settings']!=plan['settings']):
        raise ValueError('completed, consistent relation context experiment required')
    array=np.load(source/'deployment/selected_tiers.npy',allow_pickle=False)
    if array.dtype.kind not in 'iu':raise ValueError('integer deployment tiers required')
    if array.ndim!=1 or len(array)!=training['source_gaussians'] or not len(array) or int(array.min())<0 or int(array.max())>3:
        raise ValueError('deployment tier count/range differs from source scene')
    q=torch.from_numpy(array.astype(np.int64))
    if tier_hash(q)!=complete['tier_sha256'] or review['rows']['selected']['tier_sha256']!=tier_hash(q):
        raise ValueError('deployed tier identity mismatch')
    reference=training['objective_reference'];settings=training['settings']
    for name in ('d_ref','c_ref','lambda_s','beta'):
        value=float(reference[name])
        if not math.isfinite(value) or value<0 or (name in ('d_ref','c_ref') and value==0):
            raise ValueError('invalid fixed distortion/cost reference')
    if settings['snr']!=10. or settings['beta']!=reference['beta'] or settings['trials']<1:
        raise ValueError('source SNR/beta/trials mismatch')
    if not training.get('test_seed'):test_seed=int(settings['seed'])+800003
    else:test_seed=int(training['test_seed'])
    record=dict(path=source,training=training,plan=plan,complete=complete,
        reference=reference,tiers=q,test_seed=test_seed,baselines={},review=review,
        identity=dict(plan_sha256=_digest(plan),training_sha256=_digest(training),
                      complete_sha256=_digest(complete),review_sha256=_digest(review),
                      tier_sha256=tier_hash(q),baseline_evaluation_sha256={}))
    maps=dict(selected=q,uniform_q1=torch.ones_like(q),uniform_q2=torch.full_like(q,2))
    for label in BASELINES:
        document=_read(source/f'evaluation/{label}/metrics.json')
        _check_evaluation(document,maps[label],training['test_view_names'],record,label)
        if (document['metrics']['tier_sha256']!=review['rows'][label]['tier_sha256']
                or review['rows'][label]['codec_id']!=complete['codec_id']):
            raise ValueError('source review/evaluation tier identity mismatch')
        record['baselines'][label]=document['metrics']
        record['identity']['baseline_evaluation_sha256'][label]=_digest(document)
    return record


def _check_evaluation(document,q,views,record,label):
    identity=document['identity'];settings=record['training']['settings']
    expected=dict(codec_id=record['complete']['codec_id'],tier_sha256=tier_hash(q),label=label,
        views=views,trials=settings['trials'],seed=record['test_seed'],snr=settings['snr'],
        lpips=settings['lpips'],image_objective=ImageDistortion('mse_ssim',record['reference']['lambda_s']).identity())
    if any(identity.get(k)!=v for k,v in expected.items()):raise ValueError('cached evaluation protocol mismatch: '+label)
    metrics=document['metrics']
    if metrics['codec_id']!=expected['codec_id'] or metrics['tier_sha256']!=expected['tier_sha256']:
        raise ValueError('cached metric identity mismatch')
    checked=subset_metrics(metrics,views)
    for name,value in checked.items():
        if name in metrics and isinstance(value,(int,float)) and not math.isclose(value,metrics[name],rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('cached metric aggregation mismatch: '+name)


def subset_metrics(metrics,views):
    if not views or len(set(views))!=len(views):raise ValueError('unique, nonempty scope cameras required')
    observations=[r for r in metrics['per_view_trial'] if r['view'] in set(views)]
    expected={(v,t) for v in views for t in range(metrics['trials'])}
    keys=[(r['view'],r['trial']) for r in observations]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('missing/duplicate view-trial observations')
    result={k:metrics[k] for k in ('trials','tier_counts','tier_shares','codec_id','tier_sha256','protocol')}
    result.update(views=len(views),per_view_trial=observations,lambda_s=metrics['lambda_s'])
    for name in observations[0]:
        if name in ('view','trial'):continue
        values=[r[name] for r in observations]
        if all(v is None for v in values):result[name]=None
        elif any(v is None or not math.isfinite(v) for v in values):raise ValueError('invalid image metric '+name)
        else:result[name]=float(np.mean(values))
    return result


def _paths(args,record):
    out=Path(args.out).resolve()
    for source in (record['path'],args.run or record['training']['settings']['run'],
                   record['training']['spec'].get('ply'),record['training']['spec'].get('source')):
        if source is None:continue
        source=Path(source).resolve()
        if out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('controls output must be separate from all sources')
    if args.cpu_threads<1:raise ValueError('positive CPU thread count required')
    if out.exists() and not args.resume:raise FileExistsError('new controls output or --resume required')
    if args.resume and not (out/'plan.json').is_file():raise FileNotFoundError('controls plan missing for resume')
    return out


def _enrich(metrics,cost,reference):
    return dict(metrics,**cost,beta=reference['beta'],d_ref=reference['d_ref'],c_ref=reference['c_ref'],
        rd_score=metrics['source_distortion']/reference['d_ref']+
                 reference['beta']*cost['total_channel_uses']/reference['c_ref'])


def _delta(row,baseline):
    keys=('source_psnr','source_ssim','source_lpips','photo_psnr','photo_ssim','photo_lpips','rd_score')
    result={k+'_difference':row[k]-baseline[k] if row[k] is not None and baseline[k] is not None else None for k in keys}
    result['cost_saving']=1-row['total_channel_uses']/baseline['total_channel_uses']
    return result


def _archive(out):
    temp=out/'review.tar.gz.tmp'
    with tarfile.open(temp,'w:gz') as tar:
        for name in ('plan.json','complete.json','review_summary.json'):
            tar.add(out/name,arcname=name)
        for folder in ('evaluation','maps'):
            for path in sorted((out/folder).rglob('*')):
                if path.is_file():tar.add(path,arcname=path.relative_to(out).as_posix())
    temp.replace(out/'review.tar.gz')


def run_loaded(args,model,scene,record):
    out=_paths(args,record);training=record['training'];reference=record['reference']
    torch.set_num_threads(args.cpu_threads);seed_all(training['settings']['seed'])
    model.eval().requires_grad_(False);model.set_scene_geometry(scene.geometry);cid=model_id(model)
    if (cid!=record['complete']['codec_id'] or len(scene.raw)!=len(record['tiers'])
            or scene.fingerprint!=training['fingerprint'] or scene.spec['name']!=training['spec']['name']
            or scene.degree!=model.cfg.sh_degree or tuple(model.cfg.rates)!=(0,8,24,48)
            or scene.white_background!=training['white_background']):
        raise ValueError('source codec/scene identity mismatch')
    for key in ('geometry','sh_degree','resolution','images','train_view_names','validation_view_names','camera_fingerprints'):
        if key in training and key in scene.record and json.loads(json.dumps(training[key]))!=json.loads(json.dumps(scene.record[key])):
            raise ValueError('source scene/render setting mismatch: '+key)
    names=lambda cams:[str(c.image_name) for c in cams]
    lookup={str(c.image_name):c for c in scene.cameras}
    prep_names=training['preparation_view_names'];test_names=training['test_view_names']
    if (len(lookup)!=len(scene.cameras) or not set(prep_names)<=lookup.keys()
            or len(set(prep_names))!=len(prep_names) or names(scene.test_cameras)!=test_names
            or set(prep_names)&set(test_names)):
        raise ValueError('original preparation/test camera identities must be preserved')
    preparation=[lookup[name] for name in prep_names];cameras=preparation+list(scene.test_cameras)
    settings=training['settings']
    opts=SimpleNamespace(snr=settings['snr'],trials=settings['trials'],seed=record['test_seed'],
        lpips=settings['lpips'],image_views=2,
        allocation_image_objective=ImageDistortion('mse_ssim',reference['lambda_s']))
    q=record['tiers'];maps=dict(selected=q,uniform_q1=torch.ones_like(q),uniform_q2=torch.full_like(q,2),**controlled_maps(q))
    meter=ExactDeploymentCost(model,scene,cid,opts.snr,2.)
    if unified_cost(meter,torch.full_like(q,3))['total_channel_uses']!=reference['c_ref']:
        raise ValueError('original C_ref and current unified q3 accounting differ')
    costs={label:unified_cost(meter,tiers) for label,tiers in maps.items()}
    for label in BASELINES:
        if costs[label]['total_channel_uses']!=record['review']['rows'][label]['total_channel_uses']:
            raise ValueError('source baseline cost accounting changed')
    plan=dict(version=VERSION,source=str(record['path']),source_identity=record['identity'],
        codec_id=cid,source_step=record['complete']['step'],spec=scene.spec,reference=reference,
        preparation_views=prep_names,test_views=test_names,seed=opts.seed,trials=opts.trials,snr=opts.snr,
        lpips=opts.lpips,optimizer_updates=0,cost_scope='whole scene, identical across camera subsets',
        interventions=dict(restore_q0_to_q1='replace q0 by q1; preserve original q1/q2/q3',
            kept_uniform_q2='preserve q0 mask; replace retained tiers by q2'),
        interpretation='each intervention includes codec/context changes; effects are not additive')
    if args.resume and _read(out/'plan.json')!=json.loads(json.dumps(plan)):raise ValueError('controls resume identity mismatch')
    out.mkdir(parents=True,exist_ok=True);json_write(out/'plan.json',plan)
    (out/'maps').mkdir(exist_ok=True)
    scopes={name:dict(rows={},reused_source_baselines=name=='test') for name in ('preparation','test')}
    for label,tiers in maps.items():
        if label in BASELINES:
            print(f'Evaluating {label}: {len(preparation)} preparation views; reuse {len(test_names)} test views.',flush=True)
            prep=evaluate_map(model,scene,tiers,preparation,opts,folder=out/f'evaluation/preparation/{label}',label=label)
            test=record['baselines'][label]
        else:
            np.save(out/f'maps/{label}.npy',tiers.numpy().astype(np.uint8))
            print(f'Evaluating {label}: {len(preparation)} preparation + {len(test_names)} test views, {opts.trials} paired trials.',flush=True)
            combined=evaluate_map(model,scene,tiers,cameras,opts,save_images=True,
                folder=out/f'evaluation/combined/{label}',label=label)
            prep=subset_metrics(combined,prep_names);test=subset_metrics(combined,test_names)
        for scope,metrics in (('preparation',prep),('test',test)):
            scopes[scope]['rows'][label]=_enrich(metrics,costs[label],reference)
            identity=dict(codec_id=cid,tier_sha256=tier_hash(tiers),label=label,scope=scope,
                views=prep_names if scope=='preparation' else test_names,trials=opts.trials,seed=opts.seed,
                snr=opts.snr,lpips=opts.lpips,image_objective=opts.allocation_image_objective.identity())
            if scope=='test' or label not in BASELINES:
                json_write(out/f'evaluation/{scope}/{label}/metrics.json',dict(identity=identity,metrics=metrics))
    for data in scopes.values():
        data['deltas']={label:{baseline:_delta(data['rows'][label],data['rows'][baseline]) for baseline in BASELINES}
                        for label in controlled_maps(q)}
    frozen=model_id(model)==cid
    if not frozen:raise RuntimeError('codec changed during fixed-map controls')
    result=dict(status='complete',family=VERSION,source_step=record['complete']['step'],
        codec_id=cid,frozen_codec_verified=frozen,optimizer_updates=0,source_tier_sha256=tier_hash(q),
        tier_counts={name:torch.bincount(tiers,minlength=4).tolist() for name,tiers in maps.items()})
    json_write(out/'review_summary.json',dict(result=result,scopes=scopes))
    json_write(out/'complete.json',result);_archive(out)
    for scope,data in scopes.items():
        for label,row in data['rows'].items():
            print(f"{scope}/{label}: source PSNR={row['source_psnr']:.4f}, photo PSNR={row['photo_psnr']:.4f}, "
                  f"cost={row['total_channel_uses']/1e6:.6f}M, RD={row['rd_score']:.6f}",flush=True)
    return result


def main(args):
    record=read_source(args.source);_paths(args,record)
    inputs=SimpleNamespace(**record['training']['settings'])
    inputs.run=args.run or inputs.run;inputs.out=args.out;inputs.manifest=args.manifest
    saved,cfg,spec,old,scene_options,_=load_inputs(inputs)
    model=build_codec(cfg).to(device_for(args.device));model.load_state_dict(saved['state_dict'])
    model.eval().requires_grad_(False);model.set_training_snrs([10.])
    raw,degree=read_ply(spec['ply'])
    with tempfile.TemporaryDirectory(prefix='relation-controls-') as temporary:
        scene=make_scene(spec,raw,degree,model,scene_options,temporary,old=old)
        return run_loaded(args,model,scene,record)


if __name__=='__main__':main(parser().parse_args())
