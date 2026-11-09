"""Calibrate two independent entropy thresholds with shared beta comparisons."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .allocation_distortion import ImageDistortion
from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_evaluation import ExactDeploymentCost, evaluate_map, tier_hash
from .cli import device_for, seed_all
from .data import read_ply
from .entropy_model import tier_map
from .entropy_stage_one import FAMILY as SOURCE_FAMILY, review_package
from .entropy_thresholds import candidate_maps, select_maps, threshold_map
from .entropy_training import tensor_hash
from .model_factory import build_codec
from .modular_experiment import json_write
from .successive_multiscene_inputs import make_scene
from .transport import model_id

FAMILY = 'entropy_independent_thresholds_v1'


def read_json(path): return json.loads(Path(path).read_text(encoding='utf8'))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--out',required=True)
    p.add_argument('--codec-run'); p.add_argument('--manifest')
    p.add_argument('--betas',type=float,nargs='+',default=[.1,.5])
    p.add_argument('--quantiles',type=float,nargs='+',default=[0.,.1,.25,.5,.75,.9,1.])
    p.add_argument('--image-views',type=int,default=2)
    p.add_argument('--cpu-threads',type=int,default=4); p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true'); p.add_argument('--dry-run',action='store_true')
    return p


def read_source(args):
    source,out = Path(args.source).resolve(),Path(args.out).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('source/output must be separate')
    docs = {name:read_json(source/(name+'.json')) for name in
            ('plan','training','complete','objective_reference','information_summary','selection')}
    if docs['plan']['version'] != SOURCE_FAMILY or docs['complete'].get('status') != 'complete':
        raise ValueError('a completed grouped entropy stage-one run is required')
    if not docs['complete']['frozen_codec_verified']: raise ValueError('unverified source codec')
    scores = torch.load(source/'cache/information_scores.pt',map_location='cpu',weights_only=True)
    expected = docs['information_summary']['original_row_score_sha256']
    if (scores.ndim!=1 or not torch.isfinite(scores).all() or (scores<0).any()
            or tensor_hash(scores)!=expected or len(scores)!=docs['training']['scene']['source_gaussians']):
        raise ValueError('source information score identity/shape mismatch')
    if (not args.betas or len(set(args.betas))!=len(args.betas) or
            any(not math.isfinite(b) or b<0 for b in args.betas) or
            args.image_views<0 or args.cpu_threads<1): raise ValueError('invalid beta or runtime settings')
    qs=args.quantiles
    if (len(qs)<2 or qs[0]!=0 or qs[-1]!=1 or sorted(set(qs))!=qs or
            any(not math.isfinite(q) or not 0<=q<=1 for q in qs)): raise ValueError('invalid quantiles')
    settings={k:v for k,v in vars(args).items() if k not in ('out','device','cpu_threads','resume','dry_run')}
    signature=hashlib.sha256(json.dumps(docs,sort_keys=True).encode('utf8')).hexdigest()
    plan=dict(version=FAMILY,settings=settings,source_documents_sha256=signature,
        score_sha256=expected,selection_scope='calibration views only',optimizer_updates=0,
        keep_policy='all_input_rows',threshold_rule='q1 if I<=t1; q2 if t1<I<=t2; q3 otherwise',
        candidate_policy='independent_quantiles_plus_exact_q2_budget_plus_source_eta_and_uniform')
    return source,out,docs,scores,plan


class Measurements:
    """Read-only source reuse, with new hash-keyed caches for exact resume."""
    def __init__(self,source,out,model,scene,args,reference):
        self.source,self.out,self.model,self.scene,self.args = source,out,model,scene,args
        self.reference=reference; self.codec_id=model_id(model)
        self.sources={}
        for path in sorted((source/'evaluation').rglob('metrics.json')):
            record=read_json(path); ident=record['identity']
            scope='test' if 'test' in path.relative_to(source).parts else 'calibration'
            self.sources.setdefault((scope,ident['tier_sha256']),[]).append(path)

    def get(self,q,cameras,scope,images=False):
        local=argparse.Namespace(**vars(self.args)); local.lpips=self.args.lpips if scope=='test' else False
        local.image_views=max(1,self.args.image_views)
        local.allocation_image_objective=ImageDistortion(self.args.distortion,self.reference['lambda_s'])
        key=tier_hash(q); folder=self.out/'evaluation'/scope/key
        expected=dict(codec_id=self.codec_id,tier_sha256=key,label=key,
            views=[str(c.image_name) for c in cameras],trials=local.trials,seed=local.seed,
            snr=local.snr,lpips=local.lpips,images=images,
            image_objective=local.allocation_image_objective.identity())
        cached=folder/'metrics.json'
        if cached.exists():
            record=read_json(cached)
            if record['identity']!=expected: raise ValueError('threshold evaluation identity mismatch')
            return record['metrics'],record.get('origin','local_cache')
        for path in self.sources.get((scope,key),[]):
            record=read_json(path); ident=record['identity']
            if any(ident.get(k)!=expected[k] for k in
                   ('codec_id','tier_sha256','views','trials','seed','snr','lpips')): continue
            objective=ident.get('image_objective',{})
            target=expected['image_objective']
            if any(objective.get(k)!=target[k] for k in ('distortion','protocol')): continue
            # The source q3 reference measured the same images with structural weight zero.
            if objective.get('lambda_s')!=target['lambda_s'] and not (
                    scope=='calibration' and ident['label']=='q3_reference' and objective.get('lambda_s')==0.):
                continue
            if images and (not ident['images'] or self.args.image_views!=self.args.source_image_views): continue
            metrics=copy.deepcopy(record['metrics'])
            for row in [metrics]+metrics.get('per_view_trial',[]):
                row['source_distortion']=row['source_objective_mse']+self.reference['lambda_s']*row['source_structure_error']
            metrics['lambda_s']=self.reference['lambda_s']
            if images:
                for image in (path.parent/'images').glob('*.png'):
                    dest=folder/'images'/image.name; dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(image,dest)
            json_write(cached,dict(identity=expected,metrics=metrics,origin='source_cache',source=str(path)))
            return metrics,'source_cache'
        metrics=evaluate_map(self.model,self.scene,q,cameras,local,
            save_images=images,folder=folder,label=key)
        return metrics,'rendered'


def reconstruct(candidate,scores,rates):
    if candidate['kind']=='uniform': return torch.full((len(scores),),candidate['tier'],dtype=torch.long)
    if candidate['kind']=='source_eta': return tier_map(scores,candidate['eta'],rates)
    return threshold_map(scores,candidate['low'],candidate['high'])


def against_q2(row,base,beta):
    result=dict(cost_saving=1-row['cost']['total_channel_uses']/base['cost']['total_channel_uses'])
    for prefix in ('source','photo'):
        for metric in ('psnr','ssim','lpips'):
            a,b=row['metrics'][prefix+'_'+metric],base['metrics'][prefix+'_'+metric]
            result[prefix+'_'+metric+'_difference']=a-b if a is not None and b is not None else None
    result['rd_difference']=(row['normalized_distortion']-base['normalized_distortion']+
                            beta*(row['normalized_cost']-base['normalized_cost'])) if beta is not None else None
    result['meets_q2_goal']=result['cost_saving']>0 and result['source_psnr_difference']>=0
    return result


@torch.no_grad()
def main(args,max_candidates=None):
    source,out,docs,scores,plan=read_source(args)
    if args.dry_run: print(json.dumps(plan,indent=2)); return plan
    if out.exists():
        if not args.resume: raise FileExistsError('use a new output or --resume')
        if read_json(out/'plan.json')!=plan: raise ValueError('threshold settings/source identity mismatch')
    elif args.resume: raise FileNotFoundError('resume output does not exist')
    old_settings=docs['plan']['settings']
    local=argparse.Namespace(**old_settings)
    local.out=str(out); local.run=args.codec_run or local.run
    local.manifest=args.manifest or local.manifest
    local.device=args.device; local.source_image_views=old_settings['image_views']; local.image_views=args.image_views
    torch.set_num_threads(args.cpu_threads); seed_all(local.seed)
    saved,cfg,spec,old,options,identity=load_inputs(local)
    if json.loads(json.dumps(cfg.to_dict()))!=docs['plan']['codec_config']:
        raise ValueError('source codec configuration changed')
    if identity.get('source_sha256')!=docs['plan']['source'].get('source_sha256'):
        raise ValueError('source codec checkpoint changed')
    model=build_codec(cfg).to(device_for(args.device)); model.load_state_dict(saved['state_dict']); del saved
    model.eval().requires_grad_(False); model.set_training_snrs([local.snr]); codec_id=model_id(model)
    if codec_id!=docs['training']['codec_id']: raise ValueError('source model identity mismatch')
    raw,degree=read_ply(spec['ply']); scene=make_scene(spec,raw,degree,model,options,out,old=old); del raw
    if scene.record!=docs['training']['scene']: raise ValueError('source scene/camera identity mismatch')
    model.set_scene_geometry(scene.geometry)
    reference=copy.deepcopy(docs['objective_reference']); reference['source_beta']=reference.pop('beta')
    if options.net_bits_per_use!=docs['training']['net_bits_per_use']:
        raise ValueError('reliable side cost units changed')
    meter=ExactDeploymentCost(model,scene,codec_id,local.snr,options.net_bits_per_use)
    costs={}
    def cost(q):
        key=tier_hash(q)
        if key not in costs:
            costs[key]=meter.details(q); costs[key].pop('xyz_cache_hit',None)
        return costs[key]
    full=torch.full((len(scores),),3,dtype=torch.long)
    if cost(full)['total_channel_uses']!=reference['c_ref']: raise ValueError('full q3 cost reference changed')
    names=reference['calibration_view_names']; by_name={str(c.image_name):c for c in scene.cameras}
    calibration=[by_name[name] for name in names]
    if set(names)&{str(c.image_name) for c in scene.test_cameras}: raise ValueError('calibration/test overlap')
    json_write(out/'plan.json',plan); json_write(out/'objective_reference.json',reference)
    json_write(out/'information_summary.json',docs['information_summary'])
    json_write(out/'training.json',dict(version=FAMILY,source=str(source),codec_id=codec_id,
        scene=scene.record,optimizer_updates=0,score_sha256=plan['score_sha256']))
    measurements=Measurements(source,out,model,scene,local,reference)
    print('Preparing independent threshold maps and exact total costs...',flush=True)
    candidates={}
    def add(record,q):
        key=tier_hash(q)
        if key not in candidates:
            candidates[key]=dict(record,tier_sha256=key,tier_counts=torch.bincount(q,minlength=4).tolist())
        cost(q)
    for tier in (1,2,3): add(dict(kind='uniform',tier=tier),torch.full_like(full,tier))
    for row in docs['selection']['candidates']:
        q=tier_map(scores,row['eta'],cfg.rates)
        if tier_hash(q)!=row['tier_sha256']: raise ValueError('source eta map identity mismatch')
        add(dict(kind='source_eta',eta=row['eta']),q)
    for record,q in candidate_maps(scores,args.quantiles,cost): add(record,q)
    rows=[]
    for index,candidate in enumerate(candidates.values(),1):
        if max_candidates is not None and index>max_candidates:
            return dict(status='calibration_incomplete',candidates=len(rows))
        start=time.monotonic(); q=reconstruct(candidate,scores,cfg.rates)
        metrics,origin=measurements.get(q,calibration,'calibration')
        c=cost(q); d=metrics['source_distortion']/reference['d_ref']; cn=c['total_channel_uses']/reference['c_ref']
        row=dict(candidate,metrics=metrics,cost=c,normalized_distortion=d,normalized_cost=cn,
            measurement_origin=origin,rd_by_beta={str(b):d+b*cn for b in args.betas})
        rows.append(row)
        json_write(out/'metrics/calibration_progress.json',dict(completed=len(rows),total=len(candidates),candidates=rows))
        print(f"threshold {index}/{len(candidates)}: tiers={row['tier_counts']}, "
              f"cost={c['total_channel_uses']/1e6:.4f}M, D={d:.6f}, "
              f"RD={row['rd_by_beta']}, {origin}, sec={time.monotonic()-start:.2f}",flush=True)
    q2=torch.full_like(full,2); q2key=tier_hash(q2)
    selected=select_maps(rows,args.betas,cost(q2)['total_channel_uses']/reference['c_ref'])
    selection=dict(selections=selected,candidates=rows,unique_maps=len(rows),
        selection_scope='fixed calibration views only; test metrics never select thresholds',
        q2_budget_scope='minimum calibration distortion at actual total cost <= uniform q2')
    # Freeze the decisions on disk before reading or computing any test metric.
    json_write(out/'selection.json',selection); (out/'deployment').mkdir(exist_ok=True)
    test={}; uniforms={}; selections={}
    aliases={name:row['tier_sha256'] for name,row in selected.items()}
    aliases.update({'uniform_q'+str(k):tier_hash(torch.full_like(full,k)) for k in (1,2,3)})
    for label,key in aliases.items():
        q=reconstruct(candidates[key],scores,cfg.rates)
        if label in selected: np.save(out/'deployment'/(label+'_tiers.npy'),q.numpy().astype(np.uint8))
        if key not in test:
            print(f'Test {label}: {len(scene.test_cameras)} heldout views, {local.trials} paired trials.',flush=True)
            metrics,origin=measurements.get(q,scene.test_cameras,'test',images=args.image_views>0 and
                (key in {r['tier_sha256'] for r in selected.values()} or key==q2key))
            c=cost(q)
            test[key]=dict(metrics=metrics,cost=c,normalized_distortion=metrics['source_distortion']/reference['d_ref'],
                normalized_cost=c['total_channel_uses']/reference['c_ref'],measurement_origin=origin)
        row=copy.deepcopy(test[key])
        row['rd_by_beta']={str(b):row['normalized_distortion']+b*row['normalized_cost'] for b in args.betas}
        if label in selected:
            beta=selected[label]['beta']; row.update(beta=beta,threshold_candidate=candidates[key])
            selections[label]=row
        else: uniforms[label]=row
    for label,row in selections.items(): row['against_q2']=against_q2(row,uniforms['uniform_q2'],row['beta'])
    summary=dict(selections=selections,uniform=uniforms,reference=reference,betas=args.betas,
        selection_scope='calibration only',keep_policy='all input PLY rows retained',
        rate_unit='total complex channel uses with inherited reliable side net bits/use',
        entropy_scope='frozen sender information scores',unique_test_maps=len(test))
    json_write(out/'comparison/summary.json',summary)
    if model_id(model)!=codec_id or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('frozen codec identity/gradient boundary changed')
    result=dict(status='complete',family=FAMILY,optimizer_updates=0,frozen_codec_verified=True,
        score_sha256=plan['score_sha256'],betas=args.betas,calibration_maps=len(rows),unique_test_maps=len(test),
        against_q2={k:v['against_q2'] for k,v in selections.items()})
    json_write(out/'complete.json',result); review_package(out); print(json.dumps(result),flush=True)
    return result


if __name__=='__main__': main(parser().parse_args())
