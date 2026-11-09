"""Hard entropy maps selected by calibration RD and evaluated on heldout views."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from .allocation_distortion import ImageDistortion, calibrate
from .budgeted_allocation_evaluation import evaluate_map, tier_hash
from .entropy_model import eta_grid, tier_map
from .modular_experiment import json_write
from .render_objective import spaced_indices


def scored(metrics, cost, reference, beta):
    d = metrics['source_distortion']/reference['d_ref']
    c = cost['total_channel_uses']/reference['c_ref']
    return dict(metrics=metrics, cost=cost, normalized_distortion=d, normalized_cost=c, rd_score=d+beta*c)


@torch.no_grad()
def calibrate_maps(model, scene, scores, args, meter, out):
    calibration = [scene.cameras[i] for i in spaced_indices(len(scene.cameras),args.calibration_views)]
    local = argparse.Namespace(**vars(args)); local.lpips=False
    local.image_views=max(1,args.image_views)
    local.allocation_image_objective=ImageDistortion(args.distortion,0.)
    full = torch.full((len(scene.raw),),3,dtype=torch.long)
    print(f'RD reference: full q3, {len(calibration)} calibration views, {args.trials} paired trials.',flush=True)
    metrics = evaluate_map(model,scene,full,calibration,local,folder=out/'evaluation/reference',label='q3_reference')
    reference = calibrate(metrics['source_objective_mse'],metrics['source_structure_error'],
        mode=args.distortion,share=args.structure_share,lambda_s=args.lambda_s)
    cost = meter.details(full)
    reference.update(c_ref=cost['total_channel_uses'],beta=args.beta,
        calibration_view_names=[str(c.image_name) for c in calibration],q3_cost=cost,
        selection_scope='calibration views only',distortion_target='fixed input PLY renders')
    local.allocation_image_objective=ImageDistortion(args.distortion,reference['lambda_s'])
    # Recombine the already measured q3 components with the one fixed structural weight.
    metrics=copy.deepcopy(metrics)
    for row in [metrics]+metrics.get('per_view_trial',[]):
        row['source_distortion']=row['source_objective_mse']+reference['lambda_s']*row['source_structure_error']
    metrics['lambda_s']=reference['lambda_s']
    seen={tier_hash(full):scored(metrics,cost,reference,args.beta)}
    baselines={'uniform_q3':seen[tier_hash(full)]}
    for q in (1,2):
        tiers=torch.full_like(full,q);key=tier_hash(tiers)
        result=evaluate_map(model,scene,tiers,calibration,local,
            folder=out/'evaluation/calibration'/key,label=key)
        seen[key]=scored(result,meter.details(tiers),reference,args.beta)
        baselines['uniform_q'+str(q)]=seen[key]
    etas=args.etas if args.etas is not None else eta_grid(scores,model.cfg.rates,args.eta_count)
    candidates=[]
    for index,eta in enumerate(etas):
        tiers=tier_map(scores,eta,model.cfg.rates);key=tier_hash(tiers)
        reused=key in seen
        if not reused:
            result=evaluate_map(model,scene,tiers,calibration,local,
                folder=out/'evaluation/calibration'/key,label=key)
            seen[key]=scored(result,meter.details(tiers),reference,args.beta)
        row=dict(eta=eta,tier_sha256=key,tier_counts=torch.bincount(tiers,minlength=4).tolist(),
            reused=reused,**seen[key])
        candidates.append(row)
        print(f"eta {index+1}/{len(etas)}: {eta:.6g}, tiers={row['tier_counts']}, "
              f"cost={row['cost']['total_channel_uses']/1e6:.4f}M, RD={row['rd_score']:.6f}",flush=True)
    best=min(candidates,key=lambda r:(r['rd_score'],r['cost']['total_channel_uses'],r['eta']))
    selected=tier_map(scores,best['eta'],model.cfg.rates)
    record=dict(selected_eta=best['eta'],selected_tier_sha256=best['tier_sha256'],
        selected_calibration=best,selection_scope='minimum calibration RD among eta candidates',
        candidates=candidates,uniform_calibration=baselines,unique_maps=len(seen))
    json_write(out/'objective_reference.json',reference)
    json_write(out/'selection.json',record)
    (out/'deployment').mkdir(exist_ok=True)
    np.save(out/'deployment/selected_tiers.npy',selected.numpy().astype(np.uint8))
    return selected,record,reference


@torch.no_grad()
def test_selected(model,scene,selected,args,meter,out,reference,selection):
    local=argparse.Namespace(**vars(args))
    local.image_views=max(1,args.image_views)
    local.allocation_image_objective=ImageDistortion(args.distortion,reference['lambda_s'])
    rows={}
    for label,q in [('selected',selected)]+[('uniform_q'+str(k),torch.full_like(selected,k)) for k in (1,2,3)]:
        print(f'Test {label}: {len(scene.test_cameras)} heldout views, {args.trials} paired trials.',flush=True)
        metrics=evaluate_map(model,scene,q,scene.test_cameras,local,
            save_images=args.image_views>0 and label in ('selected','uniform_q2'),
            folder=out/'evaluation/test'/label,label=label)
        rows[label]=scored(metrics,meter.details(q),reference,args.beta)
    chosen,base=rows['selected'],rows['uniform_q2']
    against=dict(cost_saving=1-chosen['cost']['total_channel_uses']/base['cost']['total_channel_uses'],
        source_psnr_difference=chosen['metrics']['source_psnr']-base['metrics']['source_psnr'],
        photo_psnr_difference=chosen['metrics']['photo_psnr']-base['metrics']['photo_psnr'],
        rd_difference=chosen['rd_score']-base['rd_score'])
    for prefix in ('source','photo'):
        for metric in ('ssim','lpips'):
            a,b=chosen['metrics'][prefix+'_'+metric],base['metrics'][prefix+'_'+metric]
            against[prefix+'_'+metric+'_difference']=a-b if a is not None and b is not None else None
    against['meets_q2_goal']=against['cost_saving']>0 and against['source_psnr_difference']>=0
    summary=dict(test=rows,against_q2=against,reference=reference,selection=selection,
        rate_unit='total complex channel uses; reliable sides use the recorded net bits/use assumption',
        entropy_scope='sender-only statistical scores; not actual transmitted bits',
        keep_policy='all input PLY rows retained')
    json_write(out/'comparison/summary.json',summary)
    return summary
