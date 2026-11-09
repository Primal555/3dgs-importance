"""Independent grouped-entropy tier allocation with a frozen successive codec."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

import torch

from .budgeted_allocation_inputs import load_inputs
from .budgeted_allocation_evaluation import ExactDeploymentCost
from .cli import device_for, seed_all
from .data import read_ply
from .entropy_evaluation import calibrate_maps, test_selected
from .entropy_training import atomic_save, cache_latents, tensor_hash, train_density
from .model_factory import build_codec
from .modular_experiment import json_write
from .successive_multiscene_inputs import make_scene
from .transport import model_id

FAMILY='grouped_entropy_stage_one_v1'


def block_signature(groups):
    # The final batch can contain a shorter, unpadded single block.
    parts=[dict(shape=list(ids.shape),sha256=tensor_hash(ids)) for ids in groups]
    return hashlib.sha256(json.dumps(parts,sort_keys=True).encode('utf8')).hexdigest()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','scene','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--manifest');p.add_argument('--device',default='cuda')
    p.add_argument('--groups',nargs='+',type=int,default=[16,16,16])
    for key,value in [('hidden',128),('mixtures',3),('epochs',20),('batch-size',4096),
        ('eta-count',9),('calibration-views',4),('test-views',32),('trials',4),('image-views',2),
        ('cpu-threads',4),('seed',42)]:p.add_argument('--'+key,type=int,default=value)
    for key,value in [('bin-width',.25),('validation-fraction',.2),('lr',1e-3),('min-lr',1e-4),
        ('grad-clip',5.),('beta',.1),('snr',10.),('structure-share',.2)]:
        p.add_argument('--'+key,type=float,default=value)
    p.add_argument('--etas',type=float,nargs='+')
    p.add_argument('--lambda-s',type=float)
    p.add_argument('--distortion',choices=('mse','mse_ssim'),default='mse_ssim')
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.set_defaults(window_views=1)
    return p


def validate(args):
    if min(args.hidden,args.mixtures,args.epochs,args.batch_size,args.calibration_views,
           args.trials,args.cpu_threads,*args.groups)<1 or min(args.test_views,args.image_views)<0:
        raise ValueError('invalid model/training/evaluation counts')
    if (not 0<args.validation_fraction<1 or not 0<args.structure_share<1 or args.eta_count<2
            or not math.isfinite(args.beta) or args.beta<0 or args.snr!=10.):
        raise ValueError('invalid validation fraction, fixed-beta objective or 10dB condition')
    if (not 0<args.min_lr<=args.lr or any(not math.isfinite(v) or v<=0
            for v in (args.lr,args.min_lr,args.bin_width,args.grad_clip))):
        raise ValueError('invalid entropy optimizer or quantization interval')
    if args.etas is not None and any(not math.isfinite(v) or v<=0 for v in args.etas):
        raise ValueError('eta candidates must be finite and positive')
    if args.lambda_s is not None and (not math.isfinite(args.lambda_s) or args.lambda_s<0):
        raise ValueError('invalid structural weight')
    if args.distortion=='mse' and args.lambda_s not in (None,0.):raise ValueError('MSE requires lambda_s=0')


@torch.no_grad()
def score_latents(model,y,batch):
    scores=[];components=torch.zeros(model.dim,dtype=torch.float64)
    for part in y.split(batch):
        bits=model.component_bits(part).cpu()
        if not torch.isfinite(bits).all() or (bits<0).any():raise FloatingPointError('invalid information scores')
        scores.append(bits.sum(-1));components+=bits.sum(0)
    score=torch.cat(scores)
    return score,dict(count=len(y),mean_bits_per_gaussian=float(score.mean()),
        bits_per_dimension=(components/len(y)).tolist(),
        quantiles=dict(zip(['min','p05','p25','p50','p75','p95','max'],
            torch.quantile(score,torch.tensor([0.,.05,.25,.5,.75,.95,1.],dtype=torch.float64)).tolist())))


def review_package(out):
    paths=[out/name for name in ('plan.json','training.json','complete.json','selection.json',
        'objective_reference.json','information_summary.json')]
    for name in ('metrics','comparison','evaluation'):
        paths.extend(p for p in (out/name).rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.png'))
    temporary=out/'review.tar.gz.tmp'
    with tarfile.open(temporary,'w:gz') as archive:
        for path in sorted(paths):
            if path.is_file():archive.add(path,arcname=path.relative_to(out).as_posix())
    temporary.replace(out/'review.tar.gz')


def main(args,max_epochs=None):
    validate(args)
    saved,cfg,spec,old,options,source=load_inputs(args)
    if sum(args.groups)!=cfg.latent_dim:raise ValueError('entropy groups must sum to actual codec latent_dim')
    if tuple(cfg.rates)!=(0,8,24,48) or cfg.channel_variant!='successive':
        raise ValueError('stage one requires the existing successive 0/8/24/48 codec')
    settings={k:v for k,v in vars(args).items() if k not in ('out','device','resume','dry_run','cpu_threads')}
    plan=json.loads(json.dumps(dict(version=FAMILY,source=source,spec=spec,settings=settings,
        codec_config=cfg.to_dict(),keep_policy='all_input_rows',entropy_target='frozen_y_interval_mass',
        weights_selected='final_epoch',gradient_scope='density_model_only',writes_output=not args.dry_run)))
    if args.dry_run:print(json.dumps(plan,indent=2));return plan
    out=Path(args.out).resolve()
    if out.exists():
        if not args.resume:raise FileExistsError('use a new output or --resume')
        if not (out/'plan.json').exists():raise FileNotFoundError('missing entropy experiment plan')
        if json.loads((out/'plan.json').read_text(encoding='utf8'))!=plan:
            raise ValueError('entropy experiment settings/source identity mismatch')
    else:
        if args.resume:raise FileNotFoundError('resume output does not exist')
        json_write(out/'plan.json',plan)
    torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    device=device_for(args.device);args.device=str(device)
    model=build_codec(cfg).to(device);model.load_state_dict(saved['state_dict']);del saved
    model.eval().requires_grad_(False);model.set_training_snrs([10.]);identity=model_id(model)
    raw,degree=read_ply(spec['ply'])
    scene=make_scene(spec,raw,degree,model,options,out,old=old);del raw
    model.set_scene_geometry(scene.geometry)
    record=dict(version=FAMILY,scene=scene.record,codec_id=identity,settings=settings,
                net_bits_per_use=options.net_bits_per_use)
    path=out/'training.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8'))!=record:
        raise ValueError('entropy scene identity changed')
    json_write(path,record)
    cache_path=out/'cache/latents.pt'
    cache_identity=dict(codec_id=identity,scene=scene.record,
        block_order_sha256=block_signature(scene.group_ids))
    if cache_path.exists():
        cached=torch.load(cache_path,map_location='cpu',weights_only=True)
        if cached['identity']!=cache_identity:raise ValueError('semantic cache identity mismatch')
        y,blocks=cached['y'],cached['blocks'];del cached
    else:
        print('Encoding retained source PLY once in original codec blocks...',flush=True)
        y,blocks=cache_latents(model,scene)
        atomic_save(dict(identity=cache_identity,y=y,blocks=blocks),cache_path)
    density,logs=train_density(y,blocks,args,out,cache_identity,resume=args.resume,max_epochs=max_epochs)
    if logs[-1]['epoch']<args.epochs:
        return dict(status='density_incomplete',epoch=logs[-1]['epoch'])
    atomic_save(dict(version=FAMILY,config=density.configuration(),state_dict=density.state_dict(),
        source=cache_identity,epoch=logs[-1]['epoch']),out/'entropy_model.pt')
    scores,information=score_latents(density,y,args.batch_size)
    information.update(model_configuration=density.configuration(),
        entropy_parameters=sum(p.numel() for p in density.parameters()),
        original_row_score_sha256=tensor_hash(scores),meaning='statistical interval information, bits/G')
    json_write(out/'information_summary.json',information)
    atomic_save(scores,out/'cache/information_scores.pt')
    del y,density
    meter=ExactDeploymentCost(model,scene,identity,args.snr,options.net_bits_per_use)
    selected,selection,reference=calibrate_maps(model,scene,scores,args,meter,out)
    summary=test_selected(model,scene,selected,args,meter,out,reference,selection)
    if model_id(model)!=identity or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('frozen codec identity/gradient boundary changed')
    result=dict(status='complete',family=FAMILY,epochs=logs[-1]['epoch'],
        entropy_updates=logs[-1]['updates'],selected_eta=selection['selected_eta'],
        tier_sha256=selection['selected_tier_sha256'],frozen_codec_verified=True,
        against_q2=summary['against_q2'])
    json_write(out/'complete.json',result);review_package(out)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':main(parser().parse_args())
