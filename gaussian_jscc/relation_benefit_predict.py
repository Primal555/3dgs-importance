"""Zero-update relation allocation for a separately selected source scene.

One finite-noiseless q1/q2/q3 preparation builds the sender's CPU input bank.
The trained relation network and every fitted scale remain frozen throughout.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from .allocation_distortion import ImageDistortion
from .budgeted_allocation_evaluation import ExactDeploymentCost, evaluate_map, tier_hash
from .budgeted_allocation_state import load_checkpoint
from .cli import device_for, seed_all
from .data import read_ply
from .model_factory import build_codec
from .modular_experiment import json_write
from .multiscene import manifest
from .relation_benefit_experiment import FAMILY, _archive, unified_cost
from .relation_benefit_features import prepare_bank
from .relation_benefit_learning import marginal_cost_proxy, choose_tiers
from .relation_benefit_model import RelationBenefitNetwork
from .render_objective import spaced_indices
from .successive_multiscene_inputs import make_scene
from .successive_transfer import load_source
from .transport import model_id


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','network','manifest','scene','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--snr',type=float,default=10.)
    p.add_argument('--beta',type=float,default=.1)
    for name,default in (('views',4),('test-views',32),('trials',4),
                         ('inference-batch',4096),('cpu-threads',4),('seed',42)):
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--lpips',action=argparse.BooleanOptionalAction,default=True)
    return p


def _separate(out, source):
    source=Path(source).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('prediction output and source/network/data paths must be separate')


def validate(args):
    out=Path(args.out).resolve()
    for source in (args.run,args.network):_separate(out,source)
    if out.exists():raise FileExistsError('prediction requires a new output directory')
    if (args.snr!=10. or not math.isfinite(args.beta) or args.beta<0 or
            min(args.views,args.test_views,args.trials,args.inference_batch,args.cpu_threads)<1):
        raise ValueError('SNR10, finite beta>=0 and positive inference budgets required')
    return out


def _weights_digest(weights):
    digest=hashlib.sha256()
    for name,value in sorted(weights.items()):
        value=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _network(checkpoint, codec_id, device, codec_config):
    plan=checkpoint.get('plan',{})
    if plan.get('version')!=FAMILY:raise ValueError('relation_benefit_context_v2 network required')
    if plan.get('codec_id')!=codec_id:raise ValueError('network and source codec_id differ')
    if json.loads(json.dumps(codec_config))!=json.loads(json.dumps(plan.get('codec_config'))):
        raise ValueError('network and source codec configurations differ')
    learner=checkpoint['learner']
    if not isinstance(learner.get('updates'),int) or learner['updates']<1:
        raise ValueError('trained network with positive optimizer updates required')
    weights=checkpoint['learner']['network'];settings=plan['settings']
    dims=[]
    for name in ('node','candidate','view','context_node','context_view'):
        mean,std=weights.get(name+'_mean'),weights.get(name+'_std')
        if not torch.is_tensor(mean) or not torch.is_tensor(std):
            raise ValueError('missing saved input normalization '+name)
        if (mean.ndim!=1 or mean.numel()<1 or std.shape!=mean.shape or
                not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any()):
            raise ValueError('invalid saved input normalization '+name)
        dims.append(mean.numel())
    scale=weights['output_scale']
    if scale.shape!=(2,) or not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('invalid saved output scale')
    fitted=learner.get('scales')
    if (not torch.is_tensor(fitted) or fitted.shape!=(2,) or not torch.isfinite(fitted).all()
            or (fitted<=0).any() or not torch.equal(fitted.to(scale).cpu(),scale.cpu())):
        raise ValueError('saved fitted label scales must match network output_scale')
    teacher=checkpoint.get('teacher_identity',{})
    expected=dict(codec_sha256=codec_id,source_order='original_ply',reference_target='source',
        relative_tier=0,snr=10.,noise_protocol='fixed_pair_trials',
        components=['delta_mse','lambda_s_delta_structure'],
        lambda_s=checkpoint['reference']['lambda_s'],d_ref=checkpoint['reference']['d_ref'])
    contexts=teacher.get('contexts',[])
    if (teacher.get('version')!=2 or teacher.get('background_protocol')!='four_fixed_contexts_v1'
            or not isinstance(contexts,list) or len(contexts)!=4
            or teacher.get('context_weights')!=[.25]*4):
        raise ValueError('network requires four equally weighted context teachers')
    for index,(context,name) in enumerate(zip(contexts,('q1','q2','q3','mixed'))):
        digest=context.get('background_tiers_sha256')
        if (any(context.get(key)!=value for key,value in expected.items())
                or context.get('context_id')!=index or context.get('context_name')!=name
                or (index<3 and context.get('background_tier')!=index+1)
                or type(context.get('noise_trials')) is not int or context['noise_trials']<1
                or not isinstance(digest,str) or len(digest)!=64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('network training context teacher or codec identity differs')
    shared=('source_sha256','source_fingerprint','layout_sha256','geometry_sha256',
            'camera_sha256','view_names','view_weights','noise_trials','seed','noise_seed_rule')
    if any(any(context.get(key)!=contexts[0].get(key) for key in shared) for context in contexts[1:]):
        raise ValueError('context teacher source, cameras or noise slots differ')
    network=RelationBenefitNetwork(*dims[:3],width=settings['width'],groups=settings['attention_groups'],
        context_node_dim=dims[3],context_view_dim=dims[4]).to(device)
    try:network.load_state_dict(weights,strict=True)
    except RuntimeError as error:raise ValueError('saved relation network dimensions or weights mismatch') from error
    if any(not torch.isfinite(t).all() for t in network.state_dict().values()):
        raise ValueError('finite trained network weights required')
    network.eval().requires_grad_(False)
    return network


def _reference(checkpoint, meter, q3):
    reference=copy.deepcopy(checkpoint['reference'])
    if (not math.isfinite(reference['d_ref']) or reference['d_ref']<=0 or
            not math.isfinite(reference['lambda_s']) or reference['lambda_s']<0):
        raise ValueError('valid frozen training distortion reference required')
    cost=unified_cost(meter,q3)
    reference.update(c_ref=cost['total_channel_uses'],q3_cost=cost,
                     cost_proxy=marginal_cost_proxy(cost,len(q3)))
    policy=dict(distortion='frozen_training_reference',
        input_normalizers='frozen_training_checkpoint',output_scale='frozen_training_checkpoint',
        communication='new_scene_uniform_q3_digital_side_file_v1',
        quality_calibration='none',value_units='delta_distortion_over_training_d_ref')
    return reference,policy


def predict_loaded(args,model,scene,checkpoint,sourceinfo):
    """Pure runtime boundary for an already loaded codec, scene and network state."""
    out=validate(args);device=next(model.parameters()).device
    if scene.degree!=model.cfg.sh_degree:raise ValueError('scene and source codec SH degree differ')
    if (tuple(model.cfg.rates)!=(0,8,24,48) or model.cfg.position_delivery!='quantized'
            or model.cfg.position_bits!=16):
        raise ValueError('0/8/24/48 and reliable 16bit XYZ required')
    if len(scene.raw)<1 or not torch.isfinite(scene.raw).all():raise ValueError('finite nonempty source required')
    for key in ('source','ply'):
        if key in scene.spec:_separate(out,scene.spec[key])
    torch.set_num_threads(args.cpu_threads);seed_all(args.seed)
    model.eval().requires_grad_(False);model.set_scene_geometry(scene.geometry)
    codec_id=model_id(model);network=_network(checkpoint,codec_id,device,model.cfg.to_dict())
    network_id=_weights_digest(network.state_dict())
    names=lambda cameras:[str(c.image_name) for c in cameras]
    if len(scene.cameras)<args.views or not scene.test_cameras:
        raise ValueError('sufficient preparation poses and separate test cameras required')
    preparation=[scene.cameras[i] for i in spaced_indices(len(scene.cameras),args.views)]
    tests=list(scene.test_cameras)
    preparation_pool=list(scene.cameras)+list(getattr(scene,'val_cameras',[]))
    if (len(set(names(scene.cameras)))!=len(scene.cameras) or len(set(names(tests)))!=len(tests)
            or set(names(preparation_pool))&set(names(tests))):
        raise ValueError('preparation and test camera names must be unique and disjoint')
    settings=checkpoint['plan']['settings']
    bank_args=SimpleNamespace(snr=args.snr,cpu_threads=args.cpu_threads,
        context_seed=settings['seed'],
        feature_batch=settings.get('feature_batch',8192),
        spatial_neighbors=settings.get('spatial_neighbors',16),
        projection_neighbors=settings.get('projection_neighbors',16))
    started=time.perf_counter();bank=prepare_bank(model,scene,preparation,bank_args)
    preprocessing_seconds=time.perf_counter()-started
    dims=(bank['node'].shape[-1],bank['candidate'].shape[-1],bank['view_node'].shape[-1])
    trained_dims=(network.node_mean.numel(),network.candidate_mean.numel(),network.view_mean.numel())
    if dims!=trained_dims:raise ValueError('new scene input dimensions differ from trained network')
    contexts=bank.get('contexts',[])
    if len(contexts)!=4 or [c['name'] for c in contexts]!=['q1','q2','q3','mixed']:
        raise ValueError('new scene requires the four trained context configurations')
    for context in contexts:
        if ((context['node'].shape[-1],context['view_node'].shape[-1])!=
                (network.context_node_mean.numel(),network.context_view_mean.numel())):
            raise ValueError('new scene context dimensions differ from trained network')
    meter=ExactDeploymentCost(model,scene,codec_id,args.snr,2.)
    q3=torch.full((len(scene.raw),),3,dtype=torch.long)
    reference,policy=_reference(checkpoint,meter,q3)
    started=time.perf_counter();values=network.predict_all(bank,args.inference_batch)
    q=choose_tiers(values,reference['cost_proxy']['costs'],beta=args.beta,c_ref=reference['c_ref'])
    inference_seconds=time.perf_counter()-started
    step=checkpoint.get('progress',{}).get('window',0)
    plan=dict(version='relation_benefit_predict_context_v2',network_family=FAMILY,
        source=sourceinfo,spec=scene.spec,scene_record=scene.record,codec_id=codec_id,
        network_sha256=network_id,training_step=step,settings=vars(args),
        reference_policy=policy,preparation_view_names=names(preparation),test_view_names=names(tests),
        preparation_reference='source_ply_rendering_and_train_camera_poses',scene_optimizer_updates=0,
        background_protocol='four_fixed_contexts_v1',context_seed=settings['seed'],
        deployment_aggregation='equal_mean_of_four_context_distortion_values')
    out.mkdir(parents=True)
    json_write(out/'plan.json',plan)
    json_write(out/'preprocess.json',dict(bank['metadata'],seconds=preprocessing_seconds))
    (out/'deployment').mkdir()
    np.save(out/'deployment/selected_tiers.npy',q.numpy().astype(np.uint8))
    json_write(out/'deployment/inference.json',dict(seconds=inference_seconds,
        profile=network.last_inference_profile,tier_counts=torch.bincount(q,minlength=4).tolist(),
        cost_proxy=reference['cost_proxy'],reference_policy=policy,q0_anchor=0.,
        value_units='delta_distortion_over_d_ref',scene_optimizer_updates=0,
        context_names=[c['name'] for c in contexts],context_weights=[.25]*4,
        context_passes=4,aggregation='equal_mean_distortion_then_one_rd_argmin',
        selection='minimum predicted distortion plus explicit marginal communication cost'))
    evaluation_args=argparse.Namespace(**vars(args))
    evaluation_args.seed=args.seed+800003;evaluation_args.image_views=2
    evaluation_args.allocation_image_objective=ImageDistortion('mse_ssim',reference['lambda_s'])
    rows={}
    for label,tiers in [('selected',q)]+[(f'uniform_q{k}',torch.full_like(q,k)) for k in (1,2,3)]:
        metrics=evaluate_map(model,scene,tiers,tests,evaluation_args,save_images=label=='selected',
            folder=out/'evaluation'/label,label=label)
        cost=unified_cost(meter,tiers)
        rows[label]=dict(metrics,**cost,beta=args.beta,d_ref=reference['d_ref'],c_ref=reference['c_ref'],
            rd_score=metrics['source_distortion']/reference['d_ref']+
                args.beta*cost['total_channel_uses']/reference['c_ref'])
    frozen_network=_weights_digest(network.state_dict())==network_id
    frozen_codec=model_id(model)==codec_id
    if not frozen_network or not frozen_codec:raise RuntimeError('frozen prediction state changed')
    selected,q2=rows['selected'],rows['uniform_q2']
    against=dict(psnr_difference=selected['source_psnr']-q2['source_psnr'],
        cost_saving=1-selected['total_channel_uses']/q2['total_channel_uses'],
        rd_difference=selected['rd_score']-q2['rd_score'],
        meets_q2_goal=selected['source_psnr']>=q2['source_psnr'] and
            selected['total_channel_uses']<q2['total_channel_uses'])
    result=dict(status='complete',family='relation_benefit_predict_context_v2',scene=scene.name,
        codec_id=codec_id,network_sha256=network_id,training_step=step,step=step,
        pretrained_network_updates=checkpoint.get('learner',{}).get('updates'),
        scene_optimizer_updates=0,frozen_codec_verified=frozen_codec,frozen_network_verified=frozen_network,
        tier_sha256=tier_hash(q),value_units='delta_distortion_over_d_ref',
        total_channel_uses=selected['total_channel_uses'],source_psnr=selected['source_psnr'],against_q2=against,
        evidence_scope='zero-update execution on the selected scene; measured scene RD only')
    json_write(out/'review_summary.json',dict(result=result,reference=reference,reference_policy=policy,
        rows=rows,against_q2=against,test_seed=evaluation_args.seed))
    json_write(out/'complete.json',result);_archive(out)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main(args):
    out=validate(args)
    saved,metadata,cfg,source=load_source(args.run)
    network_root=Path(args.network).resolve()
    available=[]
    for label in ('final','latest'):
        candidate=network_root/f'checkpoints/{label}/state.pt'
        if candidate.is_file():
            state=load_checkpoint(candidate)
            available.append((candidate,state))
    if not available:raise FileNotFoundError('trained relation final/latest checkpoint missing')
    # Stable max keeps final on equal steps and deploys a newer continued run.
    path,checkpoint=max(available,key=lambda item:item[1].get('progress',{}).get('window',-1))
    specs=[s for s in manifest(args.manifest) if s['name']==args.scene]
    if len(specs)!=1:raise ValueError('manifest must contain the selected scene exactly once')
    spec=dict(specs[0],prior='none')
    for key in ('ply','source'):
        if not Path(spec[key]).exists():raise FileNotFoundError(spec[key])
        _separate(out,spec[key])
    records=metadata.get('scene_records',{});old=records.get(args.scene)
    defaults=dict(metadata.get('scene',{}),**metadata.get('settings',{}))
    if old is not None:defaults.update(old)
    options=SimpleNamespace(**{k:defaults.get(k,v) for k,v in dict(blocks_per_batch=64,
        resolution=2,images='images',white_background=False,views_per_step=2,
        net_bits_per_use=2.).items()},test_views=args.test_views,validation_views=4)
    device=device_for(args.device);seed_all(args.seed)
    model=build_codec(cfg).to(device);model.load_state_dict(saved['state_dict'])
    model.eval().requires_grad_(False);model.set_training_snrs([10.])
    _network(checkpoint,model_id(model),device,model.cfg.to_dict())
    raw,degree=read_ply(spec['ply'])
    if degree!=cfg.sh_degree:raise ValueError('scene and source codec SH degree differ')
    with tempfile.TemporaryDirectory(prefix='relation-predict-scene-') as temporary:
        scene=make_scene(spec,raw,degree,model,options,temporary,old=old)
        if spec['role']=='heldout':
            from .rendering import load_cameras
            # The role remains heldout; poses are exclusively preprocessing inputs.
            scene.cameras=load_cameras(spec['source'],options.resolution,options.white_background,
                options.images,'train')
        sourceinfo=dict(source,network_checkpoint=str(path),network_checkpoint_sha256=
            hashlib.sha256(path.read_bytes()).hexdigest())
        return predict_loaded(args,model,scene,checkpoint,sourceinfo)


if __name__=='__main__':main(parser().parse_args())
