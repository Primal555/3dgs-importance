"""Validate a completed Joint B and keep training/heldout scene inputs separate."""
import hashlib
import json
from pathlib import Path

import torch

from .model_factory import config_from_dict
from .modular_checkpoint import validate_scene
from .modular_multiscene_state import camera_fingerprint
from .multiscene import Scene,manifest
from .render_objective import spaced_indices,split_cameras
from .successive_joint import encoder_digest


def inputs(args):
    run=Path(args.run).resolve();out=Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):raise ValueError('source and output must be separate')
    path=run/'checkpoints/final/codec.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
    metadata=json.loads((run/'training.json').read_text(encoding='utf-8'))
    complete=json.loads((run/'complete.json').read_text(encoding='utf-8'))
    progress=saved.get('training',{}).get('progress',{});step=saved.get('step',0)
    plan=metadata.get('successive_joint',{})
    if (saved.get('version')!=4 or saved.get('config',{}).get('channel_variant')!='successive'
            or tuple(saved['config'].get('rates',()))!=(0,8,24,48)
            or plan.get('version')!='successive_joint_encoder_v1'
            or not progress.get('train_semantic_encoder') or not progress.get('successive_joint')
            or not progress.get('phase_complete') or progress.get('phase')!='channel_render'
            or not progress.get('channel_trained') or step<1 or progress.get('step')!=step
            or progress.get('completed',{}).get('channel_render')!=step
            or progress.get('phase_budgets',{}).get('channel_render')!=step
            or plan.get('settings',{}).get('steps')!=step
            or complete.get('status')!='complete' or complete.get('step')!=step
            or complete.get('joint_updates')!=step or complete.get('semantic_encoder_updated') is not True
            or complete.get('frozen_attribute_statistics_unchanged') is not True
            or not bool(saved['state_dict'].get('channel_decoder.neighbor_enabled',False))):
        raise ValueError('source must be a completed successive Joint B with all four modules trained')
    if json.loads(json.dumps(saved['config']))!=metadata.get('codec_config'):raise ValueError('source config mismatch')
    cfg=config_from_dict(saved['config']);validate_scene(saved['training']['scene_record'],metadata['scene'])
    if {s['name'] for s in metadata.get('selected_scenes',[])}!={'truck'}:
        raise ValueError('initializer must be Truck-only Joint B; heldout training history cannot be reused')
    stats=encoder_digest(saved['state_dict'],include_encoder=False)
    if stats!=plan.get('statistics_sha256'):raise ValueError('source attribute statistics changed')
    if not args.scenes or len(set(args.scenes))!=len(args.scenes) or 'truck' not in args.scenes:
        raise ValueError('unique training scenes including truck required')
    specs=manifest(args.manifest);lookup={s['name']:s for s in specs};selected=[]
    for name in args.scenes:
        if name not in lookup or lookup[name]['role']!='train':raise ValueError(name+': train-role scene required')
        selected.append(dict(lookup[name],prior='none'))
    heldout_lookup={s['name']:s for s in (manifest(args.heldout_manifest) if args.heldout_manifest else specs)}
    if args.heldout_scene not in heldout_lookup:
        raise ValueError('heldout scene missing; supply --heldout-manifest with a heldout-role scene')
    heldout=dict(heldout_lookup[args.heldout_scene],prior='none')
    if heldout['role']!='heldout' or heldout['name'] in args.scenes:raise ValueError('heldout scene overlaps training roles')
    if any(Path(heldout[k]).resolve()==Path(s[k]).resolve() for s in selected for k in ('ply','source')):
        raise ValueError('heldout/training path overlap or alias')
    for spec in selected+[heldout]:
        for k in ('ply','source'):
            if not Path(spec[k]).exists():raise FileNotFoundError(f'{spec["name"]}: {spec[k]}')
    # Continue the existing numerical representation without silently refitting
    # attribute moments. Truck's original camera split is retained below.
    for key in ('blocks_per_batch','resolution','images','white_background','views_per_step'):
        if getattr(args,key) is None:setattr(args,key,metadata['scene'][key])
    if args.net_bits_per_use is None:args.net_bits_per_use=metadata.get('net_bits_per_use',2.)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return saved,metadata,cfg,selected,heldout,dict(source_checkpoint=str(path),source_sha256=digest,
        source_step=step,statistics_sha256=stats)


def make_scene(spec,raw,degree,model,args,out,*,old=None):
    from .rendering import load_cameras,RenderReference
    if degree!=model.cfg.sh_degree:raise ValueError(spec['name']+': SH degree differs from source codec')
    s=Scene(spec,raw,degree,model,args,Path(out),with_mask=False,cameras=False)
    s.white_background=args.white_background;s.views_per_step=args.views_per_step
    s.net_bits_per_use=args.net_bits_per_use;s.channel_trained=True
    if spec['role']=='train':
        cameras=load_cameras(spec['source'],args.resolution,args.white_background,args.images,'train')
        names=[str(c.image_name) for c in cameras]
        if len(names)!=len(set(names)):raise ValueError('duplicate training/validation cameras')
        if old is not None:
            lookup=dict(zip(names,cameras));tn=old['train_view_names'];vn=old['validation_view_names']
            if set(tn)&set(vn) or not set(tn+vn)<=set(lookup):raise ValueError('source Truck camera split changed')
            s.cameras=[lookup[n] for n in tn];s.val_cameras=[lookup[n] for n in vn]
        else:s.cameras,s.val_cameras,_,_=split_cameras(cameras,args.validation_views)
        if len(s.cameras)<args.views_per_step:raise ValueError('insufficient training cameras')
    else:
        # A heldout scene never loads training cameras or supplies a task.
        s.cameras=[];s.val_cameras=[]
    cameras=load_cameras(spec['source'],args.resolution,args.white_background,args.images,'test')
    names=[str(c.image_name) for c in cameras]
    if len(names)!=len(set(names)) or set(names)&{str(c.image_name) for c in s.cameras+s.val_cameras}:
        raise ValueError('test cameras overlap or contain duplicate names')
    s.test_cameras=[cameras[i] for i in spaced_indices(len(cameras),args.test_views or len(cameras))]
    s.reference=RenderReference(s.raw,degree,args.white_background,'source')
    s.record=dict(fingerprint=s.fingerprint,source_gaussians=len(raw),sh_degree=degree,
        geometry=s.geometry.to_dict(),train_view_names=[str(c.image_name) for c in s.cameras],
        validation_view_names=[str(c.image_name) for c in s.val_cameras],
        test_view_names=[str(c.image_name) for c in s.test_cameras],
        camera_fingerprints={k:camera_fingerprint(v) for k,v in
            [('train',s.cameras),('validation',s.val_cameras),('test',s.test_cameras)]},
        resolution=args.resolution,images=args.images,white_background=args.white_background,
        blocks_per_batch=args.blocks_per_batch,views_per_step=args.views_per_step)
    if old is not None:validate_scene(old,{k:s.record[k] for k in old})
    return s
