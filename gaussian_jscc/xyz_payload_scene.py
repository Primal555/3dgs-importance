"""Verified Truck inputs without allocation or coordinate-compression meters."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn.utils.rnn import pad_sequence
from .allocation import scene_fingerprint
from .data import Geometry,morton_order
from .multiscene import manifest


def inputs(args):
    run=Path(args.run).resolve()
    source=json.loads((run/'training.json').read_text(encoding='utf-8'))
    path=Path(args.manifest or source['manifest']).resolve()
    selected=[s for s in manifest(path) if s['name']=='truck' and s['role']=='train']
    if len(selected)!=1 or 'truck' not in [s['name'] for s in source['selected_scenes']]:
        raise ValueError('source must contain trained Truck scene metadata')
    spec=dict(selected[0],prior='none')
    for key in ('ply','source'):
        if not Path(spec[key]).exists(): raise FileNotFoundError(spec[key])
    if source.get('codec_config',{}).get('architecture') in ('modular_jscc_v1','xyz_payload_jscc_v1'):
        old=source['scene']; source={**source,**old}
    else: old=json.loads((run/'scenes/truck/scene.json').read_text(encoding='utf-8'))
    for key in ('fingerprint','source_gaussians','train_view_names','validation_view_names'):
        if not old.get(key): raise ValueError('source metadata lacks '+key)
    return source,path,spec,old


def make_scene(spec,raw,degree,model,source,old,args):
    from .rendering import load_cameras,RenderReference
    fingerprint=scene_fingerprint(raw)
    if fingerprint!=old['fingerprint'] or len(raw)!=old['source_gaussians']:
        raise ValueError('source PLY differs from recorded Truck')
    geometry=Geometry.fit(raw[:,:3],model.cfg.morton_bits)
    order=torch.from_numpy(morton_order(geometry.quantize(raw[:,:3]).numpy()).astype('int64'))
    raw=raw[order]
    features=torch.cat((geometry.normalize(raw[:,:3]),
        (raw[:,3:]-model.attr_mean.detach().cpu())/model.attr_std.detach().cpu()),-1)
    blocks=list(features.split(model.cfg.block_size)); ids=list(order.split(model.cfg.block_size))
    batch=args.blocks_per_batch or source.get('blocks_per_batch',64)
    views=args.views_per_step or source.get('views_per_step',2)
    if batch<1 or views<1: raise ValueError('positive batch/views required')
    groups=[pad_sequence(blocks[i:i+batch],batch_first=True) for i in range(0,len(blocks),batch)]
    group_ids=[pad_sequence(ids[i:i+batch],batch_first=True,padding_value=-1) for i in range(0,len(ids),batch)]
    resolution=source.get('resolution',2); images=source.get('images','images'); white=source.get('white_background',False)
    cameras=load_cameras(spec['source'],resolution,white,images,'train')
    names=[str(c.image_name) for c in cameras]; lookup=dict(zip(names,cameras))
    tn=old['train_view_names']; vn=old['validation_view_names']
    if len(set(names))!=len(names) or set(tn)&set(vn) or not set(tn+vn).issubset(lookup):
        raise ValueError('source camera split changed')
    train=[lookup[n] for n in tn]; validation=[lookup[n] for n in vn]
    if views>len(train): raise ValueError('not enough training cameras')
    record=dict(fingerprint=fingerprint,source_gaussians=len(raw),geometry=geometry.to_dict(),
                train_view_names=tn,validation_view_names=vn,resolution=resolution,images=images,
                white_background=white,blocks_per_batch=batch,views_per_step=views)
    model.set_scene_geometry(geometry)
    return SimpleNamespace(raw=raw,geometry=geometry,order=order,groups=groups,group_ids=group_ids,
        cameras=train,val_cameras=validation,reference=RenderReference(raw,degree,white,'source'),
        degree=degree,white_background=white,views_per_step=views,record=record,channel_trained=False,
        net_bits_per_use=source.get('net_bits_per_use',2.))
