"""Read-only source metadata and scene/split verification for one-pair tests."""
import json
import hashlib
import math
from pathlib import Path
import torch
from .multiscene import Scene, manifest, parser as scene_parser
from .modular_experiment import json_write


def camera_identity(cameras):
    """Hash actual supervision/render inputs, not merely matching file names."""
    digest=hashlib.sha256()
    fields=('image_name','FoVx','FoVy','image_width','image_height','world_view_transform',
            'full_proj_transform','projection_matrix','camera_center','original_image','factor')
    for camera in cameras:
        for name in fields:
            value=getattr(camera,name,None)
            digest.update(name.encode())
            if torch.is_tensor(value):
                value=value.detach().cpu().contiguous()
                digest.update(str((value.dtype,tuple(value.shape))).encode())
                digest.update(value.numpy().tobytes())
            else:digest.update(repr(value).encode())
    return digest.hexdigest()


def inputs(args):
    run=Path(args.run).resolve()
    source=json.loads((run/'training.json').read_text(encoding='utf-8'))
    path=Path(args.manifest or source['manifest']).resolve()
    specs=[s for s in manifest(path) if s['name']=='truck' and s['role']=='train']
    if len(specs)!=1 or 'truck' not in [s['name'] for s in source['selected_scenes']]:
        raise ValueError('Truck must be a source training scene')
    spec=dict(specs[0],prior='none')
    for key in ('source','ply'):
        if not Path(spec[key]).exists():raise FileNotFoundError(spec[key])
    old=source.get('scene')
    if old is None:old=json.loads((run/'scenes/truck/scene.json').read_text(encoding='utf-8'))
    for key in ('fingerprint','source_gaussians','train_view_names','validation_view_names'):
        if not old.get(key):raise ValueError('missing source identity: '+key)
    return dict(source,**old),path,spec,old


def make_scene(spec, raw, degree, model, source, old, out):
    from .rendering import load_cameras, RenderReference
    opts=scene_parser().parse_args(['--out',str(out)])
    for key in ('blocks_per_batch','resolution','images','white_background','views_per_step','net_bits_per_use'):
        if key in source:setattr(opts,key,source[key])
    scene=Scene(spec,raw,degree,model,opts,out/'metrics',with_mask=False,cameras=False)
    if scene.fingerprint!=old['fingerprint'] or len(raw)!=old['source_gaussians']:
        raise ValueError('source PLY identity mismatch')
    cameras=load_cameras(spec['source'],opts.resolution,opts.white_background,opts.images,'train')
    names=[str(c.image_name) for c in cameras];lookup=dict(zip(names,cameras))
    tn,vn=old['train_view_names'],old['validation_view_names']
    if (len(set(names))!=len(names) or len(set(tn))!=len(tn) or len(set(vn))!=len(vn)
            or set(tn)&set(vn) or not set(tn+vn).issubset(lookup)):
        raise ValueError('camera split mismatch')
    scene.cameras=[lookup[n] for n in tn];scene.val_cameras=[lookup[n] for n in vn]
    if opts.views_per_step>len(scene.cameras):raise ValueError('insufficient training cameras')
    scene.reference=RenderReference(scene.raw,degree,opts.white_background,'source')
    scene.views_per_step=opts.views_per_step;scene.white_background=opts.white_background
    scene.net_bits_per_use=opts.net_bits_per_use
    if not math.isfinite(scene.net_bits_per_use) or scene.net_bits_per_use<=0:
        raise ValueError('invalid net bits per use assumption')
    scene.record=dict(fingerprint=scene.fingerprint,source_gaussians=len(raw),sh_degree=degree,
        geometry=scene.geometry.to_dict(),train_view_names=tn,validation_view_names=vn,
        resolution=opts.resolution,images=opts.images,white_background=opts.white_background,
        blocks_per_batch=opts.blocks_per_batch,views_per_step=opts.views_per_step,
        net_bits_per_use=scene.net_bits_per_use,
        camera_data_sha256=camera_identity(scene.cameras+scene.val_cameras))
    json_write(scene.out/'scene.json',dict(scene.record,**spec))
    return scene
