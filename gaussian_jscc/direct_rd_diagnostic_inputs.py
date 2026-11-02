"""Load only named training images; preserve unused split identity as metadata."""
from pathlib import Path
from types import SimpleNamespace

from .modular_multiscene_state import camera_fingerprint
from .multiscene import Scene


def selected_train_cameras(source,names,resolution,images):
    from scene.colmap_loader import (read_extrinsics_binary,read_intrinsics_binary,
                                    read_extrinsics_text,read_intrinsics_text)
    from scene.dataset_readers import readColmapCameras
    from utils.camera_utils import cameraList_from_camInfos
    root=Path(source);sparse=root/'sparse/0'
    if not sparse.is_dir():raise ValueError('named train-only diagnostic loader requires COLMAP source')
    if (sparse/'images.bin').is_file() and (sparse/'cameras.bin').is_file():
        extr=read_extrinsics_binary(str(sparse/'images.bin'));intr=read_intrinsics_binary(str(sparse/'cameras.bin'))
    else:
        extr=read_extrinsics_text(str(sparse/'images.txt'));intr=read_intrinsics_text(str(sparse/'cameras.txt'))
    ordered=sorted(extr,key=lambda key:Path(extr[key].name).name.split('.')[0])
    allowed={Path(extr[key].name).name.split('.')[0]:key for i,key in enumerate(ordered) if i%8!=0}
    if len(names)!=len(set(names)) or not set(names)<=set(allowed):
        raise ValueError('requested cameras must be unique COLMAP train names')
    subset={allowed[name]:extr[allowed[name]] for name in names}
    infos=readColmapCameras(subset,intr,str(root/(images or 'images')))
    lookup={c.image_name:c for c in infos}
    args=SimpleNamespace(resolution=resolution,data_device='cpu')
    return cameraList_from_camInfos([lookup[name] for name in names],1.,args)


def diagnostic_scene(spec,raw,degree,model,options,out,*,old,expected):
    from .rendering import RenderReference
    if degree!=model.cfg.sh_degree:raise ValueError('SH degree differs from source')
    scene=Scene(spec,raw,degree,model,options,Path(out),with_mask=False,cameras=False)
    scene.white_background=options.white_background;scene.views_per_step=options.views_per_step
    scene.net_bits_per_use=options.net_bits_per_use;scene.channel_trained=True
    scene.cameras=selected_train_cameras(spec['source'],old['train_view_names'],options.resolution,options.images)
    scene.val_cameras=[];scene.test_cameras=[]
    record={k:v for k,v in expected.items() if k not in ('allocation_train_view_names','calibration_view_names')}
    record=dict(record,fingerprint=scene.fingerprint,source_gaussians=len(raw),sh_degree=degree,
        geometry=scene.geometry.to_dict(),train_view_names=[str(c.image_name) for c in scene.cameras],
        resolution=options.resolution,images=options.images,white_background=options.white_background,
        blocks_per_batch=options.blocks_per_batch,views_per_step=options.views_per_step)
    fingerprints=dict(record['camera_fingerprints']);fingerprints['train']=camera_fingerprint(scene.cameras)
    record['camera_fingerprints']=fingerprints;scene.record=record
    scene.reference=RenderReference(scene.raw,degree,options.white_background,'source')
    return scene
