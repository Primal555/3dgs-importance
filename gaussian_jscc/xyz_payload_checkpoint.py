"""Separate checkpoint identity and exact Adam/RNG continuation."""
from pathlib import Path
import torch
from .data import Geometry
from .xyz_payload_config import XYZPayloadConfig
from .xyz_payload_codec import XYZPayloadCodec
from .xyz_payload_training import configure_phase
from .modular_checkpoint import capture_rng


def save_checkpoint(path,model,optimizer,progress,scene_record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    saved={'xyz_payload_version':1,'config':model.cfg.to_dict(),'state_dict':model.state_dict(),
           'progress':progress,'optimizer':optimizer.state_dict() if optimizer is not None else None,
           'rng':capture_rng(),'scene_record':scene_record}
    temporary=path.with_name(path.name+'.tmp')
    try:
        torch.save(saved,temporary); temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


def read_checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('xyz_payload_version')!=1 or saved.get('config',{}).get('architecture')!='xyz_payload_jscc_v1':
        raise ValueError('expected independent XYZ payload checkpoint, not legacy/reliable XYZ')
    return saved


def _model(saved,device):
    model=XYZPayloadCodec(XYZPayloadConfig.from_dict(saved['config'])).to(device)
    model.load_state_dict(saved['state_dict'],strict=True)
    model.set_scene_geometry(Geometry(**saved['scene_record']['geometry']))
    model.set_training_snrs(saved['progress'].get('train_snrs',(10.,)))
    return model


def load_checkpoint(path,device='cpu'):
    return _model(read_checkpoint(path),device).eval()


def restore_checkpoint(path,device):
    saved=read_checkpoint(path); progress=saved['progress']; model=_model(saved,device)
    optimizer=None
    if saved['optimizer'] is not None:
        optimizer=configure_phase(model,progress['phase'],progress['module_lrs'],
                                  joint_semantic_encoder=progress.get('joint_semantic_encoder',False))
        optimizer.load_state_dict(saved['optimizer'])
    return model,optimizer,progress,saved['rng'],saved['scene_record']
