"""Atomic exact training resume, separate from semantic-only initialization."""
import os
import random
from pathlib import Path
import numpy as np
import torch
from .model_factory import config_from_dict,build_codec
from .modular_training import configure_phase


def capture_rng():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':(n[0],n[1].tolist(),n[2],n[3],n[4]),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(rng):
    random.setstate(rng['python'])
    n=rng['numpy']; np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda']:
        if not torch.cuda.is_available() or len(rng['cuda'])!=torch.cuda.device_count():
            raise ValueError('exact resume requires the same visible CUDA device count')
        torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])


def save_training_checkpoint(path, model, optimizer, progress, rng, scene_record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    saved={'version':4,'config':model.cfg.to_dict(),'state_dict':model.state_dict(),
           'step':progress['step'],'training':{'modular_version':1,'progress':progress,
           'optimizer':optimizer.state_dict() if optimizer is not None else None,'rng':rng,'scene_record':scene_record}}
    temporary=path.with_name(path.name+'.tmp')
    try:
        torch.save(saved,temporary)
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()


def _read(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('version')!=4 or saved.get('config',{}).get('architecture')!='modular_jscc_v1' or saved.get('training',{}).get('modular_version')!=1:
        raise ValueError('expected modular experiment training checkpoint')
    return saved


def restore_training_checkpoint(path, device):
    saved=_read(path); state=saved['training']; progress=state['progress']
    model=build_codec(config_from_dict(saved['config'])).to(device)
    model.load_state_dict(saved['state_dict'])
    model.set_training_snrs(progress.get('train_snrs',()))
    optimizer=configure_phase(model,progress['phase'],progress.get('module_lrs',{}))
    if state['optimizer'] is not None: optimizer.load_state_dict(state['optimizer'])
    return model,optimizer,progress,state['rng'],state['scene_record']


def initialize_channel_run(path, device):
    saved=_read(path); progress=saved['training']['progress']
    if progress['phase']!='semantic_render' or not progress.get('phase_complete') or not progress.get('completed',{}).get('semantic_render',0):
        raise ValueError('semantic initialization requires completed nonempty semantic_render')
    model=build_codec(config_from_dict(saved['config'])).to(device)
    values={n:v for n,v in saved['state_dict'].items() if n.startswith(('semantic_encoder.','semantic_decoder.')) or n in ('attr_mean','attr_std')}
    model.load_state_dict(values,strict=False)
    return model


def validate_scene(saved, current):
    if saved!=current:
        raise ValueError('scene fingerprint, settings or camera split mismatch; refusing silent resume/init')
