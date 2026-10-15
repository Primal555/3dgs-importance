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
    optimizer=configure_phase(model,progress['phase'],progress.get('module_lrs',{}),
                              freeze_semantic=progress.get('freeze_semantic',False),
                              denoiser_only=progress.get('channel_denoiser_only',False),
                              neighbor_only=progress.get('channel_neighbor_only',False))
    if state['optimizer'] is not None: optimizer.load_state_dict(state['optimizer'])
    return model,optimizer,progress,state['rng'],state['scene_record']


def initialize_channel_run(path, device, *, cfg=None):
    saved=_read(path); progress=saved['training']['progress']
    if progress['phase']!='semantic_render' or not progress.get('phase_complete') or not progress.get('completed',{}).get('semantic_render',0):
        raise ValueError('semantic initialization requires completed nonempty semantic_render')
    original=config_from_dict(saved['config'])
    if cfg is not None:
        a=original.to_dict(); b=cfg.to_dict()
        # These affect only freshly initialized channel modules, not the teacher y.
        # Exact resume still requires the entire config, including rates, to match.
        for key in ('channel_variant','channel_hidden','channel_prefix_mode','channel_denoiser_mode',
                    'channel_denoiser_hidden','channel_denoiser_neighbors','channel_denoiser_scope',
                    'channel_neighbor_mode','channel_neighbor_hidden','channel_neighbor_neighbors','rates'):
            a.pop(key,None); b.pop(key,None)
        if a!=b: raise ValueError('semantic initialization config mismatch')
    model=build_codec(cfg or original).to(device)
    values={n:v for n,v in saved['state_dict'].items() if n.startswith(('semantic_encoder.','semantic_decoder.')) or n in ('attr_mean','attr_std')}
    model.load_state_dict(values,strict=False)
    return model


def initialize_codec_run(path, device):
    """Keep every trained tensor/config, but not the source Adam or schedule.

    Unlike exact resume, the caller selects a new optimization scope. In
    particular, neighbor-only/frozen-semantic source flags must not leak into
    the new channel-then-joint experiment.
    """
    saved=_read(path); state=saved['training']; progress=state['progress']
    if not progress.get('channel_trained'):
        raise ValueError('full codec initialization requires a trained channel')
    model=build_codec(config_from_dict(saved['config'])).to(device)
    model.load_state_dict(saved['state_dict'])
    return model,progress,state['scene_record']


def validate_scene(saved, current):
    if saved!=current:
        raise ValueError('scene fingerprint, settings or camera split mismatch; refusing silent resume/init')
