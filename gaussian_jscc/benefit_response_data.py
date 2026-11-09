"""Independent response-label data identity and training-only scale fitting."""
import hashlib
import json
from pathlib import Path

import torch

from .benefit_response_features import SCHEMA
from .benefit_response_model import combine_terms
from .benefit_stage_one_data import read_groups
from .budgeted_allocation_state import load_checkpoint
from .spatial_benefit_replay import SpatialReplay


def bootstrap(source):
    data=Path(source).resolve()/'data'
    if not (data/'complete.json').is_file(): raise ValueError('completed footprint dataset required')
    metadata=json.loads((data/'metadata.json').read_text(encoding='utf8'))
    if metadata['version']!='candidate_footprint_labels_v1':
        raise ValueError('original footprint data initializer required')
    payload=(data/'initial.pt').read_bytes()
    return dict(initial=load_checkpoint(data/'initial.pt'), config=metadata['config'],
                initial_sha256=hashlib.sha256(payload).hexdigest()),metadata


def validate_group(group):
    if group.get('supervision')=='action_gap' or not group['views']:
        raise ValueError('complete four-state response teachers required')
    for view in group['views']:
        if view.get('label_schema')!=SCHEMA or 'response_terms' not in view:
            raise ValueError('missing exact cross/energy/structure labels; recollect this dataset')
        terms=torch.as_tensor(view['response_terms']).double()
        value=torch.as_tensor(view['values']).double()
        if terms.shape!=(3,3) or value.shape!=(3,2) or not torch.isfinite(terms).all() or (terms[:,1]<0).any():
            raise ValueError('invalid response term labels')
        if not torch.allclose(combine_terms(terms),value,rtol=1e-10,atol=1e-12):
            raise ValueError('response labels disagree with final distortion')
        if 'footprints' not in view['inputs']: raise ValueError('missing candidate-region input')


def compact(group):
    return dict(source=group['source'],step=group['step'],rates=group['rates'],
        validation=group.get('validation',False),importance_weight=group.get('importance_weight',1.),
        views=[dict(values=v['values'],response_terms=v['response_terms']) for v in group['views']])


def load_data(data):
    data=Path(data).resolve()
    if not (data/'complete.json').is_file(): raise ValueError('complete response label collection first')
    metadata=json.loads((data/'metadata.json').read_text(encoding='utf8'))
    if metadata['version']!=SCHEMA: raise ValueError('response label schema mismatch; recollect labels')
    manifest=json.loads((data/'data_manifest.json').read_text(encoding='utf8'))
    root=Path(manifest['replay_root']).resolve()
    if not root.is_dir(): raise FileNotFoundError(f'replay missing: {root}')
    training,validation=manifest['training'],manifest['validation']
    ids=[e['id'] for e in training+validation]
    if not training or not validation or len(ids)!=len(set(ids)):
        raise ValueError('nonempty disjoint unique training/validation records required')
    replay=SpatialReplay(root,max_groups=1,validation_groups=len(validation),cache_mib=256)
    groups={};sources={}
    for role,entries in [('validation',validation),('training',training)]:
        groups[role]=[];sources[role]=set()
        for i,g in enumerate(read_groups(replay,entries),1):
            validate_group(g)
            if bool(g.get('validation'))!=(role=='validation'):raise ValueError('record role mismatch')
            groups[role].append(compact(g));sources[role].add(g['source'])
            if i==len(entries) or i%512==0:print(f'Validate response labels {role}: {i}/{len(entries)}',flush=True)
    if sources['training'] & sources['validation']:raise ValueError('training/validation source overlap')
    hashes={n:hashlib.sha256((data/n).read_bytes()).hexdigest()
            for n in ['metadata.json','data_manifest.json','initial.pt']}
    return replay,training,validation,groups,metadata,load_checkpoint(data/'initial.pt'),hashes


def fit_term_scales(groups):
    total=torch.zeros(3,dtype=torch.double);count=0.
    for group in groups:
        if group.get('validation'):raise ValueError('validation cannot fit response scales')
        weight=float(group.get('importance_weight',1.))
        if not 0<weight<float('inf'):raise ValueError('invalid importance weight')
        terms=torch.stack([v['response_terms'].double() for v in group['views']])
        total+=terms.square().mean((0,1))*weight;count+=weight
    if not count:raise ValueError('nonempty training groups required')
    scales=(total/count).sqrt()
    return scales.clamp_min(max(1e-8,float(scales.max())*1e-3))
