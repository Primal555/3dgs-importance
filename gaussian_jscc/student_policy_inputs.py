"""Completed teacher identities, source loading and camera boundaries."""
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .budgeted_allocation_evaluation import tier_hash
from .render_objective import spaced_indices


def json_read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def load_teacher(root, scene, codec_id, beta, cfg, *, seed, snr, trials):
    root = Path(root).resolve()
    complete = json_read(root/'complete.json'); selection = json_read(root/'deployment/selection.json')
    meta = json_read(root/'training.json'); plan = json_read(root/'plan.json')
    if (complete.get('status') != 'complete' or complete.get('codec_id') != codec_id or
            selection.get('codec_id') != codec_id or not complete.get('frozen_codec_verified') or
            meta.get('fingerprint') != scene.fingerprint or meta.get('source_gaussians') != len(scene.raw) or
            plan['codec_config'] != json.loads(json.dumps(cfg))):
        raise ValueError('completed frozen teacher codec/PLY/config identity mismatch')
    q_array = np.load(root/'deployment/selected_tiers.npy', allow_pickle=False)
    if q_array.dtype.kind not in 'iu' or q_array.shape != (len(scene.raw),) or ((q_array < 0) | (q_array > 3)).any():
        raise ValueError('teacher deployment must contain original-row four-tier integers')
    q = torch.from_numpy(q_array.astype(np.int64))
    if tier_hash(q) != complete['tier_sha256'] or tier_hash(q) != selection['tier_sha256']:
        raise ValueError('teacher selected map hash mismatch')
    reference = meta['objective_reference']
    for key in ('d_ref', 'c_ref', 'lambda_s', 'beta'):
        if reference.get(key) != complete.get(key) or reference.get(key) != selection.get(key):
            raise ValueError('teacher RD reference mismatch')
    if (reference['beta'] != beta or reference['d_ref'] <= 0 or reference['c_ref'] <= 0 or
            not all(math.isfinite(reference[k]) for k in ('d_ref', 'c_ref', 'lambda_s', 'beta'))):
        raise ValueError('teacher beta or normalized RD reference mismatch')
    for key, value in (('seed', seed), ('snr', snr), ('trials', trials)):
        if plan['settings'][key] != value: raise ValueError('teacher paired evaluation settings mismatch: '+key)
    if (meta['test_view_names'] != scene.record['test_view_names'] or
            meta.get('camera_fingerprints') != scene.record.get('camera_fingerprints')):
        raise ValueError('teacher source camera identity mismatch')
    paths = ('complete.json', 'deployment/selection.json', 'training.json', 'plan.json', 'deployment/selected_tiers.npy')
    identity = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}
    return q, copy.deepcopy(reference), dict(path=str(root), hashes=identity,
        train_names=meta['allocation_train_view_names'], calibration_names=meta['calibration_view_names'],
        test_names=meta['test_view_names'], deployment_rule='teacher final ten-draw action labels')


def split_student_cameras(scene, teacher, preparation_views):
    lookup = {str(c.image_name): c for c in scene.cameras}
    train, calibration, test = teacher['train_names'], teacher['calibration_names'], teacher['test_names']
    if (not train or not calibration or len(set(train+calibration)) != len(train+calibration) or
            set(train+calibration) != set(lookup) or set(train+calibration) & set(test)):
        raise ValueError('student training/calibration/test camera boundary mismatch')
    training, held = [lookup[n] for n in train], [lookup[n] for n in calibration]
    preparation = [training[i] for i in spaced_indices(len(training), min(preparation_views, len(training))) ]
    return training, held, preparation


def load_student_source(args):
    """Training inherits exact source identities; predict also accepts a new manifest scene."""
    from .budgeted_allocation_inputs import load_inputs
    if args.phase == 'train' or not args.manifest:
        return load_inputs(args)
    from .multiscene import manifest
    from .successive_transfer import load_source
    from .data import read_ply
    saved, metadata, cfg, source = load_source(Path(args.run).resolve())
    selected = [s for s in manifest(args.manifest) if s['name'] == args.scene]
    if len(selected) != 1: raise ValueError('one explicit scene required for student prediction')
    spec = dict(selected[0], prior='none')
    # Camera preparation at the sender is allowed; no optimizer uses the scene.
    spec['evaluation_role'] = spec['role']; spec['role'] = 'train'
    sample = next(iter(metadata['scene_records'].values()))
    options = SimpleNamespace(**{k: sample[k] for k in ('blocks_per_batch', 'resolution', 'images',
        'white_background', 'views_per_step')}, net_bits_per_use=metadata['settings']['net_bits_per_use'],
        test_views=args.test_views, validation_views=args.calibration_views)
    raw, degree = read_ply(spec['ply'])
    if degree != cfg.sh_degree or not torch.isfinite(raw).all(): raise ValueError('new scene SH/values mismatch')
    if not len(raw): raise ValueError('empty prediction scene')
    return saved, cfg, spec, None, options, dict(source, scene=args.scene, net_bits_per_use=options.net_bits_per_use)
