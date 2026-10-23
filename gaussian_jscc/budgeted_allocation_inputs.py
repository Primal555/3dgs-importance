"""Read-only source validation for a single explicitly selected trained scene."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from .allocation import scene_fingerprint
from .data import read_ply
from .model_factory import config_from_dict
from .multiscene import manifest
from .successive_transfer import load_source


def load_inputs(args):
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out): raise ValueError('source/output must be separate')
    metadata = json.loads((run/'training.json').read_text(encoding='utf-8'))
    if metadata.get('version') in ('successive_multiscene_joint_v1', 'successive_transfer_v1'):
        saved, metadata, cfg, identity = load_source(run)
        records = metadata['scene_records']; defaults = metadata['settings']
    else:
        from .successive_allocation import parser, source_plan
        legacy = parser().parse_args(['--run', str(run), '--out', str(out)])
        identity = source_plan(legacy)
        saved = torch.load(run/'checkpoints/final/codec.pt', map_location='cpu', weights_only=True)
        cfg = config_from_dict(saved['config'])
        records = {s['name']: metadata['scene'] for s in metadata['selected_scenes']}
        defaults = dict(metadata['scene'], net_bits_per_use=metadata.get('net_bits_per_use'))
    if cfg.position_delivery != 'quantized' or cfg.position_bits != 16:
        raise ValueError('reliable 16bit XYZ required')
    selected = {s['name']: s for s in metadata['selected_scenes']}
    if args.scene not in selected or selected[args.scene]['role'] != 'train' or args.scene not in records:
        raise ValueError('explicit scene must be a trained, train-role source scene; heldouts cannot be fitted')
    spec = dict(selected[args.scene], prior='none')
    if args.manifest:
        matches = [s for s in manifest(args.manifest) if s['name'] == args.scene and s['role'] == 'train']
        if len(matches) != 1: raise ValueError('relocation manifest requires one matching train-role scene')
        spec = dict(matches[0], prior='none')
    for key in ('ply', 'source'):
        if not Path(spec[key]).exists(): raise FileNotFoundError(spec[key])
    raw, degree = read_ply(spec['ply']); old = records[args.scene]
    if (degree != cfg.sh_degree or len(raw) != old['source_gaussians']
            or scene_fingerprint(raw) != old['fingerprint'] or not torch.isfinite(raw).all()):
        raise ValueError('selected scene PLY identity/degree/values differ from source record')
    del raw
    bits = defaults.get('net_bits_per_use')
    if bits is None or not math.isfinite(bits) or bits <= 0:
        raise ValueError('source must explicitly record positive reliable net bits/use')
    options = SimpleNamespace(**{k: old[k] for k in
        ('blocks_per_batch', 'resolution', 'images', 'white_background', 'views_per_step')},
        net_bits_per_use=bits, test_views=args.test_views, validation_views=args.calibration_views)
    if len(old['train_view_names']) < args.calibration_views + args.window_views:
        raise ValueError('need separate calibration views plus a distinct training window')
    return saved, cfg, spec, old, options, dict(identity, scene=args.scene, net_bits_per_use=bits)
