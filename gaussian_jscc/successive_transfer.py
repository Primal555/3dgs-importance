"""Continue a completed shared successive codec on additional, isolated scenes."""
import hashlib
import json
from pathlib import Path

import torch

from .model_factory import config_from_dict
from .modular_checkpoint import validate_scene
from .multiscene import manifest
from .successive_joint import encoder_digest
from . import successive_multiscene as engine

FAMILY = 'successive_transfer_v1'
NEW_TRAIN = ('bicycle', 'garden', 'stump', 'bonsai', 'counter', 'kitchen')
NEW_HELDOUT = ('flowers', 'treehill', 'room')
OLD_TRAIN = ('train', 'truck', 'playroom')


def parser():
    p = engine.parser()
    p.description = __doc__
    p.set_defaults(scenes=list(OLD_TRAIN + NEW_TRAIN), steps_per_scene=1250, heldout_scene=None)
    p.add_argument('--heldout-scenes', nargs='+', default=list(NEW_HELDOUT))
    return p


def load_source(run):
    """Validate the shared endpoint, never reinterpret an old Truck initializer."""
    run = Path(run).resolve()
    path = run / 'checkpoints/final/codec.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = json.loads((run / 'training.json').read_text(encoding='utf-8'))
    complete = json.loads((run / 'complete.json').read_text(encoding='utf-8'))
    progress = saved.get('training', {}).get('progress', {})
    step = saved.get('step', 0)
    version = metadata.get('version')
    counts = progress.get('scene_completed', {})
    if (saved.get('version') != 4 or version not in (engine.FAMILY, FAMILY)
            or progress.get('training_family') != version
            or saved.get('config', {}).get('channel_variant') != 'successive'
            or tuple(saved.get('config', {}).get('rates', ())) != (0, 8, 24, 48)
            or not progress.get('phase_complete') or not progress.get('train_semantic_encoder')
            or not progress.get('channel_trained') or not progress.get('final_done')
            or progress.get('phase') != 'channel_render' or progress.get('step') != step
            or step < 1 or sum(counts.values()) != step
            or progress.get('completed', {}).get('channel_render') != step
            or progress.get('phase_budgets', {}).get('channel_render') != step
            or complete.get('status') != 'complete' or complete.get('step') != step
            or complete.get('scene_completed') != counts
            or complete.get('heldout_optimizer_updates') != 0
            or complete.get('attribute_statistics_frozen') is not True
            or not bool(saved['state_dict'].get('channel_decoder.neighbor_enabled', False))):
        raise ValueError('source must be a completed shared successive multi-scene endpoint')
    if json.loads(json.dumps(saved['config'])) != metadata.get('codec_config'):
        raise ValueError('source config/metadata mismatch')
    cfg = config_from_dict(saved['config'])
    if cfg.position_delivery != 'quantized' or cfg.position_bits != 16:
        raise ValueError('reliable 16bit XYZ required')
    records = metadata['scene_records']
    validate_scene(saved['training']['scene_record'], records)
    selected = metadata['selected_scenes']
    if {s['name'] for s in selected} != set(counts) or set(records) != set(counts):
        raise ValueError('source scene history/counts/records mismatch')
    if any(value != metadata['settings']['steps_per_scene'] for value in counts.values()):
        raise ValueError('source completed counts do not match balanced per-scene budget')
    stats = encoder_digest(saved['state_dict'], include_encoder=False)
    if stats != metadata.get('statistics_sha256'):
        raise ValueError('source attribute statistics changed')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    history = sorted(set(metadata.get('training_scene_history', ())) | set(counts))
    return saved, metadata, cfg, dict(source_checkpoint=str(path), source_sha256=digest,
        source_step=step, statistics_sha256=stats, training_scene_history=history)


def inputs(args):
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('source and output must be separate')
    saved, metadata, cfg, source = load_source(run)
    if args.heldout_scene is not None or args.heldout_manifest:
        raise ValueError('use --heldout-scenes and one combined role-checked manifest')
    if not args.scenes or len(set(args.scenes)) != len(args.scenes):
        raise ValueError('training scene names must be unique and nonempty')
    if not args.heldout_scenes or len(set(args.heldout_scenes)) != len(args.heldout_scenes):
        raise ValueError('heldout scene names must be unique and nonempty')
    if set(args.heldout_scenes) & (set(args.scenes) | set(source['training_scene_history'])):
        raise ValueError('heldout overlaps current training or source training history')
    lookup = {s['name']: s for s in manifest(args.manifest)}
    specs, heldouts = [], []
    for names, role, target in ((args.scenes, 'train', specs),
                                 (args.heldout_scenes, 'heldout', heldouts)):
        for name in names:
            if name not in lookup or lookup[name]['role'] != role:
                raise ValueError(name + ': expected ' + role + '-role scene')
            target.append(dict(lookup[name], prior='none'))
    train_sources = {Path(s['source']).resolve() for s in specs + metadata['selected_scenes']}
    train_plys = {Path(s['ply']).resolve() for s in specs + metadata['selected_scenes']}
    heldout_sources = [Path(s['source']).resolve() for s in heldouts]
    if len(set(heldout_sources)) != len(heldout_sources):
        raise ValueError('heldout source aliases')
    for spec in heldouts:
        if Path(spec['source']).resolve() in train_sources or Path(spec['ply']).resolve() in train_plys:
            raise ValueError('heldout/training path alias')
    previous = {s['name']: s for s in metadata['selected_scenes']}
    for spec in specs + heldouts:
        for key in ('ply', 'source'):
            if not Path(spec[key]).exists():
                raise FileNotFoundError(f'{spec["name"]}: {spec[key]}')
        if spec['name'] in previous and any(Path(spec[k]).resolve() != Path(previous[spec['name']][k]).resolve()
                                           for k in ('ply', 'source')):
            raise ValueError(spec['name'] + ': old scene inputs changed; preserve pruned PLY and camera split')
    defaults = metadata['settings']
    for key in ('blocks_per_batch', 'resolution', 'images', 'white_background', 'views_per_step'):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    if args.net_bits_per_use is None:
        args.net_bits_per_use = defaults['net_bits_per_use']
    source['training_scene_history'] = sorted(set(source['training_scene_history']) | set(args.scenes))
    return saved, metadata, cfg, specs, heldouts, source


def main(args, max_updates=None):
    return engine.main(args, max_updates=max_updates, input_loader=inputs, family=FAMILY)


if __name__ == '__main__':
    main(parser().parse_args())
