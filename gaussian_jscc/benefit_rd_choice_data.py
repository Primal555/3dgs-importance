"""Read the immutable paired experiment, its configuration and fixed records."""
import copy
import hashlib
import io
import json
from pathlib import Path

from .benefit_stage_one_data import evaluator_settings
from .budgeted_allocation_state import load_checkpoint
from .spatial_benefit_replay import SpatialReplay


def load_trial(run, replay_root=None, pipeline_root=None, cache_mib=256., require_replay=True):
    run = Path(run).resolve()
    initial_payload = (run/'initial.pt').read_bytes()
    initial = load_checkpoint(io.BytesIO(initial_payload))
    summary = json.loads((run/'summary.json').read_text(encoding='utf8'))
    manifest = json.loads((run/'data_manifest.json').read_text(encoding='utf8'))
    settings = initial['settings']
    if summary['status'] != 'complete' or settings['source_checkpoint_sha256'] != summary['source_snapshot_sha256']:
        raise ValueError('completed consistent source comparison required')
    root = Path(replay_root or manifest['replay_root']).resolve()
    if require_replay and not root.is_dir():
        raise FileNotFoundError(f'replay missing: {root}; supply --replay-root after relocation')
    entries = manifest['training']+manifest['validation']+manifest['recent_training']
    for entry in entries if require_replay else ():
        path = (root/entry['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f'missing or outside replay shard: {path}')
    if not manifest['training'] or not manifest['validation']:
        raise ValueError('fixed training and validation manifest required')
    if len({e['id'] for e in manifest['training']}) != len(manifest['training']):
        raise ValueError('duplicate fixed training records')
    if len({e['id'] for e in manifest['validation']}) != len(manifest['validation']):
        raise ValueError('duplicate fixed validation records')
    if {e['id'] for e in manifest['training']} & {e['id'] for e in manifest['validation']}:
        raise ValueError('training/validation record overlap')
    original_root = Path(settings['source_checkpoint']).parents[3]
    roots = [Path(pipeline_root).resolve()] if pipeline_root else [original_root, root.parent.parent]
    embedded = settings.get('evaluator_configuration')
    state = dict(plan=dict(settings=copy.deepcopy(embedded or settings)), learner=initial['learner'])
    errors = []
    for candidate in roots:
        try:
            config, config_source = evaluator_settings(state, candidate)
            if embedded:
                config_source = 'initial.pt settings.evaluator_configuration'
            break
        except ValueError as error:
            errors.append(str(error))
    else:
        raise ValueError('cannot recover evaluator settings; retain original benefit_network.pt or use '
                         '--pipeline-root: '+'; '.join(errors))
    if any(config[k] != settings[v] for k, v in (('benefit_lr', 'learning_rate'),
            ('weight_decay', 'weight_decay'), ('decision_weight', 'decision_weight'))):
        raise ValueError('immutable evaluator settings disagree with original offline comparison')
    replay = (SpatialReplay(root, max_groups=1, validation_groups=len(manifest['validation']), cache_mib=cache_mib)
              if require_replay else None)
    return dict(initial=initial, initial_sha256=hashlib.sha256(initial_payload).hexdigest(),
        summary=summary, manifest=manifest, replay=replay, config=config, config_source=config_source)
