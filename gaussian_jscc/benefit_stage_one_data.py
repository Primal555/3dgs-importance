"""Immutable checkpoint view and bounded reads of existing teacher shards."""
import copy
import hashlib
import io
import json
from pathlib import Path

import torch

from .budgeted_allocation_state import load_checkpoint
from .full_history_spatial_replay import FullHistorySpatialReplay


EVALUATOR_FIELDS = ('hidden', 'benefit_lr', 'dropout', 'weight_decay', 'decision_weight',
                    'tie_fraction', 'common_weight', 'pair_weight',
                    'training_probe_groups', 'batch_groups')


def evaluator_settings(state, root):
    """Allocation plans omit evaluator knobs; recover the actual provider config."""
    settings = dict(state['plan']['settings'])
    sources = ['checkpoint.plan.settings']
    if any(key not in settings for key in EVALUATOR_FIELDS):
        candidates = [root/'benefit_network.pt']
        specified = settings.get('benefit_checkpoint')
        if specified:
            candidates.append(Path(specified))
        candidates += [root/'pretrain/plan.json', root/'plan.json']
        visited = set()
        for path in candidates:
            path = path.resolve()
            if path in visited or not path.is_file():
                continue
            visited.add(path)
            if path.suffix == '.pt':
                artifact = load_checkpoint(path)
                expected = state.get('positive_provider', {}).get('initial_network_sha256')
                if expected is not None and artifact.get('network_sha256') != expected:
                    raise ValueError(f'evaluator initializer identity mismatch: {path}')
                metadata = artifact.get('settings', {})
            else:
                metadata = json.loads(path.read_text(encoding='utf8')).get('settings', {})
            settings.update({key: metadata[key] for key in EVALUATOR_FIELDS if key in metadata})
            sources.append(str(path))
            if all(key in settings for key in EVALUATOR_FIELDS):
                break
    missing = [key for key in EVALUATOR_FIELDS if key not in settings]
    if missing:
        raise ValueError('missing evaluator settings: '+', '.join(missing)+
            '; retain benefit_network.pt or the pipeline/pretrain plan.json beside the source run')
    learner = state.get('positive_provider', {}).get('learner') or state.get('learner')
    width = learner['network']['points.0.weight'].shape[0]
    if settings['hidden'] != width:
        raise ValueError(f'evaluator settings hidden={settings["hidden"]} differs from checkpoint width={width}')
    return settings, ' + '.join(sources)


def snapshot(run, replay_root=None, cache_mib=256.):
    run = Path(run).resolve()
    if run.is_file():
        path = run
        root = path.parents[3] if path.parents[1].name == 'checkpoints' and path.parents[2].name in ('allocation', 'pretrain') else path.parent
    else:
        root = run
        candidates = [run/'allocation/checkpoints/latest/state.pt',
                      run/'pretrain/checkpoints/latest/state.pt', run/'checkpoints/latest/state.pt']
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'no evaluator checkpoint: {run}')
    if path.parents[1].name == 'checkpoints' and path.parents[2].name in ('allocation', 'pretrain'):
        root = path.parents[3]
    # A single file handle survives an atomic replacement by the live trainer.
    with path.open('rb') as stream:
        payload = stream.read()
    digest = hashlib.sha256(payload).hexdigest()
    state = load_checkpoint(io.BytesIO(payload))
    learner = state.get('positive_provider', {}).get('learner') or state.get('learner')
    replay_state = state.get('replay')
    if learner is None or learner.get('scales') is None or replay_state is None:
        raise ValueError('a fitted online/pretraining evaluator and its replay are required')
    if replay_state.get('full_history_policy') != FullHistorySpatialReplay.policy:
        raise ValueError('stage one requires the current complete-history experiment')
    settings, settings_source = evaluator_settings(state, root)
    replay_state = copy.deepcopy(replay_state)
    if replay_root:
        replay_state['root'] = str(Path(replay_root).resolve())
    folder = Path(replay_state['root'])
    if not folder.is_dir():
        raise FileNotFoundError(f'replay directory missing: {folder}; supply --replay-root after relocation')
    replay = FullHistorySpatialReplay(folder, max_groups=replay_state['max_groups'],
        validation_groups=replay_state['validation_groups'], cache_mib=cache_mib)
    replay.load_state_dict(replay_state)
    return dict(source_root=root, source_checkpoint=str(path), source_sha256=digest,
        source_step=int(state.get('progress', {}).get('step', 0)),
        source_network_updates=int(learner['updates']), learner=learner,
        settings=settings, evaluator_settings_source=settings_source, replay=replay)


def fixed_entries(replay, count, seed):
    if not replay.training or not replay.validation:
        raise ValueError('both training and independent validation records are required')
    order = torch.randperm(len(replay.training), generator=torch.Generator().manual_seed(seed))
    indices = sorted(order[:min(count, len(order))].tolist())
    training = [replay.training[i] for i in indices]
    return training, list(replay.validation)


def read_groups(replay, entries):
    for entry in entries:
        group = replay._read(entry)
        group.pop('replay_importance_weight', None)
        yield group


def compact_group(g):
    keep = {key: g[key] for key in ('source', 'step', 'rates', 'supervision', 'old_tier',
            'new_tier', 'delta_values', 'rate_delta') if key in g}
    keep['views'] = [dict(values=v['values']) if 'values' in v else {} for v in g['views']]
    return keep
