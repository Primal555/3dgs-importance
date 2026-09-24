"""Explicit center-only continuation, preserving optimizer/RNG and the source run."""
import copy
import json
import math
import shutil
from pathlib import Path


def prepare_continuation(saved, checkpoint, out, steps, lr, end_lr):
    if steps < 1 or not all(math.isfinite(x) and x > 0 for x in (lr, end_lr)) or end_lr > lr:
        raise ValueError('require positive additional steps and 0 < end LR <= start LR')
    progress = saved['progress']
    if (saved.get('format') != 'center_attribute_training_v2' or
            progress['phase'] != 'center' or progress['phase_index'] != 0 or
            not saved.get('optimizer') or not progress['phase_initialized']):
        raise ValueError('use training_state_last_center.pt with matching center Adam state')
    source = Path(checkpoint).resolve().parent
    target = Path(out).resolve()
    if target.exists() or target == source or source in target.parents:
        raise ValueError('continuation requires a NEW directory outside the original run')
    # Validate supporting history before creating anything. The checkpoint is
    # portable: use its parent, not its possibly obsolete saved output path.
    selected = [source/f'codec_selected_{phase}_{step}.pt'
                for phase, step in progress['best_steps'].items()]
    for path in selected:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, count in saved['log_counts'].items():
        path = source/name
        if count and (not path.is_file() or len(path.read_text(encoding='utf-8').splitlines()) < count):
            raise ValueError(f'missing/truncated checkpoint history: {path}')
    result = copy.deepcopy(saved)
    args, state = result['arguments'], result['progress']
    offset = state['phase_step']
    args.update(out=str(target), resume=None, center_steps=offset+steps,
                min_center_steps=offset+steps+1, attribute_steps=0, joint_steps=0,
                center_lr=lr, center_lr_schedule='late_cosine',
                center_lr_decay_start=0., center_lr_end_ratio=end_lr/lr,
                center_lr_offset=offset)
    state.update(status='running', end_reason=None)
    for group in result['optimizer']['param_groups']:
        group['lr'] = lr
    target.mkdir(parents=True)
    for name, count in saved['log_counts'].items():
        if count:
            lines = (source/name).read_text(encoding='utf-8').splitlines()
            (target/name).write_text('\n'.join(lines[:count])+'\n', encoding='utf-8')
    for path in selected:
        shutil.copy2(path, target/path.name)
    for name in ('images', 'center_drift', 'charts'):
        if (source/name).is_dir():
            shutil.copytree(source/name, target/name)
    (target/'continuation.json').write_text(json.dumps({
        'source_checkpoint': str(Path(checkpoint).resolve()), 'start_step': state['step'],
        'additional_center_updates': steps, 'start_lr': lr, 'end_lr': end_lr,
        'optimizer_rng_and_soft_history_retained': True}, indent=2), encoding='utf-8')
    return result
