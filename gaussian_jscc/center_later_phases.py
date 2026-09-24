"""Start B/C from the explicitly selected best center model, not last weights."""
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch


def prepare_later_phases(request):
    source = Path(request.after_center_run).resolve()
    target = Path(request.out).resolve()
    if target.exists() or source == target or source in target.parents:
        raise ValueError('choose a NEW output directory outside the center run')
    saved = torch.load(source/'training_state.pt', map_location='cpu', weights_only=True)
    if saved.get('format') != 'center_attribute_training_v2':
        raise ValueError('expected center_attribute_training_v2')
    progress = saved['progress']
    if progress['status'] != 'complete' or progress['completed'].get('center', 0) < 1:
        raise ValueError('finish the center-only run before starting B/C')
    if progress['completed'].get('attribute', 0) or progress['completed'].get('joint', 0):
        raise ValueError('source must be a center-only run; use --resume for interrupted B/C')
    step = progress['best_steps']['center']
    path = source/f'codec_selected_center_{step}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['config'] != saved['config'] or checkpoint['step'] != step:
        raise ValueError('selected center checkpoint metadata mismatch')
    if request.attribute_steps < 1 or request.joint_steps < 0:
        raise ValueError('positive attribute budget and nonnegative joint budget required')
    result = copy.deepcopy(saved)
    args = result['arguments']
    for name in ('attribute_steps', 'joint_steps', 'attribute_lr', 'joint_center_lr',
                 'joint_attribute_lr', 'render_backward', 'render_blocks_per_batch',
                 'validate_every', 'render_every', 'save_every', 'profile_every',
                 'later_phase_policy', 'device'):
        args[name] = getattr(request, name)
    # Architecture, normalization, scene, heldout split and seed come from A.
    args.update(out=str(target), resume=None, after_center_run=None, extend_center_steps=0)
    for name in ('ply', 'source'):
        if getattr(request, name, None):
            args[name] = getattr(request, name)
    if args['joint_steps'] and not args.get('source'):
        raise ValueError('joint rendering requires a scene source')
    from .center_attribute_train import check_args
    check_args(SimpleNamespace(**args))
    result['model'] = checkpoint['state_dict']
    result['optimizer'] = None  # New tasks: never attach LAST-center Adam to BEST weights.
    result['progress'] = {
        'phase': 'attribute', 'phase_index': 1, 'phase_step': 0,
        'step': progress['step'], 'status': 'running', 'phase_initialized': False,
        'end_reason': None, 'best': {'center': progress['best']['center']},
        'best_steps': {'center': step}, 'stale': 0, 'initial_attribute_loss': None,
        'last_validation': None,
        'completed': {'center': progress['completed']['center'], 'attribute': 0, 'joint': 0}}
    result['log_counts'] = {}
    provenance = {'source_run': str(source), 'selected_center_step': step,
                  'source_run_updates': progress['step'], 'center_gate_bypassed_explicitly': True,
                  'fresh_adam_for_each_new_phase': True,
                  'later_phase_policy': request.later_phase_policy,
                  'history': 'B/C diagnostics in this directory; A history remains in source_run'}
    target.mkdir(parents=True)
    shutil.copy2(path, target/path.name)
    shutil.copy2(path, target/'codec_center_initial.pt')
    (target/'phase_start.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return result
