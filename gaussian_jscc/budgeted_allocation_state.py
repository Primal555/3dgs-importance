"""Atomic exact resume; numeric logs use unique step/event keys."""
import json
from pathlib import Path
import torch
from .modular_checkpoint import capture_rng


def save_state(out, label, model, table, controller, blocks, views, progress, plan, scene_record, reference, q):
    path = Path(out)/'checkpoints'/label/'state.pt'; path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(version=1, family='budgeted_allocation_v1', state_dict=model.state_dict(),
        config=model.cfg.to_dict(), table=table.state_dict(), controller=controller.state_dict(),
        blocks=blocks.state_dict(), views=views.state_dict(), progress=dict(progress),
        plan=plan, scene_record=scene_record, budget_reference=reference, latest_q=q.detach().cpu(), rng=capture_rng())
    temporary = path.with_suffix('.pt.tmp'); torch.save(state, temporary); temporary.replace(path)
    return state


def restore_state(path, plan, scene_record):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('family') != 'budgeted_allocation_v1' or state['plan'] != plan
            or state['scene_record'] != scene_record):
        raise ValueError('exact resume source/settings/scene mismatch')
    return state


def upsert_jsonl(path, row, key=('step', 'event')):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    old = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()] if path.exists() else []
    identity = tuple(row[k] for k in key)
    match = [i for i, r in enumerate(old) if tuple(r[k] for k in key) == identity]
    if match:
        old[match[0]] = row
        temporary = path.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in old), encoding='utf-8')
        temporary.replace(path)
    else:
        with path.open('a', encoding='utf-8') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')


def repair(out, step):
    for path in (Path(out)/'metrics').glob('*.jsonl'):
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
        kept = [r for r in rows if r['step'] <= step]
        if len(kept) != len(rows):
            path.with_suffix('.uncheckpointed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']>step), encoding='utf-8')
            path.write_text(''.join(json.dumps(r)+'\n' for r in kept), encoding='utf-8')
