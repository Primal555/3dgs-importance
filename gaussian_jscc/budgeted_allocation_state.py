"""Atomic exact resume; numeric logs use unique step/event keys."""
import json
from pathlib import Path
import time
import torch
from .modular_checkpoint import capture_rng


def save_state(out, label, model, table, controller, blocks, views, progress, plan, scene_record, reference, q, *, learner=None):
    path = Path(out)/'checkpoints'/label/'state.pt'; path.parent.mkdir(parents=True, exist_ok=True)
    family = plan['version']
    if family not in ('budgeted_allocation_v1', 'fixed_beta_allocation_v1', 'evidence_allocation_v1',
                      'structured_evidence_allocation_v1', 'calibrated_structured_allocation_v2',
                      'learned_benefit_allocation_v1','learned_benefit_factored_v2','render_response_value_v1'):
        raise ValueError('unknown allocation checkpoint family')
    state = dict(version=1, family=family, state_dict=model.state_dict(),
        config=model.cfg.to_dict(), table=table.state_dict(),
        controller=controller.state_dict() if controller is not None else None,
        blocks=blocks.state_dict(), views=views.state_dict(), progress=dict(progress),
        plan=plan, scene_record=scene_record, latest_q=q.detach().cpu(), rng=capture_rng())
    state['objective_reference' if family != 'budgeted_allocation_v1' else 'budget_reference'] = reference
    if family in ('learned_benefit_allocation_v1','learned_benefit_factored_v2','render_response_value_v1'):
        if learner is None: raise ValueError('learned benefit checkpoint requires learner state')
        state['learner'] = learner.state_dict()
    temporary = path.with_suffix('.pt.tmp'); torch.save(state, temporary); temporary.replace(path)
    return state


def restore_state(path, plan, scene_record):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('family') != plan['version'] or state['plan'] != plan
            or state['scene_record'] != scene_record):
        raise ValueError('exact resume source/settings/scene mismatch')
    if state['family'] in ('fixed_beta_allocation_v1','evidence_allocation_v1','structured_evidence_allocation_v1',
                           'calibrated_structured_allocation_v2','learned_benefit_allocation_v1','learned_benefit_factored_v2',
                           'render_response_value_v1'):
        from .allocation_objective import FixedRateDistortion
        ref = state['objective_reference']
        objective = FixedRateDistortion(ref['beta'], ref['d_ref'], ref['c_ref'])
        if (state['controller'] is not None or 'budget' in ref or objective.beta != plan['settings']['beta']
                or ref['c_ref'] != ref['q3_cost']['total_channel_uses']
                or ref['d_ref'] != max(ref['d_ref_raw'], ref['d_ref_floor'])):
            raise ValueError('fixed RD checkpoint objective/reference mismatch')
        if state['family'] in ('structured_evidence_allocation_v1','calibrated_structured_allocation_v2',
                              'learned_benefit_allocation_v1','learned_benefit_factored_v2','render_response_value_v1'):
            from .allocation_distortion import calibrate
            settings = plan['settings']
            expected = calibrate(ref['mse_ref'], ref['structure_ref'], mode=settings['distortion'],
                share=settings['structure_share'], lambda_s=settings['lambda_s'])
            if any(ref.get(k) != v for k,v in expected.items()):
                raise ValueError('structured distortion reference mismatch')
    return state


def read_log(path):
    text = path.read_text(encoding='utf-8') if path.exists() else ''
    lines = [s for s in text.splitlines() if s.strip()]
    rows, partial = [], ''
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index != len(lines)-1 or text.endswith('\n'):
                raise ValueError(f'corrupt interior log record: {path}:{index+1}') from exc
            partial = line
    return rows, text, partial


def quarantine(path, text):
    if text:
        archive = path.with_name(f'{path.stem}.uncheckpointed_{time.time_ns()}.jsonl')
        archive.write_text(text+'\n', encoding='utf-8')


def rewrite(path, rows):
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows), encoding='utf-8')
    temporary.replace(path)


def upsert_jsonl(path, row, key=('step', 'event')):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    old, text, partial = read_log(path)
    if partial or (text and not text.endswith('\n')):
        quarantine(path, partial); rewrite(path, old)
    identity = tuple(row[k] for k in key)
    match = [i for i, r in enumerate(old) if tuple(r[k] for k in key) == identity]
    if match:
        old[match[0]] = row
        rewrite(path, old)
    else:
        with path.open('a', encoding='utf-8') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')


def repair(out, step):
    for path in (Path(out)/'metrics').glob('*.jsonl'):
        if '.uncheckpointed' in path.name: continue
        rows, text, partial = read_log(path)
        kept = [r for r in rows if r['step'] <= step]
        if partial or len(kept) != len(rows) or (text and not text.endswith('\n')):
            quarantine(path, '\n'.join([json.dumps(r) for r in rows if r['step']>step]+([partial] if partial else [])))
            rewrite(path, kept)
