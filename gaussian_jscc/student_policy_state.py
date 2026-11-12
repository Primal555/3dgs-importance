"""Compact student-only atomic checkpoints and strict source preparation identity."""
from pathlib import Path
import torch
from .modular_checkpoint import capture_rng, restore_rng
from .student_policy_features import atomic_torch

FAMILY = 'feedforward_student_policy_v1'


def cpu_copy(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_copy(v) for v in value)
    return value


def save_student(out, label, student, optimizer, progress, plan, sampler, views, *, evaluated_tiers=None):
    state = dict(family=FAMILY, config=student.cfg.to_dict(), student=cpu_copy(student.state_dict()),
        optimizer=cpu_copy(optimizer.state_dict()), progress=cpu_copy(progress), plan=plan,
        sampler=sampler.state_dict(), views=views.state_dict(), rng=capture_rng())
    if evaluated_tiers is not None:
        from .budgeted_allocation_evaluation import tier_hash
        q = evaluated_tiers.detach().cpu().long()
        if q.ndim!=1 or ((q<0)|(q>3)).any() or tier_hash(q)!=progress['best']['tier_sha256']:
            raise ValueError('evaluated best map differs from checkpoint selection')
        state['evaluated_tiers'] = q.to(torch.uint8).clone()
    atomic_torch(Path(out)/'checkpoints'/label/'student.pt', state)
    return state


def selected_map(state, inferred, expected):
    """Use the evaluated map if available; explicitly mark legacy reconstruction."""
    from .budgeted_allocation_evaluation import tier_hash
    if state['progress']['best'] != expected: raise ValueError('best checkpoint selection identity mismatch')
    saved = state.get('evaluated_tiers')
    q = inferred.detach().cpu().long() if saved is None else saved.cpu().long()
    if q.shape!=inferred.shape or ((q<0)|(q>3)).any(): raise ValueError('invalid saved best map')
    actual = tier_hash(q)
    if saved is not None and actual!=expected['tier_sha256']: raise ValueError('saved evaluated map hash mismatch')
    report = dict(source='reconstructed_legacy' if saved is None else 'saved_evaluated_map',
        original_tier_sha256=expected['tier_sha256'], deployed_tier_sha256=actual,
        inferred_tier_sha256=tier_hash(inferred), matches_original_evaluation=actual==expected['tier_sha256'],
        inference_differs_from_deployed_points=int((q!=inferred.cpu()).sum()),
        original_calibration_rd=expected['rd_score'], selected_step=expected['step'])
    return q, report


def recoverable_selected_map(state, inferred, expected, checkpoint_path, recovery_path):
    """Pin a legacy reconstruction before rendering so interrupted finalization resumes exactly."""
    import hashlib
    from .budgeted_allocation_evaluation import tier_hash
    if state['progress']['best'] != expected: raise ValueError('best checkpoint selection identity mismatch')
    if 'evaluated_tiers' in state: return selected_map(state,inferred,expected)
    with Path(checkpoint_path).open('rb') as stream: identity=hashlib.file_digest(stream,'sha256').hexdigest()
    recovery_path=Path(recovery_path)
    if recovery_path.is_file():
        record=torch.load(recovery_path,map_location='cpu',weights_only=True)
        if (record['checkpoint_sha256']!=identity or record['expected']!=expected or
                tier_hash(record['tiers'])!=record['report']['deployed_tier_sha256']):
            raise ValueError('legacy selection recovery identity/hash mismatch')
        report=dict(record['report'],inferred_tier_sha256=tier_hash(inferred),
            inference_differs_from_deployed_points=int((record['tiers']!=inferred.cpu()).sum()))
        return record['tiers'].long(),report
    q,report=selected_map(state,inferred,expected)
    atomic_torch(recovery_path,dict(checkpoint_sha256=identity,expected=expected,tiers=q,report=report))
    return q,report


def read_student(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('family') != FAMILY: raise ValueError('student checkpoint family mismatch')
    return state


def restore_student(state, student, optimizer, sampler, views, plan):
    if state['plan'] != plan or state['config'] != student.cfg.to_dict():
        raise ValueError('student resume settings/source/cache identity mismatch')
    student.load_state_dict(state['student']); optimizer.load_state_dict(state['optimizer'])
    sampler.load_state_dict(state['sampler']); views.load_state_dict(state['views']); restore_rng(state['rng'])
    return cpu_copy(state['progress'])
