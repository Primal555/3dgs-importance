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


def save_student(out, label, student, optimizer, progress, plan, sampler, views):
    state = dict(family=FAMILY, config=student.cfg.to_dict(), student=cpu_copy(student.state_dict()),
        optimizer=cpu_copy(optimizer.state_dict()), progress=cpu_copy(progress), plan=plan,
        sampler=sampler.state_dict(), views=views.state_dict(), rng=capture_rng())
    atomic_torch(Path(out)/'checkpoints'/label/'student.pt', state)
    return state


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
