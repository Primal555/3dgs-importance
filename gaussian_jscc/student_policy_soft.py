"""Exact ten-draw teacher targets and retention-weighted hierarchical distillation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .allocation import DEPLOYMENT_DRAWS, deployment_probabilities, deployment_tiers
from .budgeted_allocation_evaluation import tier_hash
from .student_policy_features import atomic_torch


def _hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def load_soft_teacher(root, teacher_q, out, *, seed, required_cap=None, chunk_size=2048):
    root, out = Path(root), Path(out)
    read = lambda name: json.loads((root/name).read_text(encoding='utf8'))
    meta, plan = read('training.json'), read('plan.json')
    complete, selection = read('complete.json'), read('deployment/selection.json')
    cap = plan['settings'].get('tier_max_log_odds')
    if required_cap is not None and cap != required_cap:
        raise ValueError(f'teacher positive cap mismatch: expected {required_cap}, got {cap}')
    if meta.get('deployment_draws') != DEPLOYMENT_DRAWS or meta.get('deployment_seed') != seed:
        raise ValueError('teacher ten-draw deployment rule/seed mismatch')
    if complete.get('selected_step') is None or complete['selected_step'] != selection.get('selected_step'):
        raise ValueError('teacher selected checkpoint identity mismatch')
    path = root/'deployment/selected_probabilities.npy'
    array = np.load(path, allow_pickle=False)
    if array.dtype.kind != 'f' or array.shape != (len(teacher_q), 4):
        raise ValueError('teacher probabilities must be original-row [N,4] floats')
    p = torch.from_numpy(array)
    if (not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any() or
            not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-7, rtol=1e-6)):
        raise ValueError('invalid categorical teacher probabilities')
    # Preserve exported dtype and full-scene RNG ordering for the identity check.
    if not torch.equal(deployment_tiers(p, seed=seed), teacher_q.cpu()):
        raise ValueError('selected teacher probabilities do not reproduce the selected map in original PLY order')
    identity = dict(version='ten_draw_hierarchical_soft_v1',
        probability_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        selected_step=complete['selected_step'], teacher_tier_cap=cap,
        selected_map_sha256=tier_hash(teacher_q), seed=seed, draws=DEPLOYMENT_DRAWS,
        positive_weighting='expected retention; normalized by sum of retention probabilities')
    cache = out/'preparation/teacher_soft_targets.pt'
    if cache.is_file():
        saved = torch.load(cache, map_location='cpu', weights_only=True)
        target = saved['target']
        if (saved['identity'] != identity or target.shape != p.shape or target.dtype != torch.float32 or
                saved['target_sha256'] != _hash(target)):
            raise ValueError('soft teacher cache identity/content mismatch')
        return target, identity
    if chunk_size < 1: raise ValueError('positive soft target chunk size required')
    target = torch.empty(p.shape, dtype=torch.float32)
    print(f'Preparing ten-draw soft teacher: selected step={identity["selected_step"]}, cap={cap}, points={len(p)}', flush=True)
    for start in range(0, len(p), chunk_size):
        result = deployment_probabilities(p[start:start+chunk_size].double())
        # Numerical normalization only; tie priorities were computed in exported precision.
        target[start:start+chunk_size] = (result/result.sum(1, keepdim=True)).float()
    atomic_torch(cache, dict(identity=identity, target=target, target_sha256=_hash(target)))
    return target, identity


def soft_teacher_loss(keep_logits, tier_logits, target):
    """Two branch means; one-hot targets reproduce the hard teacher objective."""
    target = target.to(keep_logits)
    retained = target[:, 1:].sum(1)
    keep_target = torch.stack((target[:, 0], retained), 1)
    log_keep, log_tier = keep_logits.log_softmax(-1), tier_logits.log_softmax(-1)
    keep = -(keep_target*log_keep).sum(1).mean()
    weight_sum = retained.sum().clamp_min(torch.finfo(target.dtype).tiny)
    tier = -(target[:, 1:]*log_tier).sum()/weight_sum
    with torch.no_grad():
        conditional = target[:, 1:]/retained[:, None].clamp_min(torch.finfo(target.dtype).tiny)
        keep_entropy = -torch.special.xlogy(keep_target, keep_target).sum(1).mean()
        tier_entropy = -(retained[:, None]*torch.special.xlogy(conditional, conditional)).sum()/weight_sum
        q = torch.where(keep_logits[:, 1] >= keep_logits[:, 0], tier_logits.argmax(1)+1, 0)
        expected_agreement = target.gather(1, q[:, None]).mean()
    return keep+tier, dict(keep_ce=float(keep.detach()), tier_ce=float(tier.detach()),
        teacher_keep_kl=float(keep.detach()-keep_entropy), teacher_tier_kl=float(tier.detach()-tier_entropy),
        teacher_expected_action_agreement=float(expected_agreement))
