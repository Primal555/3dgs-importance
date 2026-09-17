"""Learned-codec reconstruction objectives; weights are explicit hyperparameters."""
import torch
from torch.nn import functional as F
from .learned_objective import learned_terms


def rotation_matrix(q):
    """Scalar-first unit quaternion -> matrix; invariant to q versus -q."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def reconstruction_loss(pred, target, geometry, model, seed=None, seed_weight=0.,
                        active=None, reduction="mean", return_terms=False):
    """Source-normalized parameter regularizer; not a mask-drop reward.

    geometry/seed arguments retain the shared benchmark/replay calling convention;
    learned XYZ supervision is direct and has no bootstrap/seed term.
    """
    terms = learned_terms(pred, target, model)
    row = sum(getattr(model.cfg, name + "_weight") * value for name, value in terms.items())
    if active is not None:
        row = row * active.detach()
        terms = {key: value * active.detach() for key, value in terms.items()}
    if reduction == "sum":
        loss = row.sum()
    elif reduction == "mean":
        loss = row.sum() / max(1, len(row))
    elif reduction == "none":
        loss = row
    else:
        raise ValueError("unknown loss reduction")
    return (loss, terms) if return_terms else loss
