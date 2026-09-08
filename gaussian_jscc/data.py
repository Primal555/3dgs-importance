"""3DGS PLY layout, global coordinate normalization and spatial packet ordering."""

from pathlib import Path
import math

import numpy as np
import torch
from torch.nn import functional as F
from plyfile import PlyData, PlyElement


def read_ply(path):
    vertex = PlyData.read(str(path))["vertex"].data
    names = vertex.dtype.names
    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    degree = math.isqrt(1 + len(rest) // 3) - 1
    if degree not in range(4) or len(rest) != 3 * ((degree + 1) ** 2 - 1):
        raise ValueError("PLY must contain complete 3DGS SH coefficients of degree 0..3")
    fields = ["x", "y", "z", "opacity"] + [f"scale_{i}" for i in range(3)]
    fields += [f"rot_{i}" for i in range(4)] + [f"f_dc_{i}" for i in range(3)] + rest
    missing = set(fields) - set(names)
    if missing:
        raise ValueError(f"Not a trained 3DGS PLY; missing {sorted(missing)}")
    values = np.stack([vertex[n] for n in fields], -1).astype(np.float32)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("input PLY must contain nonempty finite Gaussian parameters")
    raw = torch.from_numpy(values)
    rotation = raw[:, 7:11]
    if (rotation.norm(dim=-1) < 1e-8).any():
        raise ValueError("input contains zero quaternions")
    rotation = F.normalize(rotation, dim=-1)
    # Use the largest-magnitude component to choose a consistent quaternion sign.
    pivot = rotation.gather(1, rotation.abs().argmax(-1, keepdim=True))
    raw[:, 7:11] = rotation * torch.where(pivot < 0, -1., 1.)
    return raw, degree


def write_ply(path, raw, degree):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = raw.detach().cpu().numpy().astype(np.float32)
    fields = ["x", "y", "z", "nx", "ny", "nz"]
    fields += [f"f_dc_{i}" for i in range(3)]
    fields += [f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))]
    fields += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    values = np.concatenate((a[:, :3], np.zeros((len(a), 3), np.float32),
                             a[:, 11:], a[:, 3:7], a[:, 7:11]), -1)
    out = np.empty(len(a), dtype=[(f, "f4") for f in fields])
    for i, f in enumerate(fields):
        out[f] = values[:, i]
    PlyData([PlyElement.describe(out, "vertex")], text=False).write(str(path))


def load_tiers(path, n, uniform=None):
    if path:
        a = np.load(path, allow_pickle=False)
        if a.shape != (n,) or not np.issubdtype(a.dtype, np.integer) or ((a < 0) | (a > 3)).any():
            raise ValueError("rate map must be an integer .npy array [N], tiers 0..3, in input PLY order")
        return torch.from_numpy(a.astype(np.int64))
    if uniform not in (1, 2, 3):
        raise ValueError("specify --rate-map or --uniform-tier 1/2/3")
    return torch.full((n,), uniform, dtype=torch.long)


def morton_order(coords):
    """True bit-interleaved Morton ordering, with stable handling of duplicate cells."""
    a = np.asarray(coords, dtype=np.uint64)
    key = np.zeros(len(a), dtype=np.uint64)
    for bit in range(16):
        for axis in range(3):
            key |= ((a[:, axis] >> bit) & 1) << (3 * bit + axis)
    return np.argsort(key, kind="stable")


class Geometry:
    def __init__(self, lower, span, bits):
        self.lower = torch.as_tensor(lower, dtype=torch.float32).cpu()
        self.span = torch.as_tensor(span, dtype=torch.float32).cpu()
        self.bits = bits
        if not isinstance(bits, int) or not 1 <= bits <= 16:
            raise ValueError("geometry bits must be an integer in 1..16")
        if self.lower.shape != (3,) or self.span.shape != (3,) or not (
            torch.isfinite(self.lower).all() and torch.isfinite(self.span).all()
        ) or (self.span <= 0).any():
            raise ValueError("invalid geometry bounds")

    @classmethod
    def fit(cls, xyz, bits):
        return cls(xyz.amin(0), (xyz.amax(0) - xyz.amin(0)).clamp_min(1e-6), bits)

    def quantize(self, xyz):
        return (((xyz.cpu() - self.lower) / self.span).clamp(0, 1) *
                (2 ** self.bits - 1)).round().long()

    def normalize(self, xyz):
        return ((xyz - self.lower.to(xyz.device)) / self.span.to(xyz.device)).clamp(0, 1)

    def denormalize(self, unit):
        return self.lower.to(unit.device) + unit * self.span.to(unit.device)

    def to_dict(self):
        return {"lower": self.lower.tolist(), "span": self.span.tolist(), "bits": self.bits}


def to_features(raw, geometry, model):
    unit = geometry.normalize(raw[:, :3])
    attrs = (raw[:, 3:] - model.attr_mean) / model.attr_std
    return torch.cat((unit, attrs), -1), unit


def to_raw(features, geometry, model):
    xyz = geometry.denormalize(features[:, :3].clamp(0, 1))
    attrs = features[:, 3:] * model.attr_std + model.attr_mean
    rotation = attrs[:, 4:8]
    identity = torch.zeros_like(rotation)
    identity[:, 0] = 1
    rotation = torch.where(rotation.norm(dim=-1, keepdim=True) > 1e-8,
                           F.normalize(rotation, dim=-1), identity)
    # Physical constraints before rendering. Never multiply opacity by tier 1/2/3.
    attrs = torch.cat((attrs[:, :1].clamp(-20, 20), attrs[:, 1:4].clamp(-20, 10),
                       rotation, attrs[:, 8:]), -1)
    return torch.cat((xyz, attrs), -1)


def prepare(raw, bits, q=None):
    geometry = Geometry.fit(raw[:, :3], bits)
    coords = geometry.quantize(raw[:, :3])
    order = torch.from_numpy(morton_order(coords.numpy()).astype(np.int64))
    return raw[order], geometry, (q[order] if q is not None else None)


def attribute_loss(pred, target):
    # Group-balanced loss: 45 high-order SH terms must not dominate geometry.
    groups = ((0, 3), (3, 4), (4, 7), (7, 11), (11, 14), (14, target.shape[1]))
    return torch.stack([F.smooth_l1_loss(pred[:, a:b], target[:, a:b])
                        for a, b in groups if b > a]).mean()
