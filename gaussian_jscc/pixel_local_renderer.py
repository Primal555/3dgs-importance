"""Frozen, exact MaskGaussian tile capture and whole-block interventions.

The CUDA extension compiles the repository's original preprocessing kernel into
its own library. All cached arrays own tensor storage. Tile lists contain every
geometric contributor, including zero-mask and early-terminated contributors.
The CPU backend implements the same projection and compositing definitions for
small mathematical fixtures; production capture defaults to CUDA.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import torch
from torch.nn import functional as F


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parents[1]
    source = root / "gaussian_jscc/csrc/pixel_local"
    original = root / "submodules/mask-diff-gaussian-rasterization"
    if not (original / "cuda_rasterizer/forward.cu").is_file():
        raise RuntimeError("pixel-local CUDA requires the mask rasterizer source submodule")
    return load(name="maskgaussian_pixel_local_v1", sources=[str(source / "bindings.cpp"),
        str(source / "kernels.cu")], extra_include_paths=[str(original / "cuda_rasterizer"),
        str(original / "third_party/glm")], extra_cuda_cflags=["-O3"], extra_cflags=["-O3"])


def _rectangles(p, height, width):
    """Match getRect's integer truncation and exclusive upper tile bound."""
    grid = p.new_tensor([(width + 15)//16, (height + 15)//16], dtype=torch.long)
    low = torch.trunc((p[:, :2] - p[:, 11:12])/16).long().clamp_min(0)
    high = torch.trunc((p[:, :2] + p[:, 11:12] + 15)/16).long().clamp_min(0)
    low, high = torch.minimum(low, grid), torch.minimum(high, grid)
    high = torch.where((p[:, 11] > 0)[:, None], high, low)
    return low, high


def _members(p, height, width):
    low, high = _rectangles(p, height, width)
    sizes = (high-low).clamp_min(0)
    counts = sizes.prod(-1)
    rows = torch.repeat_interleave(torch.arange(len(p), device=p.device), counts)
    starts = counts.cumsum(0)-counts
    offset = torch.arange(len(rows), device=p.device)-torch.repeat_interleave(starts, counts)
    sx = sizes[rows, 0]
    tiles = (low[rows, 1] + offset.div(sx, rounding_mode="floor"))*((width+15)//16)
    tiles = tiles + low[rows, 0] + offset.remainder(sx)
    return tiles, rows


def _sort_entries(tiles, rows, depths):
    # Stable ties follow original scene row IDs, even for reordered block IDs.
    order = torch.argsort(rows, stable=True)
    order = order[torch.argsort(depths[order], stable=True)]
    return order[torch.argsort(tiles[order], stable=True)]


def _ranges(tiles, count):
    sizes = torch.bincount(tiles, minlength=count)
    ends = sizes.cumsum(0)
    return torch.stack((ends-sizes, ends), -1)


def _range_entries(ranges):
    sizes = ranges[:, 1]-ranges[:, 0]
    slots = torch.repeat_interleave(torch.arange(len(ranges), device=ranges.device), sizes)
    starts = sizes.cumsum(0)-sizes
    entries = torch.arange(len(slots), device=ranges.device)-torch.repeat_interleave(starts, sizes)
    return slots, entries+ranges[slots, 0]


def _cpu_project(raw, existence, camera, degree):
    from .local_response import rotation_matrix
    from utils.sh_utils import eval_sh
    n = len(raw)
    result = raw.new_zeros(n, 12)
    if not n:
        return result
    height, width = int(camera.image_height), int(camera.image_width)
    tanx, tany = math.tan(camera.FoVx/2), math.tan(camera.FoVy/2)
    xyz1 = torch.cat((raw[:, :3], raw.new_ones(n, 1)), -1)
    view = camera.world_view_transform.to(raw)
    xyz_view = (xyz1 @ view)[:, :3]
    hom = xyz1 @ camera.full_proj_transform.to(raw)
    result[:, :2] = ((hom[:, :2]/(hom[:, 3:4]+1e-7)+1)*raw.new_tensor([width, height])-1)*.5
    result[:, 2] = xyz_view[:, 2]
    safe_z = xyz_view[:, 2].where(xyz_view[:, 2].abs() > 1e-12, 1e-12)
    tx = (xyz_view[:, 0]/safe_z).clamp(-1.3*tanx, 1.3*tanx)*safe_z
    ty = (xyz_view[:, 1]/safe_z).clamp(-1.3*tany, 1.3*tany)*safe_z
    jac = raw.new_zeros(n, 2, 3)
    fx, fy = width/(2*tanx), height/(2*tany)
    jac[:, 0, 0], jac[:, 1, 1] = fx/safe_z, fy/safe_z
    jac[:, 0, 2], jac[:, 1, 2] = -fx*tx/safe_z.square(), -fy*ty/safe_z.square()
    rot = rotation_matrix(raw[:, 7:11])
    factor = rot * raw[:, 4:7].clamp(-20, 10).exp()[:, None, :]
    cov3d = factor @ factor.transpose(-1, -2)
    transform = jac @ view[:3, :3].T
    cov = transform @ cov3d @ transform.transpose(-1, -2)
    cov = cov + .3*torch.eye(2, dtype=raw.dtype, device=raw.device)
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = a*c-b*b
    result[:, 3:6] = torch.stack((c/det, -b/det, a/det), -1)
    result[:, 6], result[:, 7] = raw[:, 3].sigmoid(), existence
    coeff = torch.cat((raw[:, 11:14, None], raw[:, 14:].reshape(n, 3, (degree+1)**2-1)), -1)
    direction = F.normalize(raw[:, :3]-camera.camera_center.to(raw), dim=-1)
    result[:, 8:11] = (eval_sh(degree, coeff, direction)+.5).clamp_min(0)
    mid = .5*(a+c)
    radius = (3*(mid+(mid.square()-det).clamp_min(.1).sqrt()).sqrt()).ceil()
    result[:, 11] = radius
    low, high = _rectangles(result, height, width)
    active = (xyz_view[:, 2] > .2) & (det != 0) & ((high-low).prod(-1) > 0)
    result[~active] = 0
    return result


def _project(raw, existence, camera, degree, backend):
    if raw.ndim != 2 or raw.shape[1] != 11+3*(degree+1)**2 or degree not in range(4):
        raise ValueError("raw must contain XYZ, opacity, scale, quaternion and SH degree 0..3")
    existence = torch.as_tensor(existence, device=raw.device, dtype=raw.dtype)
    if existence.shape != (len(raw),) or not torch.isfinite(existence).all() or not torch.isfinite(raw).all():
        raise ValueError("finite raw and one existence value per Gaussian required")
    if ((existence < 0) | (existence > 1)).any():
        raise ValueError("existence must be in [0,1]")
    if raw.dtype != torch.float32 or (backend == "cuda") != raw.is_cuda:
        raise ValueError("backend requires float32 tensors on its matching device")
    if backend == "cpu":
        return _cpu_project(raw, existence, camera, degree)
    n = len(raw)
    sh = torch.cat((raw[:, 11:14].reshape(n, 1, 3),
        raw[:, 14:].reshape(n, 3, (degree+1)**2-1).transpose(1, 2)), 1).contiguous()
    return _extension().project(raw[:, :3].contiguous(), raw[:, 3:4].sigmoid().contiguous(),
        raw[:, 4:7].clamp(-20, 10).exp().contiguous(), F.normalize(raw[:, 7:11], dim=-1).contiguous(),
        sh, existence.contiguous(), camera.world_view_transform.to(raw).contiguous(),
        camera.full_proj_transform.to(raw).contiguous(), camera.camera_center.to(raw).contiguous(),
        int(camera.image_height), int(camera.image_width), math.tan(camera.FoVx/2),
        math.tan(camera.FoVy/2), degree)


def _cpu_render(base, candidate, ranges, entries, tiles, height, width, white):
    out = base.new_zeros(len(tiles), 3, 16, 16)
    yy, xx = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    for slot, tid in enumerate(tiles.tolist()):
        ty, tx = divmod(tid, (width+15)//16)
        x, y = xx+16*tx, yy+16*ty
        valid = (x < width) & (y < height)
        trans = torch.ones(16, 16)
        done = ~valid
        color = torch.zeros(3, 16, 16)
        for idx in entries[int(ranges[slot, 0]):int(ranges[slot, 1])].tolist():
            p = base[idx] if idx >= 0 else candidate[-idx-1]
            dx, dy = p[0]-x, p[1]-y
            power = -.5*(p[3]*dx.square()+p[5]*dy.square())-p[4]*dx*dy
            alpha = (p[6]*power.exp()).clamp_max(.99)
            enabled = ~done & (power <= 0) & (alpha >= 1/255)
            next_t = trans*(1-p[7]*alpha)
            done = done | (enabled & (next_t < .0001))
            enabled = enabled & ~done
            color += p[8:11, None, None]*(p[7]*alpha*trans*enabled)[None]
            trans = torch.where(enabled, next_t, trans)
            if done.all():
                break
        out[slot] = (color+trans[None]*float(white))*valid[None]
    return out


@dataclass
class TileScene:
    image: torch.Tensor
    projected: torch.Tensor
    tile_ranges: torch.Tensor
    tile_ids: torch.Tensor
    height: int
    width: int
    tile_size: int = 16
    white_background: bool = False
    backend: str = "cuda"
    camera: object = None
    degree: int = 0

    @classmethod
    @torch.no_grad()
    def from_projected(cls, projected, height, width, white_background=False, backend="cuda"):
        if backend not in ("cpu", "cuda") or height < 1 or width < 1:
            raise ValueError("positive image dimensions and cpu/cuda backend required")
        if projected.ndim != 2 or projected.shape[1] != 12 or not torch.isfinite(projected).all():
            raise ValueError("finite [N,12] projected tensor required")
        if projected.dtype != torch.float32 or projected.is_cuda != (backend == "cuda"):
            raise ValueError("projected must be float32 on the backend device")
        p = projected.detach().clone().contiguous()
        tiles, ids = _members(p, height, width)
        order = _sort_entries(tiles, ids, p[ids, 2])
        tiles, ids = tiles[order], ids[order]
        count = ((height+15)//16)*((width+15)//16)
        scene = cls(p.new_zeros(3, height, width), p, _ranges(tiles, count), ids,
            height, width, white_background=white_background, backend=backend)
        # Chunk output allocation without dropping pixels or depth entries.
        image_tiles = p.new_empty(count, 3, 16, 16)
        for first in range(0, count, 256):
            requested = torch.arange(first, min(first+256, count), device=p.device)
            image_tiles[first:first+len(requested)] = scene._render(
                p[:0], scene.tile_ranges[requested], ids, requested)
        gy, gx = (height+15)//16, (width+15)//16
        scene.image = image_tiles.reshape(gy, gx, 3, 16, 16).permute(2, 0, 3, 1, 4).reshape(
            3, gy*16, gx*16)[:, :height, :width].contiguous()
        return scene

    @torch.no_grad()
    def project(self, raw, existence):
        if self.camera is None:
            raise ValueError("project requires a scene created by capture")
        return _project(raw, existence, self.camera, self.degree, self.backend)

    def _ids(self, ids):
        ids = torch.as_tensor(ids, device=self.projected.device, dtype=torch.long)
        if ids.ndim != 1 or len(ids.unique()) != len(ids) or ((ids < 0) | (ids >= len(self.projected))).any():
            raise ValueError("replacement IDs must be unique scene row IDs")
        return ids

    @torch.no_grad()
    def affected_tiles(self, ids, candidates, halo_tiles=0):
        ids = self._ids(ids)
        candidates = torch.as_tensor(candidates, device=self.projected.device, dtype=self.projected.dtype)
        if candidates.shape != (4, len(ids), 12) or halo_tiles < 0:
            raise ValueError("candidates must be [4,K,12] and halo_tiles nonnegative")
        p = torch.cat((self.projected[ids], candidates.reshape(-1, 12)))
        tiles = _members(p, self.height, self.width)[0].unique(sorted=True)
        if halo_tiles and len(tiles):
            gx, gy = (self.width+15)//16, (self.height+15)//16
            offsets = torch.arange(-halo_tiles, halo_tiles+1, device=p.device)
            dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
            x = tiles[:, None].remainder(gx)+dx.flatten()
            y = tiles[:, None].div(gx, rounding_mode="floor")+dy.flatten()
            good = (x >= 0) & (x < gx) & (y >= 0) & (y < gy)
            tiles = (y*gx+x)[good].unique(sorted=True)
        return tiles

    def _render(self, candidate, ranges, entries, tiles):
        if self.backend == "cpu":
            return _cpu_render(self.projected, candidate, ranges, entries, tiles,
                self.height, self.width, self.white_background)
        return _extension().render(self.projected, candidate.contiguous(), ranges.contiguous(),
            entries.contiguous(), tiles.contiguous(), self.height, self.width, self.white_background)

    @torch.no_grad()
    def render_tiles(self, ids, projected, tile_ids):
        ids = self._ids(ids)
        p = torch.as_tensor(projected, device=self.projected.device, dtype=self.projected.dtype)
        tiles = torch.as_tensor(tile_ids, device=self.projected.device, dtype=torch.long)
        if p.shape != (len(ids), 12) or not torch.isfinite(p).all():
            raise ValueError("replacement projected must be finite [K,12]")
        if tiles.ndim != 1 or len(tiles.unique()) != len(tiles) or ((tiles < 0) | (tiles >= len(self.tile_ranges))).any():
            raise ValueError("tile_ids must be unique valid tile IDs")
        slot, pos = _range_entries(self.tile_ranges[tiles])
        old = self.tile_ids[pos]
        keep = ~torch.isin(old, ids)
        slot, old = slot[keep], old[keep]
        candidate_tiles, candidate_rows = _members(p, self.height, self.width)
        sorted_tiles, permutation = tiles.sort()
        at = torch.searchsorted(sorted_tiles, candidate_tiles)
        valid = at < len(tiles)
        if len(tiles):
            valid &= sorted_tiles[at.clamp_max(len(tiles)-1)] == candidate_tiles
        candidate_rows, at = candidate_rows[valid], at[valid]
        slots = torch.cat((slot, permutation[at]))
        original_ids = torch.cat((old, ids[candidate_rows]))
        depths = torch.cat((self.projected[old, 2], p[candidate_rows, 2]))
        entries = torch.cat((old, -candidate_rows-1))
        order = _sort_entries(slots, original_ids, depths)
        return self._render(p, _ranges(slots[order], len(tiles)), entries[order], tiles)


@torch.no_grad()
def capture(raw, camera, degree, existence, white_background=False, backend="cuda"):
    p = _project(raw, existence, camera, degree, backend)
    scene = TileScene.from_projected(p, int(camera.image_height), int(camera.image_width),
        white_background=white_background, backend=backend)
    # Copy camera tensors so later camera/raw mutations cannot stale the cache.
    from types import SimpleNamespace
    scene.camera = SimpleNamespace(image_height=scene.height, image_width=scene.width,
        FoVx=float(camera.FoVx), FoVy=float(camera.FoVy),
        world_view_transform=camera.world_view_transform.detach().to(raw).clone(),
        full_proj_transform=camera.full_proj_transform.detach().to(raw).clone(),
        camera_center=camera.camera_center.detach().to(raw).clone())
    scene.degree = degree
    return scene
