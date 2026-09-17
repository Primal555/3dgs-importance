"""Self-contained simulated receiver packet; no source PLY/clean features at decode.

metadata.bin is a CRC-protected zlib-compressed metadata record. It is assumed
reliably delivered, as in NTSCC; CRC detects file corruption, not channel FEC.
received.npy stores channel outputs for simulation, NOT a digital wire bitrate.
Per-Gaussian coordinates are carried by the JSCC payload, not this metadata.
"""

import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

import numpy as np
import torch

from .codec import CodecConfig, GaussianCodec, channel
from .data import Geometry, prepare, to_features, to_raw


def model_id(model):
    digest = hashlib.sha256()
    digest.update(json.dumps(model.cfg.to_dict(), sort_keys=True).encode())
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path, model, step, training=None):
    torch.save({"version": 4 if model.cfg.architecture == 'learned_joint' else 3 if model.cfg.architecture == "geometry_first" else 2,
                "config": model.cfg.to_dict(), "state_dict": model.state_dict(),
                "step": step, "training": training or {}}, path)


def load_checkpoint(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") not in (2, 3, 4):
        raise ValueError("unsupported codec checkpoint version")
    cfg = CodecConfig.from_dict(saved["config"])
    if saved['version'] != {'legacy': 2, 'geometry_first': 3, 'learned_joint': 4}[cfg.architecture]:
        raise ValueError("checkpoint version/architecture mismatch")
    model = GaussianCodec(cfg)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.position_head_needs_initialization = False
    return model.to(device).eval()


def pack_tiers(q):
    q = np.asarray(q, dtype=np.uint8)
    out = np.zeros((len(q) + 3) // 4, dtype=np.uint8)
    for shift in range(4):
        part = q[shift::4]
        out[:len(part)] |= part << (2 * shift)
    return out.tobytes()


def unpack_tiers(data, n):
    packed = np.frombuffer(data, dtype=np.uint8)
    q = np.empty(n, dtype=np.int64)
    for shift in range(4):
        part = q[shift::4]
        part[:] = (packed[:len(part)] >> (2 * shift)) & 3
    return q


def encode_metadata(header, q):
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    body = struct.pack("<I", len(text)) + text
    # Complete q sequence frames the variable-length stream and preserves q=0.
    # Packing four 2-bit decisions per byte avoids a wasteful uint8 array.
    body += pack_tiers(q)
    compressed = zlib.compress(body, level=9)
    return b"GJS2" + struct.pack("<I", zlib.crc32(compressed)) + compressed


def decode_metadata(data):
    if len(data) < 8 or data[:4] != b"GJS2":
        raise ValueError("invalid metadata magic")
    if zlib.crc32(data[8:]) != struct.unpack("<I", data[4:8])[0]:
        raise ValueError("metadata CRC failure")
    body = zlib.decompress(data[8:])
    size = struct.unpack("<I", body[:4])[0]
    header = json.loads(body[4:4 + size])
    n = header["source_count"]
    if n < 0 or len(body) != 4 + size + (n + 3) // 4:
        raise ValueError("invalid metadata lengths")
    offset = 4 + size
    q = unpack_tiers(body[offset:], n)
    if int((q > 0).sum()) != header["count"]:
        raise ValueError("invalid retained tier count")
    return header, torch.from_numpy(q)


def metadata_channel_uses(bits, snr, code_rate=None, modulation_bits=2):
    if code_rate is None:
        # Ideal complex AWGN capacity. An accounting assumption, not a tested code.
        return math.ceil(bits / math.log2(1 + 10 ** (float(snr) / 10)))
    if not 0 < code_rate <= 1 or modulation_bits < 1:
        raise ValueError("invalid metadata code rate/modulation")
    return math.ceil(bits / (code_rate * modulation_bits))


@torch.no_grad()
def transmit(model, raw, q, snr, kind, seed, output, code_rate=None, modulation_bits=2):
    from .codec import validate_tiers
    validate_tiers(q, len(raw))
    if kind not in ("none", "awgn", "rayleigh") or not math.isfinite(float(snr)):
        raise ValueError("invalid channel/SNR")
    # Validate accounting before producing any packet files.
    metadata_channel_uses(8, snr, code_rate, modulation_bits)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.eval()
    device = next(model.parameters()).device
    raw, geometry, q = prepare(raw, model.cfg.morton_bits, q)
    received = []
    transmitted_energy = 0.
    generator = torch.Generator(device=device).manual_seed(seed)
    for start in range(0, len(raw), model.cfg.block_size):
        end = start + model.cfg.block_size
        keep = q[start:end] > 0
        if not keep.any():
            continue
        qb, rb = ((q[start:end],raw[start:end]) if model.cfg.individual_tiers else
                  (q[start:end][keep], raw[start:end][keep]))
        features, unit = to_features(rb.to(device), geometry, model)
        z = model.encode(features, unit, qb.to(device), snr)
        transmitted_energy += float(z.square().sum())
        received.append(channel(z, snr, kind, generator).cpu())
    received = torch.cat(received) if received else torch.empty((0, 2))
    header = {"version": 2, "config": model.cfg.to_dict(), "model_id": model_id(model),
              "count": int((q > 0).sum()), "source_count": len(raw),
              "geometry": geometry.to_dict(), "snr_db": float(snr), "channel": kind}
    metadata = encode_metadata(header, q.numpy())
    (output / "metadata.bin").write_bytes(metadata)
    np.save(output / "received.npy", received.numpy(), allow_pickle=False)
    bits = len(metadata) * 8
    meta_uses = metadata_channel_uses(bits, snr, code_rate, modulation_bits)
    stats = {"source_gaussians": len(raw), "retained_gaussians": int((q > 0).sum()),
             "tier_counts": torch.bincount(q, minlength=4).tolist(),
             "payload_complex_symbols": len(received), "metadata_bytes": len(metadata),
             "transmitted_mean_complex_energy": transmitted_energy / max(1,len(received)),
             "tier_map_uncompressed_bytes": (len(q) + 3) // 4,
             "per_gaussian_coordinates_in_metadata": False,
             "global_geometry_floats": 6,
             "metadata_channel_uses": meta_uses, "total_channel_uses": len(received) + meta_uses,
             "total_uses_per_source_gaussian": (len(received) + meta_uses) / len(raw),
             "snr_db": snr, "channel": kind, "seed": seed,
             "metadata_assumption": "reliably delivered; no header channel errors simulated",
             "metadata_cost_model": "ideal_complex_AWGN_capacity" if code_rate is None else "specified_code_modulation",
             "metadata_code_rate": code_rate, "metadata_modulation_bits": modulation_bits,
             "rayleigh_csi": "perfect receiver CSI, per-symbol MMSE" if kind == "rayleigh" else None,
             "model_weights": "shared in advance; excluded from channel uses",
             "shared_model_tensor_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
             "disk_payload_bytes": (output / "received.npy").stat().st_size}
    stats["architecture"] = model.cfg.architecture
    if model.cfg.position_head == 'block_relative_v4':
        stats.update(position_coding='block_relative_v4',
                     block_reference_in_metadata=False,
                     geometry_reference_real_slots_per_retained_gaussian=4,
                     geometry_energy_completion_real_slots_per_retained_gaussian=1,
                     power_normalization='geometry constant energy per row; attributes unit mean energy per block')
    elif model.cfg.position_head == 'block_pilot_v5':
        stats.update(position_coding='block_pilot_v5',
                     geometry_group_size=model.cfg.geometry_group_size,
                     block_reference_in_metadata=False,
                     geometry_reference_real_slots_per_retained_gaussian=4,
                     geometry_pilot_real_slots_per_retained_gaussian=1,
                     geometry_energy_completion_real_slots_per_retained_gaussian=0,
                     power_normalization='geometry unit mean energy per deterministic retained-row group; '
                                         'attributes unit mean energy per block')
    elif model.cfg.position_head == 'reference_v6':
        stats.update(position_coding='reference_v6',geometry_group_size=model.cfg.geometry_group_size,
                     block_reference_in_metadata=False,geometry_reference_real_slots_per_retained_gaussian=4,
                     geometry_local_pilot_real_slots_per_retained_gaussian=1,
                     reference_frequencies=[1,4,16],small_group_fallback='v5 when retained group count < 24',
                     power_normalization='fixed-energy phase reference plus local detail normalization; group mean unit energy')
        if model.cfg.individual_tiers:
            stats.update(grouping='fixed source-row intervals before q0 removal',
                         reference_energy='same per retained Gaussian, independent of positive tier',
                         geometry_local_pilot_real_slots_per_retained_gaussian=1,
                         geometry_energy_completion_real_slots_per_tier=[0,0,1,2],
                         sparse_geometry_completion_real_slots_per_tier=[0,1,2,3],
                         local_pilot_scope='dense groups only; sparse base uses known gain and energy completion',
                         small_group_fallback='tier-independent shared analog base plus personal enhancement for fewer than 24 retained rows',
                         power_normalization='tier-independent base group normalization plus per-row enhancement energy; attribute per-row tanh/RMS with energy floor .1, mean energy <= 1')
    stats["position_seed_is_final"] = model.cfg.architecture in ("geometry_first", "learned_joint")
    if model.cfg.architecture == 'learned_joint':
        stats.update(position_coding='learned_joint', geometry_sub_budget=None,
                     block_reference_in_metadata=False, handcrafted_coordinate_symbols=0,
                     receiver_context='local received features; no source or predicted XYZ inputs',
                     grouping='fixed source-row intervals; q0 holes retained in syntax',
                     power_normalization='smooth per-row RMS; mean complex energy <= 1')
    if model.cfg.architecture == "geometry_first":
        geometry_symbols = int(torch.as_tensor(model.cfg.geometry_rates)[q].sum())
        stats.update(geometry_complex_symbols=geometry_symbols,
                     attribute_complex_symbols=len(received) - geometry_symbols,
                     geometry_rates=list(model.cfg.geometry_rates))
    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


@torch.no_grad()
def receive(model, packet, return_position_seed=False):
    """Decode a packet, optionally exposing the decoder bootstrap XYZ.

    The optional seed is a diagnostic derived from the same received symbols;
    it is neither added to the packet nor used by normal deployment decoding.
    """
    packet = Path(packet)
    header, q = decode_metadata((packet / "metadata.bin").read_bytes())
    if header["version"] != 2 or header["model_id"] != model_id(model):
        raise ValueError("packet and shared codec checkpoint do not match")
    if CodecConfig.from_dict(header["config"]) != model.cfg:
        raise ValueError("codec configuration mismatch")
    z = np.load(packet / "received.npy", mmap_mode="r", allow_pickle=False)
    expected = int(torch.as_tensor(model.cfg.rates)[q].sum())
    if z.shape != (expected, 2) or z.dtype != np.float32 or not np.isfinite(z).all():
        raise ValueError("invalid received symbols")
    geometry = Geometry(**header["geometry"])
    if geometry.bits != model.cfg.morton_bits:
        raise ValueError("geometry/checkpoint Morton quantization mismatch")
    device = next(model.parameters()).device
    rows, position_seeds = [], []
    i = j = 0
    model.eval()
    for start in range(0, len(q), model.cfg.block_size):
        qb = q[start:start + model.cfg.block_size]
        qb = (qb if model.cfg.individual_tiers else qb[qb > 0]).to(device)
        count = int((qb>0).sum())
        if not count:
            continue
        length = int(torch.as_tensor(model.cfg.rates, device=device)[qb].sum())
        symbols = torch.from_numpy(z[j:j + length].copy()).to(device)
        decoded = model.decode(symbols, qb, header["snr_db"],
                               return_seed=return_position_seed)
        if return_position_seed:
            pred, seed = decoded
            if model.cfg.individual_tiers:
                pred,seed=pred[qb>0],seed[qb>0]
            position_seeds.append(geometry.denormalize(seed).cpu())
        else:
            pred = decoded
            if model.cfg.individual_tiers:
                pred=pred[qb>0]
        rows.append(to_raw(pred, geometry, model).cpu())
        i += count
        j += length
    recovered = torch.cat(rows) if rows else torch.empty((0, 3 + model.cfg.attr_dim))
    if return_position_seed:
        seeds = torch.cat(position_seeds) if position_seeds else torch.empty((0, 3))
        return recovered, seeds
    return recovered
