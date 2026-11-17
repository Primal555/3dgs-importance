"""Eight-tier packet boundary and actual three-file digital side accounting."""
from pathlib import Path
import json
import struct

import numpy as np
import torch

from comparison_baselines.side_information_accounting import side_information_cost
from .codec import unpack, channel
from .data import Geometry, to_raw
from .position_delivery import encode_positions, decode_positions
from .successive_delivery_packet import batch_layout, pad_rows
from .transport import encode_metadata, decode_metadata

PROTOCOL = 'progressive_attribute_eight_v1'


def present_scales(q):
    return torch.stack([(q >= j).any(1) for j in range(1, int(q.max())+1)], 1)


def scale_frame(scales, q):
    present = present_scales(q)
    if scales.shape != present.shape:
        raise ValueError('scale shape mismatch')
    values = scales[present].detach().cpu()
    if (not torch.isfinite(values).all() or (values <= 0).any() or
            not torch.equal(values, values.half().float())):
        raise ValueError('binary16 positive scales required')
    return b'SCE8'+struct.pack('<I', len(values))+values.numpy().astype('<f2').tobytes()


def read_scale_frame(data, q):
    present = present_scales(q)
    count = int(present.sum())
    if len(data) != 8+2*count or data[:4] != b'SCE8' or struct.unpack('<I', data[4:8])[0] != count:
        raise ValueError('scale framing mismatch')
    values = torch.from_numpy(np.frombuffer(data[8:], dtype='<f2').astype(np.float32).copy()).to(q.device)
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError('invalid scale values')
    result = torch.ones(present.shape, device=q.device)
    result[present] = values
    return result


@torch.no_grad()
def export_packet(model, scene, cached, layouts, snr, seed, out, *, noiseless=False):
    """Layouts and cached rows use scene Morton order, including original padding."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    device = next(model.parameters()).device
    model.eval(); model.set_scene_geometry(scene.geometry)
    if len(cached) != len(layouts) or len(cached) != len(scene.groups):
        raise ValueError('packet groups differ from original semantic blocks')
    qs = torch.cat([q[ids >= 0].cpu() for q, ids in zip(layouts, scene.group_ids)])
    units = torch.cat([f[..., :3][ids >= 0] for f, ids in zip(cached, scene.group_ids)]).to(device)
    header = dict(packet_protocol=PROTOCOL, model_id=model.identity(),
        config=dict(rates=list(model.layer_config.rates)),
        semantic_config=model.cfg.to_dict(), channel_config=model.layer_config.to_dict(),
        arm=model.semantic.arm, count=int((qs > 0).sum()), source_count=len(qs),
        geometry=scene.geometry.to_dict(), snr_db=float(snr),
        neighbor=bool(model.neighbor_enabled), block_size=model.cfg.block_size,
        blocks_per_batch=cached[0].shape[0])
    (out/'metadata.bin').write_bytes(encode_metadata(header, qs.numpy()))
    (out/'xyz.bin').write_bytes(encode_positions(units, qs.to(device), model.cfg))
    generator = torch.Generator(device=device).manual_seed(seed)
    frames, payload = [], []
    for data, q in zip(cached, layouts):
        data, q = data.to(device), q.to(device)
        z, gains = model.encode(data[..., 59:], q, snr)
        noise = channel(z.new_zeros(*q.shape, 64, 2), snr,
                        'none' if noiseless else 'awgn', generator).flatten(-2)
        r = z + noise[..., :z.shape[-1]]
        mask = torch.arange(r.shape[-1], device=device) < q[..., None]*16
        payload.append(r[mask].reshape(-1, 2).cpu())
        frames.append(scale_frame(gains, q))
    symbols = torch.cat(payload)
    np.save(out/'received.npy', symbols.numpy(), allow_pickle=False)
    (out/'scales.bin').write_bytes(b''.join(frames))
    measured = dict(payload_complex_symbols=len(symbols), xyz_bytes=(out/'xyz.bin').stat().st_size,
        metadata_bytes=(out/'metadata.bin').stat().st_size, scale_stream_bytes=(out/'scales.bin').stat().st_size)
    return measured, side_information_cost(measured)


@torch.no_grad()
def receive_packet(model, packet):
    """The receiver accepts only shared weights and the received packet files."""
    packet = Path(packet)
    h, q = decode_metadata((packet/'metadata.bin').read_bytes())
    expected = json.loads(json.dumps(model.layer_config.to_dict()))
    if (h.get('packet_protocol') != PROTOCOL or h['model_id'] != model.identity() or
            h['channel_config'] != expected or h['semantic_config'] != json.loads(json.dumps(model.cfg.to_dict())) or
            h['arm'] != model.semantic.arm or h['neighbor'] != bool(model.neighbor_enabled) or
            h['block_size'] != model.cfg.block_size):
        raise ValueError('receiver model/packet identity mismatch')
    device = next(model.parameters()).device
    geometry = Geometry(**h['geometry'])
    model.eval(); model.set_scene_geometry(geometry)
    side = decode_positions((packet/'xyz.bin').read_bytes(), q.to(device), model.cfg)
    symbols = np.load(packet/'received.npy', mmap_mode='r', allow_pickle=False)
    total = int(torch.tensor(model.layer_config.rates)[q].sum())
    if symbols.dtype != np.float32 or symbols.shape != (total, 2) or not np.isfinite(symbols).all():
        raise ValueError('invalid analogue payload')
    scales = (packet/'scales.bin').read_bytes()
    rows = []; offset = symbol_offset = scale_offset = 0
    for lengths in batch_layout(len(q), model.cfg.block_size, h['blocks_per_batch']):
        n = sum(lengths)
        tiers = pad_rows(q[offset:offset+n], lengths).to(device)
        xyz = pad_rows(side[offset:offset+n], lengths)
        count = int(tiers.sum())*8
        length = 8+2*int(present_scales(tiers).sum())
        gains = read_scale_frame(scales[scale_offset:scale_offset+length], tiers)
        z = torch.from_numpy(np.array(symbols[symbol_offset:symbol_offset+count], copy=True)).to(device)
        signal = unpack(z, tiers.flatten(), model.layer_config.rates).reshape(*tiers.shape, 128)
        y, _ = model.receive(signal[..., :16*int(tiers.max())], gains, tiers, xyz, h['snr_db'])
        pred = model.semantic.decode(y, tiers > 0, xyz)
        rows.append(to_raw(pred[tiers > 0], geometry, model).cpu())
        offset += n; symbol_offset += count; scale_offset += length
    if symbol_offset != total or scale_offset != len(scales):
        raise ValueError('trailing or missing packet bytes')
    return torch.cat(rows)
