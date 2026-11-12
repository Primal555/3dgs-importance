"""Byte-framed successive receiver; sender source tensors never enter decode."""
from pathlib import Path
import json
import numpy as np
import torch
from .codec import channel, pack, unpack
from .data import Geometry, to_raw
from .position_delivery import encode_positions, decode_positions, delivered_positions
from .successive_channel import present_scales
from .transport import model_id, encode_metadata, decode_metadata

SIDE_FILES = ('xyz.bin', 'metadata.bin', 'scales.bin')


def batch_layout(count, block_size, blocks_per_batch):
    if min(count, block_size, blocks_per_batch) < 1: raise ValueError('positive packet dimensions required')
    lengths = [min(block_size, count-i) for i in range(0, count, block_size)]
    for i in range(0, len(lengths), blocks_per_batch): yield lengths[i:i+blocks_per_batch]


def pad_rows(rows, lengths):
    result = rows.new_zeros((len(lengths), max(lengths), *rows.shape[1:]))
    offset = 0
    for i, n in enumerate(lengths): result[i, :n] = rows[offset:offset+n]; offset += n
    return result


@torch.no_grad()
def export_packet(model, scene, original_q, snr, seed, out, *, noiseless=False):
    """Preserve original batched codec context and fixed full-slot AWGN sampling."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    if model.cfg.channel_variant != 'successive' or model.cfg.position_delivery != 'quantized':
        raise ValueError('successive codec with quantized XYZ required')
    q = original_q.detach().cpu().long()
    if q.shape != (len(scene.raw),) or ((q < 0) | (q > 3)).any(): raise ValueError('invalid original-row tiers')
    packet_q = q[scene.order]
    blocks_per_batch = scene.groups[0].shape[0]
    layout = list(batch_layout(len(q), model.cfg.block_size, blocks_per_batch))
    if len(layout) != len(scene.groups): raise ValueError('packet batch layout differs from scene')
    model.eval(); model.set_scene_geometry(scene.geometry)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    header = dict(version=2, config=model.cfg.to_dict(), model_id=model_id(model),
        count=int((q>0).sum()), source_count=len(q), geometry=scene.geometry.to_dict(),
        snr_db=float(snr), channel='none' if noiseless else 'awgn',
        tx_condition_snr_db=model.resolve_condition_snr(snr), rx_condition_snr_db=model.resolve_condition_snr(snr),
        packet_protocol='successive_side_files_v1', blocks_per_batch=blocks_per_batch)
    (out/'metadata.bin').write_bytes(encode_metadata(header, packet_q.numpy()))
    # Reuse the exact normalized source coordinates used by the codec, including
    # device arithmetic performed when the scene feature blocks were prepared.
    units = torch.cat([f[..., :3][ids>=0] for f,ids in zip(scene.groups,scene.group_ids)])
    units = units.to(device); wire_q = packet_q.to(device)
    xyz_frame = encode_positions(units, wire_q, model.cfg)
    (out/'xyz.bin').write_bytes(xyz_frame)
    torch.testing.assert_close(decode_positions(xyz_frame, wire_q, model.cfg),
        delivered_positions(units, wire_q, model.cfg), atol=0, rtol=0,
        msg='Packet XYZ differs from the original device quantizer')
    del units, wire_q
    payload = []; frames = []
    for features, ids, lengths in zip(scene.groups, scene.group_ids, layout):
        if features.shape[:2] != (len(lengths), max(lengths)): raise ValueError('packet batch padding differs')
        f = features.to(device); tiers = torch.where(ids>=0, q[ids.clamp_min(0)], 0).to(device)
        y = model.encode_representation(f, f[..., :3], tiers>0)
        encoded = model.encode_channel_packet(y, tiers, header['tx_condition_snr_db'])
        dense = unpack(encoded['payload'], tiers.flatten(), model.cfg.rates)
        noisy = channel(dense.reshape(-1, 2), snr, 'none' if noiseless else 'awgn', generator).reshape_as(dense)
        payload.append(pack(noisy, tiers.flatten(), model.cfg.rates).cpu())
        frames.append(encoded['scales'])
    symbols = torch.cat(payload)
    np.save(out/'received.npy', symbols.numpy(), allow_pickle=False)
    (out/'scales.bin').write_bytes(b''.join(frames))
    return dict(payload_complex_symbols=len(symbols), xyz_bytes=(out/'xyz.bin').stat().st_size,
                metadata_bytes=(out/'metadata.bin').stat().st_size, scale_stream_bytes=(out/'scales.bin').stat().st_size)


@torch.no_grad()
def receive_packet(model, packet):
    """Only shared model and received packet paths are available at this boundary."""
    packet = Path(packet); header, q = decode_metadata((packet/'metadata.bin').read_bytes())
    if (header.get('packet_protocol') != 'successive_side_files_v1' or
            header['config'] != json.loads(json.dumps(model.cfg.to_dict())) or header['model_id'] != model_id(model)):
        raise ValueError('received codec identity/protocol mismatch')
    geometry = Geometry(**header['geometry']); model.eval(); model.set_scene_geometry(geometry)
    device = next(model.parameters()).device
    xyz = decode_positions((packet/'xyz.bin').read_bytes(), q.to(device), model.cfg)
    symbols = np.load(packet/'received.npy', mmap_mode='r', allow_pickle=False)
    expected = int(torch.tensor(model.cfg.rates)[q].sum())
    if symbols.dtype != np.float32 or symbols.shape != (expected, 2) or not np.isfinite(symbols).all():
        raise ValueError('invalid received analogue symbol layout')
    scales = (packet/'scales.bin').read_bytes()
    device = next(model.parameters()).device; rows = []; offset = symbol_offset = scale_offset = 0
    for lengths in batch_layout(len(q), model.cfg.block_size, header['blocks_per_batch']):
        n = sum(lengths); tiers = pad_rows(q[offset:offset+n], lengths).to(device)
        side = pad_rows(xyz[offset:offset+n], lengths).to(device)
        count = int(torch.tensor(model.cfg.rates, device=device)[tiers].sum())
        frame_length = 8+2*int(present_scales(tiers).sum())
        frame = scales[scale_offset:scale_offset+frame_length]
        z = torch.from_numpy(np.array(symbols[symbol_offset:symbol_offset+count], copy=True)).to(device)
        y_hat = model.decode_channel_packet(z, tiers, header['rx_condition_snr_db'], frame, delivered_xyz=side)
        pred = model.decode_representation(y_hat, tiers, side)
        rows.append(to_raw(pred[tiers>0], geometry, model))
        offset += n; symbol_offset += count; scale_offset += frame_length
    if symbol_offset != expected or scale_offset != len(scales): raise ValueError('received packet framing mismatch')
    return torch.cat(rows), q
