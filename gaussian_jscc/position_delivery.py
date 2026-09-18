"""Experimental reliable XYZ side stream, not a noise-free JSCC claim.

Coordinates are normalized with the existing transmitted global bbox. Only
q>0 rows are sent, in packet order. No entropy coding, FEC or packet loss is
implemented. Training simulates exactly this quantizer; packet export measures
actual bytes, including framing/CRC. Float32 is a costly diagnostic control,
not bit-exact original-world-coordinate recovery after normalization.
"""
import math
import struct
import zlib
import numpy as np
import torch


def delivered_positions(unit, q, cfg):
    if cfg.position_delivery == 'learned':
        return None
    value = unit.detach().to(torch.float32)
    if cfg.position_delivery == 'quantized':
        levels = 2**cfg.position_bits - 1
        value = (value.clamp(0, 1) * levels).round() / levels
    return value * (q > 0)[..., None]


def position_cost(cfg, retained):
    bits = 0 if cfg.position_delivery == 'learned' else (32 if cfg.position_delivery == 'float32' else cfg.position_bits)
    content = int(retained) * 3 * bits
    # Header: magic(4), mode(1), precision(1), count(8), CRC(4).
    size = 0 if not bits else 18 + (content + 7)//8
    return {'position_delivery': cfg.position_delivery, 'position_bits_per_axis': bits,
            'position_content_bits': content, 'position_stream_bytes': size,
            'position_stream_bits': size*8}


def encode_positions(unit, q, cfg):
    if cfg.position_delivery == 'learned':
        raise ValueError('learned mode has no XYZ side stream')
    xyz = delivered_positions(unit, q, cfg)[q > 0].cpu().numpy()
    if not np.isfinite(xyz).all() or ((xyz < 0) | (xyz > 1)).any():
        raise ValueError('side XYZ must be finite normalized coordinates')
    if cfg.position_delivery == 'float32':
        mode, bits = 1, 32
        payload = xyz.astype('<f4').tobytes()
    else:
        mode, bits = 2, cfg.position_bits
        integers = np.rint(xyz.reshape(-1) * (2**bits-1)).astype(np.uint32)
        binary = ((integers[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
        payload = np.packbits(binary.reshape(-1), bitorder='little').tobytes()
    body = struct.pack('<BBQ', mode, bits, len(xyz)) + payload
    return b'GXYZ' + struct.pack('<I', zlib.crc32(body)) + body


def decode_positions(data, q, cfg):
    if len(data) < 18 or data[:4] != b'GXYZ':
        raise ValueError('invalid XYZ stream header')
    if zlib.crc32(data[8:]) != struct.unpack('<I', data[4:8])[0]:
        raise ValueError('XYZ stream CRC failure')
    mode, bits, count = struct.unpack('<BBQ', data[8:18])
    expected = (1, 32) if cfg.position_delivery == 'float32' else (2, cfg.position_bits)
    if cfg.position_delivery == 'learned' or (mode, bits) != expected or count != int((q > 0).sum()):
        raise ValueError('XYZ stream configuration/count mismatch')
    if len(data) != position_cost(cfg, count)['position_stream_bytes']:
        raise ValueError('XYZ stream length mismatch')
    if mode == 1:
        xyz = np.frombuffer(data[18:], dtype='<f4').copy().reshape(-1, 3)
    else:
        binary = np.unpackbits(np.frombuffer(data[18:], dtype=np.uint8), bitorder='little')
        valid = count*3*bits
        if binary[valid:].any():
            raise ValueError('nonzero XYZ padding bits')
        integers = (binary[:valid].reshape(-1, bits).astype(np.uint32) << np.arange(bits, dtype=np.uint32)).sum(1)
        xyz = (integers.astype(np.float32)/(2**bits-1)).reshape(-1, 3)
    if not np.isfinite(xyz).all() or ((xyz < 0) | (xyz > 1)).any():
        raise ValueError('invalid decoded XYZ')
    result = torch.zeros((*q.shape, 3), dtype=torch.float32, device=q.device)
    result[q > 0] = torch.from_numpy(xyz).to(q.device)
    return result


def training_position_cost(cfg, retained, source_count, payload, bits_per_use):
    if not math.isfinite(bits_per_use) or bits_per_use <= 0:
        raise ValueError('position net bits/use must be positive and finite')
    result = position_cost(cfg, retained)
    uses = math.ceil(result['position_stream_bits']/bits_per_use)
    result.update(position_assumed_net_bits_per_use=bits_per_use,
                  position_channel_uses_estimate=uses,
                  position_uses_per_source_gaussian=uses/max(1, source_count),
                  payload_plus_position_uses_per_source_gaussian=(payload+uses)/max(1, source_count),
                  position_reliability='assumed reliable; no FEC/packet errors simulated',
                  rate_scope='payload + XYZ framing; excludes existing bbox/tier/model metadata')
    return result
