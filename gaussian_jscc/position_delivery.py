"""Experimental reliable XYZ side stream, not a noise-free JSCC claim.

Coordinates are normalized with the existing transmitted global bbox. Only
q>0 rows are sent, in packet order. Optional delta/byte-plane/zlib coding is
lossless over quantized integers. No FEC or packet loss is implemented.
Training simulates exactly this quantizer; packet export measures
actual bytes, including framing/CRC. Float32 is a costly diagnostic control,
not bit-exact original-world-coordinate recovery after normalization.
"""
import math
import struct
import zlib
import time
import numpy as np
import torch
from collections import OrderedDict


def delivered_positions(unit, q, cfg):
    if cfg.position_delivery == 'learned':
        return None
    value = unit.detach().to(torch.float32)
    if cfg.position_delivery == 'quantized':
        levels = 2**cfg.position_bits - 1
        value = (value.clamp(0, 1) * levels).round() / levels
    return value * (q > 0)[..., None]


def position_cost(cfg, retained, stream_bytes=None, *, allow_unmeasured=False):
    bits = 0 if cfg.position_delivery == 'learned' else (32 if cfg.position_delivery == 'float32' else cfg.position_bits)
    content = int(retained) * 3 * bits
    # Header: magic(4), mode(1), precision(1), count(8), CRC(4).
    size = 0 if not bits else 18 + (content + 7)//8
    packed_size = size
    coding = getattr(cfg, 'position_compression', 'none')
    if coding != 'none':
        if stream_bytes is None and not allow_unmeasured:
            raise ValueError('compressed coordinate cost requires measured stream_bytes')
        if stream_bytes is not None and (not isinstance(stream_bytes, int) or stream_bytes < 18):
            raise ValueError('invalid measured coordinate stream length')
        size = stream_bytes
    return {'position_delivery': cfg.position_delivery, 'position_bits_per_axis': bits,
            'position_compression': coding, 'position_packed_stream_bytes': packed_size,
            'position_content_bits': content, 'position_stream_bytes': size,
            'position_stream_bits': None if size is None else size*8,
            'position_compression_level': getattr(cfg, 'position_compression_level', 9) if coding != 'none' else None,
            'position_cost_status': 'not_measured' if size is None else ('measured' if coding != 'none' else 'exact_packed')}


def _frame_positions(mode, bits, count, payload):
    body = struct.pack('<BBQ', mode, bits, count) + payload
    return b'GXYZ' + struct.pack('<I', zlib.crc32(body)) + body


def _encode_quantized_integers(integers, cfg):
    """Shared export/meter path. Integer coordinates are exact cached inputs.

    Selection must happen BEFORE differencing: deleting a row changes its
    successor's delta. Compression level changes bytes, never decoded XYZ.
    """
    bits = cfg.position_bits
    if getattr(cfg, 'position_compression', 'none') == 'delta_zlib':
        mode = 3
        delta = np.diff(integers, axis=0, prepend=np.zeros((1, 3), np.int32)).astype('<i4')
        planes = delta.T.copy().view(np.uint8).reshape(3, len(integers), 4).transpose(0, 2, 1)
        payload = zlib.compress(planes.tobytes(), level=getattr(cfg, 'position_compression_level', 9))
    else:
        mode = 2
        binary = ((integers.reshape(-1, 1).astype(np.uint32) >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8)
        payload = np.packbits(binary.reshape(-1), bitorder='little').tobytes()
    return _frame_positions(mode, bits, len(integers), payload)


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
        bits = cfg.position_bits
        integers = np.rint(xyz * (2**bits-1)).astype(np.int32)
        return _encode_quantized_integers(integers, cfg)
    return _frame_positions(mode, bits, len(xyz), payload)


def decode_positions(data, q, cfg):
    if len(data) < 18 or data[:4] != b'GXYZ':
        raise ValueError('invalid XYZ stream header')
    if zlib.crc32(data[8:]) != struct.unpack('<I', data[4:8])[0]:
        raise ValueError('XYZ stream CRC failure')
    mode, bits, count = struct.unpack('<BBQ', data[8:18])
    expected = (1, 32) if cfg.position_delivery == 'float32' else (2, cfg.position_bits)
    if getattr(cfg, 'position_compression', 'none') == 'delta_zlib':
        expected = (3, cfg.position_bits)
    if cfg.position_delivery == 'learned' or (mode, bits) != expected or count != int((q > 0).sum()):
        raise ValueError('XYZ stream configuration/count mismatch')
    if mode != 3 and len(data) != position_cost(cfg, count)['position_stream_bytes']:
        raise ValueError('XYZ stream length mismatch')
    if mode == 3:
        # Bound decompression by the verified q>0 count. Reject trailing streams,
        # truncated data and oversized expansion even if an attacker repairs CRC.
        decoder = zlib.decompressobj()
        try:
            planes = decoder.decompress(data[18:], count*12+1)
        except zlib.error as exc:
            raise ValueError('invalid compressed XYZ payload') from exc
        if len(planes) != count*12 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('compressed XYZ length/trailing data mismatch')
        delta = np.frombuffer(planes, np.uint8).reshape(3, 4, count).transpose(0, 2, 1).copy()
        integers = delta.view('<i4').reshape(3, count).T.cumsum(axis=0, dtype=np.int64)
        if ((integers < 0) | (integers > 2**bits-1)).any():
            raise ValueError('decoded XYZ integer outside quantization range')
        xyz = integers.astype(np.float32)/(2**bits-1)
    elif mode == 1:
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


def training_position_cost(cfg, retained, source_count, payload, bits_per_use, stream_bytes=None, *, allow_unmeasured=False):
    if not math.isfinite(bits_per_use) or bits_per_use <= 0:
        raise ValueError('position net bits/use must be positive and finite')
    result = position_cost(cfg, retained, stream_bytes, allow_unmeasured=allow_unmeasured)
    uses = None if result['position_stream_bits'] is None else math.ceil(result['position_stream_bits']/bits_per_use)
    result.update(position_assumed_net_bits_per_use=bits_per_use,
                  position_channel_uses_estimate=uses,
                  position_uses_per_source_gaussian=None if uses is None else uses/max(1, source_count),
                  payload_plus_position_uses_per_source_gaussian=None if uses is None else (payload+uses)/max(1, source_count),
                  position_reliability='assumed reliable; no FEC/packet errors simulated',
                  rate_scope='payload + XYZ framing; excludes existing bbox/tier/model metadata')
    return result


class PositionCostMeter:
    """Measure real compressed bytes once per retention mask, not every step.

    Source coordinates and packet order are fixed for this meter. Tier 1/2/3
    differences do not change coordinate bits. Bounded cache also supports q0.
    """
    def __init__(self, cfg, unit):
        self.cfg = cfg
        self.unit = unit.detach().cpu().clone()
        self.integers = None
        if getattr(cfg, 'position_compression', 'none') == 'delta_zlib':
            if not torch.isfinite(self.unit).all():
                raise ValueError('side XYZ must be finite normalized coordinates')
            self.integers = (self.unit.float().clamp(0, 1) * (2**cfg.position_bits-1)).round().numpy().astype(np.int32)
        self.cache = OrderedDict()
        self.last_seconds = 0.
        self.last_cache_hit = False

    def stream_bytes(self, q):
        started = time.perf_counter()
        q = q.detach().cpu().reshape(-1)
        if len(q) != len(self.unit):
            raise ValueError('coordinate mask/source length mismatch')
        if getattr(self.cfg, 'position_compression', 'none') == 'none':
            self.last_seconds = time.perf_counter()-started
            self.last_cache_hit = False
            return None
        keep = (q > 0).numpy()
        key = (getattr(self.cfg, 'position_compression_level', 9), np.packbits(keep, bitorder='little').tobytes())
        self.last_cache_hit = key in self.cache
        if key not in self.cache:
            self.cache[key] = len(_encode_quantized_integers(self.integers[keep], self.cfg))
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        self.last_seconds = time.perf_counter()-started
        return self.cache[key]
