"""Compare lossless XYZ file formats on an existing 16-bit delivery packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path
import struct
import subprocess
import time
from types import SimpleNamespace
import zlib

import numpy as np
import torch

from gaussian_jscc.position_delivery import decode_positions
from gaussian_jscc.transport import decode_metadata

from .side_information_accounting import side_information_cost


METHODS = {'delta_zigzag_zlib': 1, 'morton_xor_lzma': 2}
FRAME = struct.Struct('<4sBQI')


def morton_keys(xyz):
    keys = np.zeros(len(xyz), dtype=np.uint64)
    xyz = xyz.astype(np.uint64)
    for bit in range(16):
        for axis in range(3):
            keys |= ((xyz[:, axis] >> bit) & 1) << (3 * bit + axis)
    return keys


def morton_positions(keys):
    xyz = np.zeros((len(keys), 3), dtype=np.int32)
    for bit in range(16):
        for axis in range(3):
            xyz[:, axis] |= (((keys >> (3 * bit + axis)) & 1) << bit).astype(np.int32)
    return xyz


def byte_planes(words, width):
    return words.view(np.uint8).reshape(len(words), words.dtype.itemsize)[:, :width].T.copy().tobytes()


def unpack_planes(data, count, width, dtype):
    planes = np.frombuffer(data, dtype=np.uint8).reshape(width, count)
    words = np.zeros(count, dtype=dtype)
    words.view(np.uint8).reshape(count, words.dtype.itemsize)[:, :width] = planes.T
    return words


def encode(xyz, method):
    if method == 'delta_zigzag_zlib':
        delta = np.diff(xyz, axis=0, prepend=np.zeros((1, 3), dtype=np.int32))
        zigzag = ((delta << 1) ^ (delta >> 31)).astype('<u4')
        payload = zlib.compress(byte_planes(zigzag.T.copy().reshape(-1), 3), level=6)
    elif method == 'morton_xor_lzma':
        keys = morton_keys(xyz)
        # Each word preserves the changed bits relative to the preceding point.
        previous = np.concatenate((np.zeros(1, dtype=np.uint64), keys[:-1]))
        changes = np.bitwise_xor(keys, previous).astype('<u8')
        payload = lzma.compress(byte_planes(changes, 6), preset=3)
    else:
        raise ValueError('unknown XYZ coding method')
    return FRAME.pack(b'XZ16', METHODS[method], len(xyz), zlib.crc32(payload)) + payload


def decode(data, expected_count):
    if len(data) < FRAME.size:
        raise ValueError('truncated XYZ frame')
    magic, mode, count, crc = FRAME.unpack_from(data)
    payload = data[FRAME.size:]
    if magic != b'XZ16' or count != expected_count or count < 1 or zlib.crc32(payload) != crc:
        raise ValueError('invalid XYZ frame header, count, or CRC')
    if mode == METHODS['delta_zigzag_zlib']:
        decoder = zlib.decompressobj()
        size = count * 9
        data = decoder.decompress(payload, size + 1)
        if len(data) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('invalid XYZ delta payload length')
        values = unpack_planes(data, count * 3, 3, '<u4').astype(np.int32)
        delta = ((values >> 1) ^ -(values & 1)).reshape(3, count).T
        xyz = np.cumsum(delta, axis=0, dtype=np.int64)
        if xyz.min() < 0 or xyz.max() > 65535:
            raise ValueError('XYZ delta positions outside the 16-bit grid')
        return xyz.astype(np.int32)
    if mode == METHODS['morton_xor_lzma']:
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=32 << 20)
        size = count * 6
        data = decoder.decompress(payload, max_length=size + 1)
        if len(data) != size or not decoder.eof or decoder.unused_data:
            raise ValueError('invalid XYZ Morton payload length')
        return morton_positions(np.bitwise_xor.accumulate(unpack_planes(data, count, 6, '<u8')))
    raise ValueError('unknown XYZ frame mode')


def run_benchmark(packet, out, *, digital_python=None, snr_db=10., seed=42):
    packet, out = Path(packet), Path(out)
    old_xyz = (packet / 'xyz.bin').read_bytes()
    meta = (packet / 'metadata.bin').read_bytes()
    scales = packet / 'scales.bin'
    header, tiers = decode_metadata(meta)
    cfg = SimpleNamespace(position_delivery='quantized', position_bits=16, position_compression='delta_zlib')
    xyz = decode_positions(old_xyz, tiers, cfg)
    keep = tiers > 0
    retained = xyz[keep]
    coordinates = (retained * 65535).round().numpy().astype(np.int32)
    if not len(coordinates) or not torch.equal(torch.from_numpy(coordinates).float() / 65535, retained):
        raise ValueError('expected nonempty 16-bit grid coordinates')
    attribute_symbols = int(torch.tensor(header['config']['rates'])[tiers].sum())
    sizes = dict(payload_complex_symbols=attribute_symbols, xyz_bytes=len(old_xyz),
                 metadata_bytes=len(meta), scale_stream_bytes=scales.stat().st_size)
    baseline = side_information_cost(sizes)
    report = dict(packet=str(packet.resolve()), point_count=len(tiers), retained_count=len(coordinates),
        baseline_xyz_sha256=hashlib.sha256(old_xyz).hexdigest(), baseline=baseline, variants={})
    out.mkdir(parents=True, exist_ok=False)
    for method in METHODS:
        start = time.perf_counter()
        encoded = encode(coordinates, method)
        encode_seconds = time.perf_counter() - start
        path = out / f'{method}.bin'
        path.write_bytes(encoded)
        start = time.perf_counter()
        recovered = decode(path.read_bytes(), len(coordinates))
        decode_seconds = time.perf_counter() - start
        if not np.array_equal(coordinates, recovered):
            raise RuntimeError(f'{method} altered quantized coordinates or order')
        recovered_unit = torch.from_numpy(recovered).float() / 65535
        if not torch.equal(retained, recovered_unit):
            raise RuntimeError(f'{method} altered renderer XYZ input')
        cost = side_information_cost(dict(sizes, xyz_bytes=path.stat().st_size))
        report['variants'][method] = dict(
            exact_coordinates_and_order=True, max_abs_xyz_difference=0.0, xyz_file=path.name,
            xyz_bytes_saved=len(old_xyz) - path.stat().st_size,
            channel_uses_saved=baseline['total_channel_uses'] - cost['total_channel_uses'],
            encode_seconds=encode_seconds, decode_seconds=decode_seconds, accounting=cost)
    best = min(report['variants'], key=lambda name: report['variants'][name]['accounting']['total_channel_uses'])
    report['best_tested_variant'] = best
    report['digital_link_test'] = None
    summary = out / 'summary.json'
    summary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if digital_python:
        links = {}
        for method, source in [('baseline', packet / 'xyz.bin'), (best, out / f'{best}.bin')]:
            received = out / f'{method}.received.bin'
            link_path = out / f'{method}.link.json'
            subprocess.run([digital_python, '-m', 'comparison_baselines.digital_link', 'transmit',
                '--source', str(source), '--destination', str(received), '--report', str(link_path),
                '--snr-db', str(snr_db), '--seed', str(seed)], check=True)
            record = json.loads(link_path.read_text(encoding='utf-8'))
            accounting = baseline if method == 'baseline' else report['variants'][method]['accounting']
            if record['total_channel_uses'] != accounting['streams']['coordinates']['total_channel_uses']:
                raise RuntimeError('actual XYZ link cost differs from packet accounting')
            if record['status'] == 'success':
                if method == 'baseline':
                    if not torch.equal(decode_positions(received.read_bytes(), tiers, cfg), xyz):
                        raise RuntimeError('baseline changed after digital delivery')
                elif not np.array_equal(decode(received.read_bytes(), len(coordinates)), coordinates):
                    raise RuntimeError('candidate XYZ changed after digital delivery')
            links[method] = record
        report['digital_link_test'] = links
        summary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--digital-python', help='optional Sionna interpreter for the paired XYZ link test')
    parser.add_argument('--snr-db', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = run_benchmark(args.packet, args.out, digital_python=args.digital_python,
                           snr_db=args.snr_db, seed=args.seed)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
