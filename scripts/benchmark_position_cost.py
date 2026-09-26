"""CPU-only XYZ metering benchmark; does not train or modify checkpoints.

Compare cold random masks and cache hits using the actual metering path.
The same mask and quantized coordinates are used across compression levels.
Output includes exact byte counts and bit-exact round-trip assertions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from gaussian_jscc.codec import CodecConfig
from gaussian_jscc.data import read_ply, Geometry, morton_order
from gaussian_jscc.position_delivery import (PositionCostMeter, encode_positions,
                                            decode_positions, delivered_positions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ply',required=True)
    parser.add_argument('--out',required=True,help='New JSON result file; never overwrites')
    parser.add_argument('--levels',type=int,nargs='+',default=[6,9])
    parser.add_argument('--drop',type=float,default=.05)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--cpu-threads',type=int,default=4)
    args=parser.parse_args()
    if not 0<=args.drop<=1 or args.repeats<1 or args.cpu_threads<1:
        parser.error('invalid drop/repeats/cpu-threads')
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    configs=[CodecConfig(position_delivery='quantized',position_bits=16,
                         position_compression='delta_zlib',position_compression_level=v) for v in args.levels]
    torch.set_num_threads(args.cpu_threads)
    raw,_=read_ply(args.ply)
    geometry=Geometry.fit(raw[:,:3],16)
    order=torch.from_numpy(morton_order(geometry.quantize(raw[:,:3]).numpy()).astype(np.int64))
    unit=geometry.normalize(raw[order,:3])
    generator=torch.Generator().manual_seed(42)
    masks=[]
    for _ in range(args.repeats):
        q=torch.randint(1,4,(len(raw),),generator=generator)
        q[torch.rand(len(raw),generator=generator)<args.drop]=0
        masks.append(q)
    results=[]
    for cfg in configs:
        started=time.perf_counter()
        meter=PositionCostMeter(cfg,unit)
        preparation=time.perf_counter()-started
        for i,q in enumerate(masks):
            # A fresh meter cache tests changed-mask compression, not reuse.
            meter.cache.clear()
            size=meter.stream_bytes(q)
            cold=meter.last_seconds
            assert not meter.last_cache_hit
            assert meter.stream_bytes(q)==size and meter.last_cache_hit
            cached=meter.last_seconds
            # Full export is separate from the timed meter; asserts accounting
            # matches actual packet bytes and reconstruction is bit-exact.
            blob=encode_positions(unit,q,cfg)
            assert len(blob)==size
            torch.testing.assert_close(decode_positions(blob,q,cfg),delivered_positions(unit,q,cfg),rtol=0,atol=0)
            row=dict(level=cfg.position_compression_level,trial=i,points=len(raw),retained=int((q>0).sum()),
                     stream_bytes=size,preparation_seconds=preparation,cold_seconds=cold,
                     cached_seconds=cached,roundtrip_exact=True,export_size_matches=True)
            results.append(row)
            print(json.dumps(row),flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:
        json.dump(dict(ply=str(Path(args.ply).resolve()),cpu_threads=args.cpu_threads,drop=args.drop,
                       timing_scope='CPU PositionCostMeter.stream_bytes; no network/render/CUDA; export verification timed separately',
                       results=results),stream,indent=2)


if __name__=='__main__':main()
