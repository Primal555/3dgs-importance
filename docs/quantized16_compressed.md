# 16-bit coordinates with lossless side-stream compression

Current entry: `scripts/train_quantized16_render_only.sh`. The historical
`train_quantized12_render_only.sh` remains an unchanged comparison launcher.

Only coordinate precision and serialization change. The lightweight attribute
JSCC network, source-scene multiview RGB MSE, random initialization, 1e-4 LR,
10 dB AWGN, tiers 8/16/32, replay and 64-block batching remain unchanged.
Default: 5000 render updates; validation/save every 500 updates.

## Wire representation

Normalize XYZ by the transmitted global bbox and round each axis to 0..65535.
Only q>0 rows are included, in existing packet/Morton order; duplicates and
attribute correspondence are retained. First coordinate delta is relative to
zero. Encode consecutive signed XYZ deltas as little-endian int32, transpose
to axis-major byte planes, then zlib level 9. This is reversible; no learned
position correction, deduplication or further quantization occurs.

`GXYZ` mode 3 identifies this format. The existing 18-byte header carries mode,
precision, count and CRC32. Decoder verifies config/count/CRC, limits expansion
to the exact 12*count delta bytes, rejects trailing/truncated streams and invalid
integer ranges, then reconstructs normalized float32 XYZ. Modes 1/2 still read
historical float32/packed-quantized streams. Default `position_compression=none`
is omitted from serialized config to preserve historical model hashes.

**Lossless means lossless over the quantized integers, not original float XYZ.**
The digital side channel is still assumed reliable; FEC, modulation, packet
loss and retransmission are not simulated. CRC is detection, not correction.

## Cost and memory

Packet stats use actual `xyz.bin` bytes including framing. Training/validation
measure the same encoder and cache at most eight retention masks: switching
q1/q2/q3 does not recompress identical XYZ. A changed q0 mask requires a new
measurement. First validation may pause for CPU compression. No compression
ratio is assumed from the number of points alone; small/unstructured streams
may expand. `position_content_bits` is uncompressed quantized information width;
`position_stream_bits` is measured coded size. Assumed net bits/use remains
explicit, with bbox/tier/model overhead reported separately in packet export.

This reduces wire/file size, not necessarily GPU or peak host RAM. Decoder
returns float32 coordinates for the existing renderer; signed deltas and
compression buffers are temporary allocations. No VRAM saving is promised.

Full-scene local protocol check (883,438 Truck Gaussians, all retained, existing
Morton order, including the 18-byte XYZ header): packed12 = 3,975,489 bytes;
compressed16 = 3,370,816 bytes (15.21% smaller). All recovered quantized coordinates
were bit-exact. World-coordinate component RMSE against source XYZ decreased
from 0.0118983 to 0.000744034; point-distance P95 from 0.0300356 to 0.00187716.
These are coordinate metrics, not a new GPU-render PSNR claim. On this local
run, concurrently with tests, encoding took 52.95 s and decoding 0.118 s;
runtime is machine/load dependent. Compression is not inside each training step.

## Launch and compatibility

```bash
CUDA_VISIBLE_DEVICES=2 RENDER_STEPS=5000 nohup bash \
  scripts/train_quantized16_render_only.sh "$OUT" > "${OUT}.log" 2>&1 &
```

Select a free GPU. Overrides: PLY, SCENE, PYTHON_BIN, RENDER_STEPS,
BLOCKS_PER_BATCH, VALIDATE_EVERY, SAVE_EVERY. No inferred checkpoint reuse:
default is random weights. `INITIALIZATION=checkpoint INIT=...` requires a
matching 16-bit delta_zlib model and starts a fresh optimizer, not exact resume.
12-bit checkpoints are not silently reinterpreted as 16-bit checkpoints.

CLI equivalent flags: `--position-delivery quantized --position-bits 16
--position-compression delta_zlib`. Model config propagates these settings to
packet transmitter and receiver. Attribute payload is unchanged; the XYZ
stream consumes additional, explicitly counted channel resources.
