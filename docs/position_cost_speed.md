# Faster coordinate cost accounting

The 49-second mixed render updates were dominated by repeated CPU zlib level-9
compression, not the neural codec: 5% random dropping changes the retention mask
each mixed update, defeating the whole-mask coordinate cost cache.

## Changes

1. **Render updates:** keep random dropping, layouts, image loss and gradients
   unchanged. Do not compress coordinates for per-step logging. For compressed
   XYZ, actual stream bytes/bits, coordinate channel uses and total channel uses
   are JSON `null`, with `position_cost_status: not_measured`. Retained counts,
   packed/uncompressed coordinate bits and JSCC payload remain available. No
   estimates or stale measurements are substituted. Validation still measures
   actual compressed costs at the normal validation interval.
2. **Joint allocation:** continue measuring each sampled mask's actual XYZ and
   tier-map bytes, including their contribution to REINFORCE. New training uses
   zlib level **6** for XYZ. Validation, normalizer and export use the same saved
   setting. The tier-map compressor remains at level 9: it was not the bottleneck.
3. **Fixed preprocessing:** cache the quantized integer XYZ once per cost meter.
   On a new retention mask, select integers, recompute retained-neighbor deltas,
   then compress. Never reuse pre-drop deltas after deleting points. Export and
   cost metering share the same integer encoder and framing.

Neither compression level changes the recovered 16-bit coordinates. Lower
levels generally trade compressed size for time, not coordinate precision.
Actual size still depends on the scene and mask. Compression is not a GPU VRAM
optimization. No hard communication-budget guarantee is introduced.

## Configuration and compatibility

`train-learned --position-compression-level 6` explicitly sets the level on a
new run; omitting it defaults to 6. Existing launch scripts inherit this default.
The resolved setting is recorded in training.json and checkpoint configuration.

`--init` preserves the checkpoint's compression level; historical checkpoints
without the new field retain level 9 and their existing model/packet hashes.
An explicitly conflicting level is rejected rather than silently invalidating
matched codec/allocation checkpoints. This is weight initialization, NOT exact
optimizer/schedule resume. Changing a running checkout does not hot-patch an
already-running Python process. Do not stop/restart existing jobs automatically.

## Diagnostics

Per-step JSON adds `rate_accounting_seconds`, `position_compression_seconds` and
`codec_task_seconds`; joint also adds `tier_map_compression_seconds` (summed over
mask samples). Console progress includes `rate_sec`. These are wall-clock
subranges, not synchronized CUDA kernel timings. `step_seconds` continues to
exclude checkpoint writing, validation and plotting. Validation cost metadata
includes the actual compression level and measurement status.

## Reproducible CPU-only benchmark

From the project root, with the Python environment activated:

```bash
python -u scripts/benchmark_position_cost.py \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --out output/position_cost_benchmark.json \
  --levels 6 9 --drop 0.05 --repeats 1 --cpu-threads 4
```

The output file must not already exist. The benchmark uses the same source,
Morton ordering and sampled mask across levels; measures cold cost-meter calls
and cache hits; and separately verifies export byte counts and exact decoded XYZ.
It writes no training checkpoint and needs no CUDA renderer. Level-9 export
verification repeats the slow compression, so allow a few minutes. Local CPU
timings do not establish full server GPU training speed or final image quality.

Regression coverage includes empty/all-kept/mixed masks, multiple bit depths
and levels, cache behavior, legacy configuration, packet round trips, render
updates without compression, and synthetic three-stage allocation training with
the measured-cost penalty still active.

## Local measurement, 2026-09-26

One paired mask on the truck source (883438 points, 839115 retained), 4 CPU
threads. Both rows use the new `PositionCostMeter` with precomputed integers;
these are cold calls, not cache hits. This is a single local timing sample,
not a server GPU throughput benchmark; other CPU activity can affect timings.

| XYZ zlib level | Framed bytes | Cold meter seconds | Cached seconds |
|---|---:|---:|---:|
| 6 | 3303453 | 1.020 | 0.00079 |
| 9 | 3208695 | 26.625 | 0.00072 |

Level 6 used 94758 extra bytes (+2.95%) and the cold meter call was about 26x
faster in this sample. Both matched exported stream size and recovered exactly
the same quantized coordinates. Render updates no longer call this compressor
at all. The full test suite at this change passed (119 tests, 3 skipped);
synthetic renderer tests check control flow/gradients, not image fidelity.
