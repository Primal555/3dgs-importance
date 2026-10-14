# Progressive latent channel recovery (Truck experiment)

This is an **opt-in** `channel_variant=progressive` experiment. Existing
baseline and residual runs, packet layouts, model identities and checkpoints
remain unchanged. It does not modify the semantic representation or allocator.

## What changes

The transmitter is exactly the existing residual channel encoder: hidden 192,
three pointwise residual blocks, one encoding of the complete symbol sequence,
the same per-point/per-segment power normalization and floor. It is retrained
from fresh channel weights, **not** loaded from a failed communication run.
There is no sender-side residual simulator or receiver feedback in this first
experiment; this isolates the new receiver-side recovery path.

The channel decoder becomes three pointwise stages:

1. The first 4 complex symbols recover the full 48-real-dimensional `y1`.
2. The next 8 complex symbols and `y1` predict `delta2`; `y2 = y1 + delta2`.
3. The next 12 complex symbols and `y2` predict `delta3`; `y3 = y2 + delta3`.

Each stage uses input projection, SNR conditioning, three residual MLP blocks
and a latent readout; width 192, no LayerNorm or cross-point mixing. Enhancement
readouts start with weight std 0.001 and zero bias, so the initial correction is
small but its trunk can receive gradients immediately. These are engineering
initialization choices, not guaranteed optimal settings.

Only stages actually available for a point are run. An entirely unused stage
keeps `grad=None`, avoiding stale Adam momentum changes on q1-only updates.
Stage 1 never sees the final tier ID or enhancement symbols; stage 2 never sees
stage-3 symbols. Under matched received prefix symbols, their intermediate
predictions are independent of whether a higher prefix was delivered. q0 and
padding produce zero latents.

All active stages remain jointly differentiable: **q3 loss can still update
stages 1 and 2**. There is no permanent stage freezing, stop-gradient between
stages, or artificially enforced PSNR ordering. Additive features alone do not
guarantee quality improvement. The decoder has more parameters/compute than
the direct residual decoder; this is not a parameter-matched architecture test.

## What stays fixed

- One already-trained clean-color semantic teacher; both semantic modules and
  attribute normalization statistics remain bit-identical throughout.
- Reliable 16bit XYZ, existing spatial context, row correspondence, block size
  and scene/camera split. XYZ never enters the latent corrections.
- 4 / 12 / 24 complex symbols per Gaussian, with no extra side information.
- 10 dB AWGN; q1 / q2 / q3 / mixed rotation (mixed is random positive tiers,
  not an importance allocator).
- 2000 channel latent-MSE updates, then 2000 source-PLY RGB-MSE + 1 * latent-MSE
  updates. LR 1e-4 for both channel modules, Adam reset at the phase boundary,
  replay rendering, no gradient clipping.

The semantic decoder is frozen, but gradients from rendered images still pass
through it into the channel decoder. The supervised target remains the same
clean semantic `y` for every tier, not the transmitted symbols or quantized XYZ.

## Start on the server

Choose an actually available GPU; `2` below is an example.

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_clean_color_gpu1_20261014_103802"
OUT="$PWD/output/truck_progressive_latent_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_progressive_latent_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log" "$OUT/progressive.log"
```

This default trains **only the progressive arm**, not another semantic model or
the old baseline. To run a newly matched direct-residual control as well, append
`--groups residual progressive`. Do not merge source runs with different
teachers, budgets, view splits or objectives when interpreting the comparison.

To continue an interrupted experiment, repeat the same script with the original
`RUN`, `OUT`, GPU and options and add `--resume-suite`. This restores Adam,
progress and RNG; it is not merely a weights-only initialization.

All outputs are inside `OUT`: `progressive/metrics`, checkpoints, charts,
initial/final images, and `comparison`. Numeric validation occurs every 500
updates. The final numeric tests cover all original test cameras, q1/q2/q3/mixed,
clean bypass, finite noiseless, and 0/5/10/15/20 dB with two paired-noise trials.
The summary audits frozen semantic weights. Compare absolute quality, color
errors, latent NMSE and enhancement gains; a larger gap created by a worse q1
is not success. Clean bypass is a reference, not a mathematical PSNR upper bound.

## After training: shuffle only enhancement symbols

```bash
RUN_PROGRESSIVE="$OUT/progressive"
CUDA_VISIBLE_DEVICES=2 python -u -m gaussian_jscc.latent_prefix_probe \
  --run "$RUN_PROGRESSIVE" \
  --out "$RUN_PROGRESSIVE/diagnostics/prefix_probe" \
  --snr 10 --trials 2 --test-views 4
```

This independent diagnostic makes **zero optimizer updates**. It can also be
used on an existing baseline/residual modular arm. `--test-views 0` tests all
original test cameras; `--manifest` can remap migrated data paths while preserving
the source scene and camera split.

For finite-noiseless and 10 dB communication separately, it compares:

- q1: untouched vs shuffled untransmitted suffix (negative control).
- q2: untouched, shuffled stage-2 symbols, shuffled untransmitted stage-3 symbols
  (negative control).
- q3: untouched, shuffled stage 2, shuffled stage 3, shuffled both enhancements.

The private permutation RNG deranges active points **within each codec batch**,
excluding q0/padding; it does not sample donors from the entire scene. Complete
already-normalized segments are moved without re-normalization. Base symbols,
tier, transmitted lengths, XYZ and physical noise draws remain fixed. No source
feature is given to the receiver. Singleton batches cannot be deranged.

`diagnostics/prefix_probe/summary.json` and `metrics/prefix_probe.jsonl` report
PSNR, SSIM, color errors, latent MSE/NMSE, actual symbol counts, PSNR drop and
latent-error increase, and actual deranged/affected-point counts (including
singleton no-op batches). No extra render panels are saved. The diagnostic
refuses to mark completion if a negative control changes image/latent metrics.
A large positive degradation shows content-dependent
use of the suffix; little degradation alone cannot distinguish encoder redundancy,
decoder underuse or a weak local permutation intervention.

Local verification uses real neural/transport paths and CPU synthetic-render
integration, not actual Truck CUDA rendering. Whether q2/q3 gain improves is a
server experiment result, not a claim made from the local unit tests.
