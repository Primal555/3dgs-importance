# Eight-prefix attribute JSCC experiment

The experiment retains the fixed 16-bit, losslessly delta/zlib-compressed
quantized coordinate side stream. It learns **one** SNR-conditioned codeword,
with cumulative attribute budgets of **4, 8, 12, 16, 20, 24, 28, 32 complex
symbols per Gaussian**. q0 still means drop/padding, not the 4-symbol tier.
q1..q8 index these lengths; old q3=32 is now q8=32.

The 4-symbol spacing is an experimental measurement grid, not a theoretically
optimal allocation. The maximum payload stays at 32 complex symbols. Decoder
tier embeddings have grown; no new Transformer blocks or hidden width were
added. Positive tiers do not condition the encoder, including neighboring
points' tiers. The receiver sees only each delivered prefix and its length.

## Training and comparability

- Random initialization, no historical three-tier checkpoint.
- Stage A: isolated local-response attribute initialization (no scene rendering
  in its training objective). Stage B: full-scene multiview source-render MSE.
- Fixed 10 dB AWGN, LR 1e-4 in both phases, 16-bit coordinates fixed throughout.
- Replay backward, 64 blocks/batch, no learned mask or q0 sampling.
- Each phase cycles q1..q8 then a per-Gaussian independently sampled mixed
  layout: **nine optimizer updates per full cycle**, not nine renders per update.
- Each 4-symbol incremental layer has its own power normalization. This changes
  boundaries versus the historical 8+8+16 partition. Even the 8/16/32 endpoints
  are therefore not a strict same-weights ablation of the old codeword.

Defaults are **11250 bootstrap + 11250 render updates**: 1250 uniform updates
per tier per phase, matching 5000/4 in the old three-tier-plus-mixed schedule.
This does **not** match wall time, total data exposure, mixed-layout frequency,
or optimization difficulty. `BOOTSTRAP_STEPS=5000 RENDER_STEPS=5000` gives an
equal-total-update preliminary scan, but only about 556 uniform updates/tier.
Do not interpret unfinished high-prefix learning as a fundamental rate ceiling.

Training logs record the actual per-phase layout update counts. Validation uses
the same held-out training-camera subset and paired point/symbol noise across
all prefixes. All positive tiers plus mixed are validated; selection uses their
mean MSE (equal weight per layout), so the lowest-rate objective can influence
the selected checkpoint. Final and best checkpoints remain distinct.

## Server launch

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git pull --ff-only --no-recurse-submodules origin main
GPU=2  # example only: check availability before launching
OUT="$PWD/output/truck_progressive16_rate_sweep_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/train_progressive16_rate_sweep.sh \
  "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

`Ctrl+C` exits `tail`, not the nohup training process. All checkpoints, logs,
images and charts are inside `OUT`; the launcher log and PID file are alongside
it for compatibility with existing launchers.

## What to inspect

Figures refresh at every validation (default every 500 **phase** updates), not
just at successful completion. Phase endpoints are also always evaluated.

- `charts/rate_distortion.png` / `.svg`: measured length vs source-render PSNR,
  MSE and SSIM, showing latest endpoints of each phase. No mixed-layout point is
  passed off as a uniform intermediate length. Connecting segments do not claim
  performance at untrained lengths.
- `charts/rate_distortion.csv`: all measured checkpoints and uniform prefixes,
  also includes photo metrics and the payload+coordinate cost estimate.
- `charts/rate_marginal_gain.png`: latest paired adjacent-prefix PSNR gain and
  MSE reduction divided by extra complex symbols/Gaussian. Negative gains stay
  visible. This is not a monotonicity constraint or an automatic optimum claim.
- `charts/prefix_gains.csv`: adjacent gains at every check; includes the fraction
  of paired view/trial observations improved.
- `charts/validation_quality.png`, `charts/bootstrap_validation.png`: all tiers'
  histories, keeping the two objectives conceptually separate.
- `validation.jsonl`: per-view/per-trial metrics, actual lengths and full tier
  histograms. `loss.jsonl`: objectives, gradients, timings and update exposure.
- `validation_images/<global_step>/1_view00.png` through `8_view00.png` plus
  `mixed_view00.png`, repeated for all validation views. Panels show Photo,
  Source PLY, Received, and absolute error x4.

Validation views are used for monitoring/selection, **not** final test evidence.
Four views/two noise trials are a lightweight default, not a confidence guarantee.
Training curves charge attribute payload and coordinate cost where labelled;
they do not silently call that complete over-the-air cost. Dense tier IDs need
4 raw bits/point versus the old 2; packet metadata compresses and accounts for
them. Actual full metadata accounting is available via `benchmark-codec` and
`transmit`. Shared weights remain assumed available at the receiver.

## Other commands / checkpoint compatibility

`--rates` accepts a variable-length increasing table starting at zero (2..256
entries). `evaluate` and `benchmark-codec` default to all positive checkpoint
tiers. `--tiers 2 4 8` selects 8/16/32 symbols for this experiment. `transmit
--uniform-tier 8` sends 32 symbols. Explicit `--rates` with `--init` must match
the checkpoint; no embedding resize or reinterpretation is performed.

Existing four-entry checkpoints retain their embedding sizes, config hashes
and 2-bit metadata format. Expanded tables use the same checkpoint container
with their full rate table and bit-width-tagged packet metadata. Old software
cannot decode new dense-tier packets; both ends must update.
