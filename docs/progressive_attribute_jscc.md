# Progressive attribute JSCC with reliable compressed 16-bit XYZ

This experiment keeps the current shared lightweight backbone, reliable XYZ
delivery, fixed 10 dB AWGN, and source-scene multiview RGB MSE. It does **not**
introduce separate representation/channel networks, a learned position path,
parameter reconstruction auxiliaries, or residual encoder/decoder cascades.

## Coding contract

`CodecConfig.prefix_mode = "progressive"` is explicit checkpoint/packet metadata.
For a fixed source block, SNR, and retained set (`q > 0`), the encoder computes
one dense 32-complex-symbol codeword per retained Gaussian. It sees SNR but not
the individual positive tier choices, including those of neighboring points.

| Tier | Layers delivered | Total complex symbols |
| --- | --- | --- |
| q0 | none; no XYZ side-stream row | 0 |
| q1 | base 8 | 8 |
| q2 | base 8 + enhancement 8 | 16 |
| q3 | base 8 + enhancement 8 + enhancement 16 | 32 |

Each incremental layer is power-normalized independently with the existing
positive power floor. Mean energy per complex symbol is <= 1 in each layer;
more symbols consume more total energy. Appending a layer never rescales the
base. All positive tiers share the same 16-bit coordinate precision.

The shared decoder still receives tier/SNR conditions and a validity mask.
Unreceived symbols are zeroed before decoding. Each Gaussian can have its own
tier within a block. Neighboring received enhancements may help through decoder
context; no unreceived enhancement is accessible. Changing the retained set
can change sender context, and therefore requires a new encoding.

For actual encode-once reuse, `GaussianCodec.encode_full(features, xyz,
retained_bool, snr)` returns dense real/imag slots. `pack(full, q, cfg.rates)`
selects any desired layout without another encoder call; the caller must keep
the same retained set. Ordinary `encode` does both in one call. The existing
packet container packs selected symbols point by point; a prefix per Gaussian
is **not** equivalent to truncating the entire `received.npy` file. No new
incremental network transport protocol or ACK/feedback channel is claimed.

This follows the shared-decoder/structured-prefix approach in
[DeepJSCC-l, IV-E1](https://arxiv.org/html/2009.12480#S4.SS5.SSS1), adapted to
Gaussian attributes and rendering. It is not a full reproduction of the paper.

## Training and observation

The schedule continues to cycle q1, q2, q3, per-Gaussian mixed: one multiview
render objective and one optimizer update each step. No forced PSNR gap, no
monotonic penalty, no staged forgetting of low rates. Default random weights,
5000 render steps, LR 1e-4, 64 blocks per batch, replay, CPU training-data cache,
resolution 2. No bootstrap and no mask-optimization phase.

Validation fixes views and assigns common random noise to dense point/symbol
slots before selecting the received prefix. Thus all layouts share the noise
of their shared symbols. Unreceived noisy slots are not used, sent or charged.
Noise is resampled independently between trials and restored outside validation.
Training and packet simulation retain ordinary packed-channel sampling.

All results remain inside a single output directory:

- `training.json`: mode, layer lengths, optimizer and validation settings.
- `loss.jsonl`: training layout, MSE, gradients, updates, timing.
- `validation.jsonl`: source-render and photo PSNR/SSIM/MSE; `prefix_gains`
  records q1->q2 and q2->q3 PSNR gains, MSE reductions and improved paired
  view/trial fractions. Negative gains are reported, not hidden.
- `validation_images/000500/`, etc.: all three tiers plus mixed, saved every
  validation interval (default 500) and at initialization/final evaluation.
- `charts/validation_quality.*`, `charts/prefix_gains.*` and CSV exports after
  training. Sparse histories use markers, not an implied smooth trend.
- `codec_500.pt`, etc., `codec_best_render.pt`, `codec.pt` (last, not best).

Coordinate compressed bytes and assumed reliable-link costs retain their
existing accounting. The layer table counts attribute payload only, not the
digital XYZ/metadata cost. Side-stream FEC remains an assumption, not simulated.
Progressive structure does not mathematically guarantee monotonic quality or a
higher q3 ceiling. CPU synthetic-render checks test plumbing, not scene PSNR.

## Server launch

First activate `maskgs`, pull `main`, and select a genuinely available GPU. This
example chooses GPU 2, but does not assert its current availability.

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c http.version=HTTP/1.1 -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main || exit 1

OUT="$PWD/output/truck_progressive16_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 INITIALIZATION=random RENDER_STEPS=5000 \
  nohup bash scripts/train_progressive16_render_only.sh "$OUT" > "${OUT}.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "${OUT}.pid"
printf 'PID: %s\nOutput: %s\n' "$TRAIN_PID" "$OUT"
tail -f "${OUT}.log"
```

`Ctrl+C` stops `tail`, not the background training. Disconnecting the client
does not terminate this nohup job. `RENDER_STEPS=10000` changes the step budget;
`VALIDATE_EVERY`, `SAVE_EVERY`, `BLOCKS_PER_BATCH` retain baseline overrides.

Historical checkpoints default to `adaptive` and retain their hashes and exact
coding behavior. Changing mode under old weights is not silently allowed.
The new launcher defaults to random initialization; `INITIALIZATION=checkpoint`
requires a matching progressive/16-bit/delta_zlib checkpoint, loads weights but
starts fresh Adam (not exact resume). The previous launcher remains adaptive
by default for comparison.
