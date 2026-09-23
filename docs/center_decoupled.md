# Learned centroid + zero-mean local positions, with guarded center updates

## Experiment scope

One random-start run, **5000 center-phase update attempts**, base LR **1e-4**,
seed 42, batches of 32 blocks of 256 points. Attribute and joint phases are not
run. The loss is unchanged:

`mean(sqrt(||predicted_XYZ - source_XYZ||_world^2 + 0.001^2) - 0.001)`.

There is **no replacement by MSE, extra centroid/residual loss, loss weighting,
source-coordinate skip/side delivery, gradient clipping, or test-set acceptance
criterion**. The original absolute-coordinate variant remains loadable and its
default config/checkpoint identity is unchanged.

## Position structure

`--center-position-layout centroid_residual` selects:

1. Common encoder: source absolute normalized XYZ -> learned per-point common
   features (16 of the existing 32 center features).
2. Independent local encoder: source XYZ minus its valid block mean -> learned
   per-point local features (remaining 16). Its geometric context also uses the
   centered coordinates. There are no common/local shared trainable parameters.
3. Common decoder: masked mean of received common features -> MLP -> predicted
   block centroid. It has no source coordinate input.
4. Independent local decoder: existing multi-tap affine-readout Transformer,
   self-only attention, on local features -> per-point residual. Subtract its
   masked predicted block mean, then add the **predicted** common centroid.

`predicted_XYZ_i = predicted_centroid + zero_mean(predicted_residual)_i`.

The encoder computes a source mean to center its *input*, but does not pass that
mean to the decoder. Both centroid and local positions must be learned. Total
center-feature width remains 32; this clean-representation experiment does not
train a communication payload or change individual Gaussian tier allocation.

Only local parameters changing cannot move the output block mean; only common
parameters changing cannot deform its relative positions. This is output and
parameter-path decoupling, **not independence of the existing loss gradients**.
Both paths are free to train; subsequent joint rendering can still update both.
Local residual centering introduces a within-block dependency even with a
self-only Transformer. Group membership must stay fixed for comparisons.

The total parameter count increases because the two encoders are independent;
this is not a parameter-matched architecture ablation. Halving the local feature
width, common-centroid errors affecting a whole block, small centered inputs and
block boundary discontinuities are risks to examine, not assumed solved.

## Update guard

`--center-step-guard --center-max-backtracks 4` applies **only during center
training**, from the first step:

- Compute the original distance loss and gradient once on the sampled training
  batch. Make one Adam proposal, saving original parameters and optimizer state.
- Evaluate the same loss on that same batch at proposal scales
  **1, 1/2, 1/4, 1/8, 1/16**, accepting the first finite non-increasing loss.
- No extra backward pass or rendering is required, but each attempted scale
  costs an additional center forward pass. There is no universal speed guarantee.
- An accepted reduced step keeps Adam's new moments/counter exactly once, as
  if that iteration used the scaled LR. The next base LR remains 1e-4.
- If all proposals fail, restore parameters **and** Adam moments/counter.
  The attempted-training-step counter advances and the skipped update is logged.
  Exceptions also restore the candidate state before being re-raised.

There is no added permanent LR decay schedule or arbitrary coordinate-motion
threshold. Acceptance protects the current batch, **not other blocks or PSNR**.
Frequent rejection is a diagnostic signal, not a claim that training succeeded.
The guard is scoped to the current deterministic center networks (no dropout or
running-stat buffers); do not reuse it unchanged for stochastic objectives.

## Server launch

Replace GPU 2 with a currently available card:

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c http.version=HTTP/1.1 -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_center_decoupled_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_center_decoupled.sh "$OUT" > "${OUT}.launcher.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.launcher.log"
```

All analysis results and a console-log copy are inside OUT. External launcher
log/PID are only background-process conveniences. No old checkpoint is loaded.

## What to inspect

- Every 500 attempts: existing PSNR/SSIM, center-only images, coordinate snapshots
  and drift curves. Full-render attributes are untrained; use `center_only` to
  assess center reconstruction. The best-validation `codec.pt` need not be step
  5000; explicit `codec_5000.pt` and `codec_last.pt` are the last weights.
- Every 50 attempts and step 1: fixed-probe before/after *accepted* update drift,
  split into global mean, block mean and pointwise residual. Rejections produce
  zero accepted movement; read the guard record to distinguish this from a
  genuinely small proposal.
- `loss.jsonl` stores `stats.step_guard`: before/after loss, all tried scales and
  losses, accepted flag, scale and effective LRs. The usual `loss` is pre-update.
- Separate common/local encoder/decoder parameter gradient/update norms are
  recorded, without double-counting shared parameters (there are none here).
- `charts/center_step_guard.png` shows accepted scales, before/after same-batch
  loss and cumulative skipped fraction. `summary.json` counts accepted, reduced,
  skipped updates. `training_state_last_center.pt` retains matching last Adam
  state before best-weight export.

This experiment combines structural separation and update control. It can test
whether the combination helps, but cannot attribute all improvement to either
change alone. No CUDA quality gain is claimed from CPU unit/smoke tests.
