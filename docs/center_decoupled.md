# Learned centroid + zero-mean local positions, with branch guard v2

The server wrapper now selects the corrected **branch_directional_v2** update
mechanism. Network structure, latent widths and position loss are unchanged
from the first decoupled experiment. Old `--center-guard-mode joint` behavior
remains available for explicit replay of earlier configurations, not as the
recommended server experiment.

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

## Corrected update guard

`--center-step-guard --center-guard-mode branch --center-max-backtracks 8`
applies **only during center training**, from the first step:

1. Propose a common-branch Adam update, leaving local parameters and their
   optimizer counters untouched. Check `gradient dot parameter_update < 0`.
2. An uphill/zero/nonfinite direction is not blindly shrunk. Restore the
   pre-candidate state, clear **only active first moments**, retain second
   moments/counters, and rebuild a candidate from the current gradient.
3. For a descent direction try **1, 1/2, ..., 1/256**, accepting the first finite
   **strictly lower** original loss on the same training batch. If the original
   direction passes its dot-product check but all scales fail, also try the
   first-moment-restart candidate. At most two directions are considered.
4. After common acceptance/rejection, recompute the same original objective
   and the **fresh local gradient at the actual new parameters**. Propose and
   accept/reject the local branch independently. No stale local gradient and
   no all-or-nothing veto from a failed common branch.

An accepted update advances that branch's moments/counters **once**, regardless
of retries. Partial acceptance scales the current proposal, not the next base
LR. If both candidate directions fail, restore that branch's weights, second
moments and counters, but leave its first moments cleared for the next batch.
This intentional rejection-state change avoids repeatedly restoring the same
known bad momentum. Exceptions instead restore the **entire outer iteration**,
including an already accepted common step, for consistent checkpoint recovery.

The base LR stays 1e-4; there is no permanent LR decay or new coordinate-motion
threshold. No heldout loss, image, or rendering enters acceptance. Each trial
costs a center forward pass; the second branch additionally needs a fresh
gradient computation. This version is not claimed to be as fast as the old
single-update code. It remains scoped to the current deterministic networks
(no dropout/running-stat buffers), not arbitrary stochastic training.

`--center-guard-warn-after 50 --center-guard-stop-after 200` are disclosed
engineering safeguards: warn every 50 consecutive rejected attempts for a
branch; stop if **either** branch reaches 200. Accepted steps reset only that
branch's streak. Streaks persist across exact resume. Stopping saves last
weights/Adam, metrics, images and a summary with **status `guard_stalled`**;
later phases are not run. It is not declared convergence. A branch could be
nearly converged while another is still learning, so this conservative stop is
a request to inspect diagnostics, not proof of a broken network.

Strict float-loss equality is rejected, not counted as learning. On each
accepted branch step, current-batch loss must decrease; no guarantee is made
about other blocks or PSNR.

## Server launch

Replace GPU 2 with a currently available card:

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c http.version=HTTP/1.1 -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_center_decoupled_v2_$(date +%Y%m%d_%H%M%S)"
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
- `loss.jsonl` stores `stats.step_guard.branches.common/local`: candidate
  directional derivatives, momentum restarts, all trial scales/losses, accepted
  flags, each branch's actual gradient norm/effective LRs and rejection streaks.
  Top-level `scale` is only the mean of the two branch scales, **not** a global
  effective LR. The usual `loss` is before either branch update.
- Separate common/local encoder/decoder parameter gradient/update norms are
  recorded, without double-counting shared parameters (there are none here).
- `charts/center_step_guard.png` shows separate branch scales and skipped
  fractions plus before/after same-batch loss. `center_momentum_restarts.png`
  shows restart fractions and original directional derivatives. `summary.json`
  counts accepted/reduced/skipped/restarted updates per branch.
  `training_state_last_center.pt` retains matching last Adam state before
  best-weight export, including on stall.

This experiment combines structural separation and update control. It can test
whether the combination helps, but cannot attribute all improvement to either
change alone. No CUDA quality gain is claimed from CPU unit/smoke tests.
