# Pointwise absolute XYZ + joint directional backtracking

## Restored baseline

Git reverts `ebc89d0` and `85c45ff` restore the center architecture from
`52edc1c`. The encoder, decoder, configuration, and center loss are unchanged
from that baseline. Explicit learned block centroids, split 16+16 latents,
zero-mean residual constraints, and branchwise updates have been removed.
Old decoupled runs and checkpoints remain on disk and in Git history, but
cannot be resumed into this different architecture.

The experiment uses a 32-dimensional per-point center latent, a four-layer
Transformer-structured decoder with **self-only** scope (no cross-point
attention in the decoder), and affine multilevel readouts to absolute XYZ.
The encoder still uses geometric context. `center` means each Gaussian's
center, not a block centroid. Attributes are frozen and untrained in this run;
use `center_only` (source attributes + predicted XYZ), not `full`, to judge XYZ.

## Only the update rule changes

With `--center-step-guard`, encoder and decoder propose one joint Adam update.
No extra loss, target, latent, position bypass, or validation feedback is added.

1. Check the gradient dot proposed displacement. If uphill, skip shrinking
   that direction and retry after clearing the active Adam first moments.
2. For a downhill candidate, test scales 1, 1/2, ..., 1/256 on the **same
   training batch**, accepting the first finite strict loss decrease.
3. If all trials fail, retry with cleared first moments. Adam second moments
   and counters are retained from before the proposal; this is not fresh Adam.
4. A total rejection restores weights, second moments and counters, but clears
   first moments so stale momentum is not retried forever. Exceptions restore
   the complete pre-step optimizer state, including first moments.

Accepted scaled proposals retain the candidate Adam moments and advance its
counter once, not once per trial. Base LR stays 1e-4: scale is per-step, not a
permanent LR schedule. Warnings after 50 consecutive rejections, and a stop
after 200, prevent silently reporting stalled training as convergence.
The guard applies only to center pretraining; B/C optimizer behavior is unchanged.

This guarantees neither heldout error nor render quality improvement. Extra
forwards cost time, and strict minibatch descent can slow learning. Logs expose
acceptance, each trial loss, scale, restart, direction, and rejection streak.
The original `scripts/test_center_drift.sh` remains an unguarded baseline;
`STEP_GUARD=0` also disables guarding in the new wrapper for a matched run.

## Server launch (fresh random initialization)

Select an actually available GPU; the example uses GPU 2, not a reservation:

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_center_absolute_guard_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_center_absolute_guard.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

Defaults: 5000 center steps, no B/C stages, base LR 1e-4, no gradient clipping,
seed 42, 256-point blocks, 32 blocks/batch. Checkpoints, PSNR/SSIM and rendered
images every 500 steps; fixed-point drift records every 50 steps. No `--init`
or previous weights. Logs, images, metrics and charts share the run directory.
`codec.pt` selects the best validation checkpoint; `codec_5000.pt` is the final
iteration, while `training_state_last_center.pt` preserves its matching Adam
state for diagnostics. `charts/center_step_guard.png` explains actual updates.
