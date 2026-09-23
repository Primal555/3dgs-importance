# Single-run center drift diagnostic

This is **one fresh random-initialization run**, not a checkpoint fork or LR
sweep. It retains the previous self-only Transformer center decoder, affine
readout, shared center encoder, fixed seed 42, batch 32, and unchanged world
smooth-distance center loss. LR is fixed at **1e-4 for 5000 center steps**.
No attribute/joint training, noise/tier experiment, visibility test, new loss,
alignment correction, or gradient clipping is introduced. The encoder still
has geometric context; `self` disables cross-point attention only in the decoder.

## Server launch

Select a currently free GPU; `2` below is an example, not an availability claim.

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c http.version=HTTP/1.1 -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_center_drift_lr1e4_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_center_drift.sh "$OUT" > "${OUT}.launcher.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "${OUT}.pid"
tail -f "${OUT}.launcher.log"
```

Ctrl-C exits `tail`, not the background training. Console logs are also copied
into the experiment folder. Download **the entire OUT directory** for analysis;
the external launcher log/PID are only process-management conveniences.

## What is recorded

- Existing loss, parameter-gradient norms/updates, fixed-camera PSNR/SSIM and
  images at steps 0, 500, ..., 5000 remain available. Judge the `center_only`
  images/metrics for position recovery; the full model's attributes are untrained.
- `center_drift_updates.jsonl`: at step 1 and every 50 steps, fixed 32 training
  probe blocks and all heldout blocks are decoded immediately before and after
  **one** Adam update. This is not the displacement accumulated over 50 steps.
  It records mean XYZ movement, component RMSE, point-distance median/P95/max,
  block means/residuals, common-component SSE fractions, error before/after,
  direction cosine toward the original target, parameter updates and gradients.
- `center_drift_validation.jsonl`: absolute prediction-minus-source errors on
  those same fixed point sets at validation times; never changing supervision.
- `center_drift/drift.png`: mean signed XYZ bias, raw/debiased coordinate error,
  and single-update movement curves, separately for fit/heldout samples.
- `center_drift/fixed_points.pt` and `xyz_*.pt`: sampled original coordinates,
  block IDs, normalization, and predictions at each validation. These let us
  inspect later checkpoint-to-checkpoint drift without retraining.
- `training_state_last_center.pt`: matching last-step model/Adam/RNG state,
  saved before the trainer exports best-validation weights. Existing final
  `codec.pt` selection semantics are unchanged.

All position diagnostics are in **world units**, not normalized feature units.
`rmse` means sqrt(mean squared XYZ components), while `distance_p95` measures
Euclidean point distance. They differ by definition.

For displacement/error vectors d_i, common energy fraction is
`N * ||mean(d)||² / sum ||d||²`. Block-common energy includes this global common
component. Point weighting handles different-sized blocks. Removing means is
an **oracle diagnostic**, never a model correction or an improved render PSNR.
These samples are not sufficient to claim a whole-scene rigid transform.

To narrow down the update source, the diagnostic also evaluates the updated
decoder using pre-update encoder latents. Total displacement is decomposed into
`new_decoder(old_latent) - old_prediction` and
`new_prediction - new_decoder(old_latent)`. This is an ordered decomposition;
the latter includes nonlinear interactions and is not independent causal proof.

## Reading the results

Large common movement with alternating mean-error signs suggests shared-path
update oscillation; large block means with a small global mean suggests local
drift cancelling across regions. Decoder-only versus encoder-under-new-decoder
terms help decide which path deserves a lower LR or better conditioning later.
Large residual error even after mean removal means a translation correction
alone would not solve precision. A smaller LR is not assumed to fix any of these.

This run can be compared descriptively with the previous 2e-4 random-start run;
it is **not** a matched continuation or a multi-seed statistical claim. Diagnostics
do not contribute to loss, gradients, point selection or validation selection;
they add inference overhead. Logged training step seconds exclude post-update
diagnostic inference, so use wall time for total runtime comparisons.
