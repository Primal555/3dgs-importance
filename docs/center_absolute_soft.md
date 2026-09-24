# Continuous absolute XYZ training (replaces strict rejection)

The model and world-center distance loss remain the `52edc1c` pointwise absolute
XYZ baseline. No explicit block centroid, new geometry representation, or loss
weight has been introduced. The decoder remains self-only with affine readout.

## Update rule

Use `--center-update-policy soft`, NOT `--center-step-guard`.

- Exactly one Adam proposal per training batch. No extra loss forward, no
  same-batch descent requirement, no loss-based rejection, no momentum reset.
- During the first 100 steps, collect proposal magnitudes without rescaling.
- For encoder and decoder separately, measure `r = ||delta|| / ||parameters||`.
  History stores the last 100 **unscaled** proposal ratios divided by the
  scheduled LR. The next cap is `3 * median(history) * current_lr`.
- If `r` exceeds the cap, scale that group's displacement by `cap/r`; otherwise
  retain the full proposal. Adam moments/counters still advance once. A zero
  historical reference does not freeze a newly active group. History uses raw
  proposals so protection cannot recursively shrink its own reference.
- Nonfinite gradients, optimizer state, or proposals are errors. Restore a
  failed proposal's weights/optimizer state and stop; never silently advance
  training with corrupted values. Resume from the last saved training state.
- Scheduled LR stays at 1e-4 through the first half of the 5000-step budget,
  then smoothly follows a cosine to 2e-5 at step 5000. B/C schedules unchanged.

100 steps, multiplier 3, halfway decay and final ratio .2 are transparent,
configurable **engineering starting points**, not thresholds calibrated from
the user's newest server logs or claimed optimal values. Small scales can still
slow training, so log actual update norms and warn if scales below .25 or zero
updates exceed 10% at a 100-step check. No guarantee of monotonic validation,
PSNR, or absence of coordinate drift. Whole-module norm ratios do not bound
each parameter or output coordinate independently.

## Launch

Fresh random initialization, 5000 center-only steps; no old weights. Images,
PSNR/SSIM and checkpoints every 500 steps, fixed-point drift every 50 steps.
All generated artifacts live in one output directory. `full` uses untrained
attributes in this experiment: assess learned XYZ with `center_only` renders.

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_center_absolute_soft_$(date +%Y%m%d_%H%M%S)"
# Choose an actually available GPU; 2 is only an example.
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_center_absolute_soft.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

`charts/center_soft_updates.png` displays LR, scale, proposed/actual relative
updates, and limiting/zero-update frequency. `loss.jsonl` records these per step;
`summary.json` includes per-module counts. Controller history and phase step
are checkpointed, so exact resume reproduces scaling and scheduling.

The old `test_center_absolute_guard.sh` filename now forwards to this new
experiment (unset retired `STEP_GUARD` environment variable). Strict-guard CLI
support remains only for explicit historical reproduction/exact resume. A
previous guarded run **does not switch policy on resume**, because stored
arguments win. Use the new random-start script to test this policy.
