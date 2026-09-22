# Teacher-axis position experiment (representation only)

This is an opt-in, deliberately aggressive test, not a replacement for the
research baseline. The encoder, Transformer decoder, latent dimension, logcov
output, shape objective and centered appearance objective are unchanged. The
receiver gets no teacher axes, coordinates, grouping centers or extra payload.
Both communication modules stay frozen; clean training bypasses them completely.

## Exact change

For source Gaussian i, let R_i be its principal axes (columns), s_i its three
standard deviations, and d_i = predicted XYZ - source XYZ in world coordinates.

```
s_floor = percentile(all source Gaussian axis standard deviations, 1)
s_eff_i = maximum(s_i, s_floor)
u_i = (R_i.T @ d_i) / s_eff_i
L_position = mean_i(sqrt(1 + dot(u_i, u_i)) - 1)
L_total = (L_position + L_logcov_shape + L_centered_appearance) / 3
```

The three changes are anisotropic teacher-axis normalization instead of a
scene-scale denominator, a once-fitted small-scale floor, and replacement of the
old coarse/fine position terms rather than adding another auxiliary penalty.
The floor is fitted once to the original PLY, not per minibatch and not from
predicted shapes. Source eigendecomposition is detached, in float64, and uses log
covariance so very thin splats do not require inverting an ill-conditioned
covariance matrix. No eigenvector derivatives enter training. Repeated teacher
eigenvalues do not cause an eigenvector-backward singularity.

The 1st percentile, unit pseudo-Huber transition, and retained three-term mean
are explicit engineering choices, **not a paper-derived optimum or pixel-error
calibration**. The floor's world value and affected fraction are saved. Override
the percentile with `AXIS_FLOOR_PERCENTILE`, or set an explicit world-unit floor
with `AXIS_FLOOR_WORLD`. The latter wins and is recorded as an explicit override.

This does not redesign the XYZ head: it still predicts globally normalized XYZ.
It corrects the supervision scale, not the decoder coordinate parameterization.
Existing global feature statistics/bbox still use the whole input scene; heldout
blocks measure scene-fitting generalization, not unseen-scene generalization.

## Risks deliberately kept visible

- Very thin source axes can produce much larger XYZ and network gradients. A
  pseudo-Huber linear tail is not a proof against gradient explosion. The
  per-point world-XYZ gradient bound is `1/s_floor` before mean and `/3`; network
  parameter gradients also depend on the decoder Jacobian and global span.
- The floor relaxes supervision below that scale. Native (unfloored) ellipsoid
  errors are recorded alongside effective errors so this cannot hide damage.
- An ellipsoid's thin-axis tolerance is not necessarily the screen-space
  tolerance of the training cameras. Position fitting can improve without PSNR
  improving, and may compete with attribute fitting through shared parameters.
- Raw loss/gradient magnitudes are not comparable to the old objective. Adam can
  partially cancel a common gradient rescaling; larger gradients alone are not
  evidence of better updates or convergence.
- Random weights, fixed LR `2e-4`, no clipping, no communication or render-loss
  training remain the defaults. Nonfinite loss/gradients abort rather than silently
  changing the optimizer. Finite large gradients are logged, not suppressed.

## Server launch

Choose an actually free GPU; `2` below is only an example. Copy only the commands.

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

OUT="$PWD/output/truck_teacher_axis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
env -u INIT CUDA_VISIBLE_DEVICES=2 REPRESENTATION_STEPS=5000 \
  ADAPTER_STEPS=0 JOINT_STEPS=0 LR=0.0002 RENDER_EVERY=500 \
  nohup bash scripts/test_teacher_axis_representation.sh "$OUT" \
  > "$OUT/console.log" 2>&1 &
echo $! > "$OUT/run.pid"
tail -f "$OUT/console.log"
```

The launcher refuses an inherited initializer or enabled later stages. It uses
the same full-scene PLY, random seed, block split and architecture as
`scripts/test_representation_codec.sh` by default. For a paired control run use
that original script with `POSITION_OBJECTIVE=scene-scale` and a new output
directory; keep all other settings identical. Do not judge new-vs-old raw loss.

For exact recovery after an interruption (not a new trial):

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m gaussian_jscc train-representation \
  --resume "$OUT/training_state.pt"
```

Resume retains the fitted floor, model, Adam state, split, RNG and stored options.

## What to inspect

Everything is under the one output directory:

- `training.json`: exact objective, resolved floor, original-PLY affected axes
  and points, architecture, split, and hyperparameters.
- `loss.jsonl`: weighted position/shape/appearance values, total/module gradients,
  parameter updates, and every 10 steps each objective's gradient norm in the
  encoder and decoder. These are parameter gradients, not scalar loss shares.
- `validation.jsonl`: fixed heldout clean world-XYZ RMSE, native/effective
  ellipsoid-relative errors, within-one-ellipsoid fractions, shape and appearance.
  Reported p50/p95 are point-weighted means of block quantiles, **not** pooled
  scene quantiles. Legacy coarse/fine losses are diagnostics only.
- `render_validation.jsonl`: fixed-camera clean PSNR/SSIM against the input PLY
  render and photos separately, at step 0 and every 500 steps and final step.
- `images/000500/view_00/clean.png` etc.: recovered images; `comparison.png`
  contains labelled photo/source/clean/communication views. The communication
  path is untrained and is not the success criterion in this experiment.
- `charts/teacher_axis_diagnostics.png`: axis-relative precision, floor effects
  and weighted per-objective/module gradient norms.
- `charts/render_quality.png`: actual image-quality evolution.
- `codec_*.pt`, `training_state.pt`: periodic weights and exact recovery state.
  `codec_best_representation.pt` is selected by heldout objective, not render PSNR.

Success requires better **clean images/PSNR together with finer positioning**,
without attribute collapse. A falling anisotropic loss alone is insufficient.

## Local checks

Unit tests cover the analytical anisotropic value/gradient, rotated axes, source
detach, repeated eigenvalues, floor behavior, world-unit/bbox invariance, unchanged
attribute gradients, replacement (not addition), unchanged communication weights,
and exact interrupted continuation with identical fitted-floor metadata.

CPU paired fitting check (no CUDA render claim):

```bash
python scripts/check_representation_local.py \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --out output/axis_cpu_trial --representation-steps 300 \
  --adapter-steps 0 --joint-steps 0 --position-objective teacher-axis
```

Repeat with `--position-objective scene-scale` and a different output directory.
This samples 32 Morton blocks / 8192 real Gaussians, fits global statistics and
the floor to that subset, uses hidden size 48 and holds out four blocks. It is
not a full-scene quality experiment; the server defaults remain hidden size 96.

### Completed paired CPU check, 2026-09-22

Both runs used identical initial weights (verified tensor by tensor), identical
8192-point source/split and 300 representation steps. Communication weights were
verified unchanged. No parameter clipping or LR reduction was used.

| Metric at step 300 | Existing scene-scale control | Teacher-axis experiment |
| --- | ---: | ---: |
| Fitted 7168 points, pooled world XYZ RMSE | 4.2293 | 6.3874 |
| Heldout 1024 points, pooled world XYZ RMSE | 18.2132 | 18.0723 |
| Heldout effective ellipsoid error, pooled p50 | 5925.98 | 7073.08 |
| Heldout logcov shape objective | 9.5372 | 9.9722 |
| Heldout centered appearance objective | 0.06651 | 0.08035 |
| Training total gradient L2, median | 67.56 | 820052 |
| Training total gradient L2, maximum | 166.25 | 5060126 |

For this comparison both checkpoints were evaluated using the **same** fitted
floor `2.4538572215e-6` world units. Source minimum axis was `6.0067e-8`;
1.001% of axes / 2.454% of points were affected. The p50 in this table is pooled
across all heldout points, unlike the block-averaged quantiles in training logs.

The aggressive recipe did **not** improve overall reconstruction in this short
check. The tiny heldout world-RMSE improvement does not outweigh worse fitted
precision, relative precision and attributes. At the final profiled step, decoder
gradient norms from position / shape / appearance were approximately
`521875 / 7.12 / 0.0768`, demonstrating severe objective-gradient imbalance.
All gradients stayed finite; this is evidence of scale amplification, not proof
of a dynamically diverging gradient explosion. These results justify retaining
this as an isolated diagnostic option, **not adopting it as a new default**.

Local artifacts: `output/teacher_axis_cpu_check_20260922/run`,
`output/scene_scale_cpu_control_20260922/run`, and
`output/teacher_axis_cpu_check_20260922/paired_report.json`. No CUDA rasterizer is
available locally, so there is no local PSNR or image-quality improvement claim.
