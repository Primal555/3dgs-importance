# Camera-supervised initialization → complete-scene rendering

This opt-in flow addresses the XYZ ablation: replacing predicted XYZ improved
rendering, but decoded attributes at source XYZ still remained substantially
worse than the original PLY. Neither more centered-Gaussian supervision nor
fixing position alone is assumed sufficient. Codec architecture, log-covariance
representation, noisy payload and individual Gaussian tiers are unchanged.

## Two objectives, separate phase plots

`camera_init` trains from random weights:

```
received symbols -> decoder -> predicted XYZ + predicted attributes/covariance
                                  |                    |
                      camera projection/depth     source XYZ + predicted attrs
                                  |                    |
                            geometry labels       full-scene rendering -> RGB MSE
```

For each sampled camera, source XYZ selects points inside its frustum. This is
NOT occlusion visibility. Predicted points outside the frustum or behind the
camera are still supervised. Targets and camera parameters are detached.

- Pixel coordinates use focal lengths derived from the loaded image size/FOV.
  `L_pixel = mean SmoothL1((predicted_pixel - source_pixel) / 2)` (both axes).
- Relative camera depth disambiguates equal image rays:
  `L_depth = mean SmoothL1((predicted_z - source_z) / (0.1 * source_z))`.
- Projection divides by `max(predicted_z, 0.1 * source_z, camera_near)` to avoid
  a singularity when random predicted depth is zero. This is a training-only
  safe surrogate, not a receiver operation or an exact projection behind camera.
- `L_camera_init = mean_views [RGB_MSE(render(source_XYZ, decoded_attrs),
  render(source_PLY)) + 0.01 * (L_pixel + L_depth)]`.

The **2 pixel / 10% depth scales and 0.01 coefficient are explicit engineering
hyperparameters**, not literature-derived optimal weights. Pixel scale depends
on the chosen image resolution. These terms do not sum separate manually
weighted opacity/SH/scale/rotation errors. Source-frustum counts, unweighted
pixel/depth errors, weighted geometry contribution, RGB loss, scene-gradient
norms and parameter branch gradients/actual updates are logged. Bounded
SmoothL1 residual slopes do NOT prove bounded network gradients or solve all
gradient-conditioning problems.

Rendering attributes at source XYZ supplies a meaningful spatial scaffold for
cold-start attribute learning, including full-scene compositing and occlusion.
Source coordinates NEVER enter the decoder. The teacher render has zero
gradient to predicted XYZ; the camera objective supplies that gradient. Shared
codec parameters receive both terms. A camera with no source points in its
frustum contributes zero geometry loss; the count makes that case observable.

`render` then uses **only mean multiview RGB MSE of the complete predicted
scene** against the original PLY rendering. No teacher XYZ is substituted in
the training image, and no parameter/projection auxiliary is evaluated.
Weights continue from `camera_init`; Adam moments reset at the objective
change. Both learning rates remain fixed at `2e-4`. No clipping or automatic
learning-rate reduction is enabled in the launcher. The stages have independent
loss axes; the numerical objective drop at transition is not quality evidence.

The default 1000 + 1000 steps are bounded iteration-test budgets, NOT established
convergence requirements. This is full-scene training on two sampled cameras
per step, not a small subset of Gaussians. Initialization now also costs a
full-scene codec/render/backward step. Replay limits codec activation memory;
it reuses exactly the forward channel noise. GPU speed/quality must be measured.

## Receiver and resource contract

XYZ is still learned in the single JSCC payload. No per-Gaussian coordinate
side stream is added. q1/q2/q3 retain 8/16/32 complex uses; mixed layouts assign
tiers per Gaussian, not per block. Existing reliable global metadata/model
statistics/tier syntax assumptions remain; payload-only plots are not total
over-the-air cost. The new launcher does not optimize a mask policy or drop
source Gaussians; q0 is used for padding. Existing codec paths remain available
for comparison, but no old checkpoint or bootstrap is selected implicitly.

## Launch on the server

Activate the `maskgs` environment and pull the commit first. Pick an available
GPU according to the current server state; GPU 2 below is only an example.

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
OUT="$PWD/output/truck_camera_scene_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_camera_scene_training.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

Set `PLY` and `SCENE` if the launcher defaults do not match your data. The script
requires an explicitly selected GPU and a new output directory. It never uses
an inherited `INIT`. Override `CAMERA_INIT_STEPS`, `RENDER_STEPS`, `LR`,
`BLOCKS_PER_BATCH`, `RESOLUTION`, `VALIDATE_EVERY`, `SAVE_EVERY`,
`CAMERA_GEOMETRY_WEIGHT`, `CAMERA_PIXEL_SCALE`, `CAMERA_DEPTH_SCALE` as needed.
`RENDER_STEPS=0` isolates initialization, with the same rendering validation.

## What to examine (all under OUT)

- `validation_images/000500/`: Photo | original PLY | **complete decoding** |
  amplified error. q1/q2/q3/mixed, all four held-out validation cameras.
- `teacher_xyz_images/000500/`: same noisy decoded attributes, replacing only
  XYZ. **Diagnostic/oracle, not receiver output**. No second decode/noise draw.
- `validation.jsonl`: PSNR/SSIM/MSE/L1 relative to original PLY and photographs;
  both complete and teacher-XYZ variants, per camera/trial and aggregate values.
- `charts/validation_quality.png`, `teacher_xyz_gap.png`,
  `camera_geometry.png`, `training_objectives.png`, `optimization.png`,
  `validation_metrics.csv`; created at completion, or manually with
  `python -m gaussian_jscc plot-stats --training "$OUT" --out "${OUT}/charts_live"`
  (use a new charts output directory).
- `loss.jsonl`, `training.json`: objectives, phase, config, gradient/update
  diagnostics; `codec_500.pt` etc. checkpoints every 500 steps.
- `codec_best_camera_init.pt`, `codec_best_render.pt`: separately selected by
  **complete decoded source-render MSE**, averaged over four layouts. Teacher
  XYZ never selects a checkpoint. `codec_end_*` and `codec.pt` preserve end/last
  weights; they are not silently called best. See `selection.json`.

Validation runs at step 0, each 500 steps and phase ends, with fixed cameras,
tiers and noise seeds (two trials). It uses held-out cameras from the training
camera split, so it is selection/diagnostic validation, NOT an independent final
test or cross-scene generalization evidence. No separate sibling render-history
run is needed. At a phase boundary identical weights may be evaluated twice;
plots deduplicate that step. Track the complete-vs-teacher gap as well as their
absolute levels: a narrowing gap alone can also mean the teacher result worsened.

## Local verification scope

`tests/test_camera_scene.py` checks pixel/depth behavior, detached teachers,
behind-camera gradients, teacher render gradient separation, actual codec
AWGN mixed-tier direct/replay/checkpoint gradient equivalence, unchanged actual
validation scores, single-decode oracle pairing, and the random-start two-stage
training/checkpoint/image/plot workflow. CPU tests use a differentiable synthetic
renderer: they verify code and gradient plumbing, not Truck PSNR or CUDA rasterizer
performance. Existing CUDA-only tests remain conditional on a usable GPU.
