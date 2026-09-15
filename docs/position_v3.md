# Position-v3: position learning and gradient isolation

This is an explicit, opt-in codec upgrade after the attached/detached diagnostic.
It changes the XYZ output parameterization and reconstruction objective, not the
symbol layout. It does **not** add per-Gaussian coordinate side information or
default to detaching attribute context from XYZ.

## Why these changes

In `context_xyz_gradients_20260915_165514`, detaching the attribute-context XYZ
path produced very similar attribute-training gradients and losses. At the
probed steps, `position_seed.weight` accounted for roughly 75–90% of the squared
total gradient norm. That points at position supervision and the output head,
not context feedback as the principal cause in that run. These are observations
from one scene/checkpoint; that initializer used `physical_v1`, not `balanced_v2`.

The previous position objective was `log(1 + source_scaled_distance_squared)`.
For large distance its correction slope decreases; normalization by very small
source Gaussian scales can also create large derivatives at intermediate error.
Frequent clipping shows gradients are large relative to the chosen cap. It does
not alone prove exploding Adam updates or explain the poor render quality.

## Architecture and objective

| Part | Previous geometry-first | Explicit position-v3 |
|---|---|---|
| XYZ head | Linear(hidden), then sigmoid | LayerNorm(hidden), then `0.5 + 0.25 * Linear(...)` |
| Coordinate system | Per-axis global bounding box | Same global bounding box |
| Position loss | Log source-scaled distance | Bounded-slope SmoothL1 plus large-error penalty |
| Attribute context | Decoded XYZ grid context | Unchanged; attached gradients remain enabled |
| Clipping in supplied script | One global cap | Independent caps on six disjoint parameter groups |
| Attribute-stage tiers | One sampled layout | q1/q2/q3 on the same block and SNR; mean gradients |

The LayerNorm has no trainable gain and uses epsilon `1e-5`. The affine head
avoids sigmoid saturation and bounds the feature norm entering its weight
gradient. This is **not a local-anchor or local-residual coordinate codec**:
the global-bbox representation and its outlier/dynamic-range limitations remain.

Let `e_ij` be predicted minus source coordinate in per-axis bbox units. Position
loss per Gaussian is:

```text
base_i = mean_j SmoothL1_beta(e_ij, 0)
tail_i = mean_j SmoothL1_beta(max(abs(e_ij) - margin, 0), 0)
L_xyz  = mean_i(base_i + tail_weight * tail_i)
```

Defaults: beta=0.001, margin=0.01, tail_weight=2, geometry_weight=10.
The margin is **1% of each coordinate-axis span**, not a pixel tolerance,
Gaussian radius, P95 threshold, or 1% of scene diagonal. All these values are
engineering choices, not paper-reproduced optimum weights. SmoothL1 is linear
beyond beta, so very bad coordinates retain corrective force instead of the
old log penalty's decreasing slope. The tail term increases that slope beyond
the margin; it does not directly optimize P95.

Position loss uses the **unclipped** prediction. Thus an output outside `[0,1]`
still gets a recovery gradient. Receiver conversion/rendering continues to clamp
XYZ to the bbox for safety. Evaluation records both clipped and unclipped RMSE
plus the out-of-bounds fraction, so the safety clamp cannot hide divergence.

Attributes retain `balanced_v2`: log-covariance shape loss, sorted log-scale
guard, alpha/logit opacity loss, and normalized DC/SH reconstruction. Weights:
shape=0.25, scale=1, opacity=1, DC=1, SH=0.25.

```text
Attribute objective = 10*L_xyz + .25*L_shape + L_scale + L_opacity + L_DC + .25*L_SH
Render objective    = L_render + attr_weight * Attribute objective
L_render           = .8*L1(image, target) + .2*(1-SSIM(image, target))
```

The supplied script uses `attr_weight=1`, source-PLY render targets, and separate
attribute/render learning rates. Do not compare the numerical total loss to the
old 9–11 scale: its definition and units changed.

## What this can and cannot do about large gradients

For the direct position objective, the derivative per coordinate per row is
bounded by `(1 + tail_weight)/3` before batch averaging and geometry weighting.
Combined with the fixed-gain normalized head, its **geometry-only** weight-gradient
Frobenius norm is bounded by:

```text
geometry_weight * (1 + tail_weight) * 0.25 * sqrt(hidden / 3)
```

For hidden=96 and the defaults this is about 42.43. This bound applies only to
the direct position term, **not** to the head's total gradient through attribute
context/rendering or to upstream parameters. It is an analytic guard against
the observed direct position-head amplification, not a global stability proof.

Branch clipping groups are geometry encoder, geometry decoder, attribute
encoder, attribute decoder, shared encoder, and conditioning. Each existing
group is capped at 1 in the script. An isolated large geometry-decoder gradient
no longer scales down all attribute parameters. Shared encoder/conditioning
still carry coupled gradients. The overall clipped norm may reach `sqrt(6)`;
this is intentionally **not** the same optimization constraint as global cap 1.
Adam may also respond differently to the changed gradient history.

Nonfinite gradients abort before the optimizer update. Logs record pre/post
clipping group norms every step; every `profile-every` steps they additionally
record the largest parameter gradients and actual relative Adam updates.
These checks help identify instability but cannot prevent every possible NaN or
large finite gradient from covariance, quaternion, grid, or render operations.

## Initialization and compatibility

Old checkpoints still decode with their original head and retain their model
hashes. A change from sigmoid to normalized affine requires
`--upgrade-position-head`. Only `position_seed.weight/bias` are reset: small
random weights and a trainable bias initialized near the source XYZ median.
This median becomes ordinary trained model parameters, not a new packet field.
Other codec weights, attribute normalization, and rate tables are retained.
As before, model distribution cost is outside the per-scene payload counter.

This is **not function-preserving**: the new position head initially forgets the
old coordinate mapping and needs training. A loaded v3 checkpoint does not reset
its head again. `--init` always starts a fresh optimizer/schedule, not exact resume.

Symbol counts and geometry/attribute prefixes do not change. q0 remains absent;
positive q carries XYZ in the existing payload. Old reference models and new
models must not be interchanged at the receiver; the model hash checks that.
New checkpoints require this code version. CLI defaults remain `balanced_v2`;
use the explicit profile/script rather than assuming a pull changes objectives.

Both codec-only and joint-mask training support the new profile/head and clipping
options. All-tier coverage is a codec-pretraining option; joint mask training
continues to optimize its chosen tiers and rate objective.

## Run on the server

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only origin main

# Choose an existing geometry-first checkpoint; the script enforces v3 even if
# that checkpoint recorded physical_v1. The source checkpoint is never replaced.
INIT="$PWD/output/truck_geometry_first_20260915_095729/codec.pt"
OUT="$PWD/output/truck_position_v3_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_codec_position_v3.sh \
  "$INIT" "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

Override `INIT` if using another migrated checkpoint. `PLY`, `SCENE`, and
`PYTHON_BIN` may be set as environment variables. Output must not already exist.

The script allocates 3000 attribute optimizer steps and 300 render steps.
These are adjustable run budgets (`POSITION_STEPS`, `RENDER_STEPS`), not evidence
of sufficient convergence. Each attribute step sequentially runs three tier
graphs and averages their gradients before one update. Thus it costs roughly
three attribute forwards/backwards, without retaining all three graphs. The
full-scene render phase still samples uniform/mixed layouts and is not tripled.
Training remains under random AWGN SNR 0–20 dB. Uniform all-tier attribute
coverage does not guarantee monotonic quality or exhaustively train mixed layouts.

## Results to inspect

- `training.json`: actual head/profile/weights, initializer, rate table, clipping.
- `loss.jsonl`: components, scalar contributions, per-tier attribute-step metrics,
  group clipping, and profiled actual updates. Random-step metrics are not fixed
  validation curves.
- `position_evaluation.json`: step 0, every 500 steps and phase/final boundaries;
  q1/q2/q3, no-noise at conditioning SNR10 and AWGN SNR0/10/20. Same up-to-eight
  evenly spaced source blocks and fixed noise seeds, with no impact on training
  RNG. Step 0 is **after** explicit head reset, not old-checkpoint performance.
- `charts/training_fixed_positions.png`: fixed XYZ RMSE and Euclidean P95 by tier.
- `charts/training_gradient_groups.png`: group norms before/after clipping and
  actual relative updates; the tiny newly initialized head can have larger
  relative changes without huge absolute updates.
- Existing phase-separated objective, unweighted-component and weighted-
  contribution charts remain; data are also exported as CSV.
- `codec_*.pt`, `codec.pt`: periodic and final weights. No automatic best-model
  selection or early stopping is introduced by this upgrade.

Fixed-block position checks are not test-view rendering or unseen-scene
generalization. Final acceptance still needs the existing `benchmark_codec.py`
render evaluation at matched tiers/SNR, source-PLY reference and position/attribute
hybrids. A lower new training loss alone does not establish improved communication
quality. In particular, watch whether q1/q2 P95 improve along with q3, and whether
better positions translate into render quality without attribute deterioration.

## Local verification

`tests/test_position_v3.py` covers explicit migration, saved-head reuse, no source
checkpoint overwrite, finite/zero-identity loss, bounded non-vanishing position
gradients, head input-scale resistance, independent clipping, all-tier gradient
averaging, fixed-evaluation RNG isolation, replay/checkpoint gradient agreement,
CPU train/packet/receive, and generated diagnostics. Historical compatibility and
joint training are also exercised by the existing suite. These CPU numerical
contracts are not a 4090 quality or convergence result.
