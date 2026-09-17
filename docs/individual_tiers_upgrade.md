# Stable groups and per-Gaussian resource control

## Activation and compatibility

Use `--position-head reference_v6 --individual-tiers --loss-profile robust_v4`.
When changing from another position head also pass `--upgrade-position-head`.
`individual_tiers` is an explicit checkpoint/wire flag. Its absence preserves
the old waveform and model identity. The new flag changes the waveform even
when all tiers are identical; old weights are an initializer, not a validated
new codec. Adam starts fresh with `--init`.

## Four changes

1. **Stable group identity.** Groups are fixed intervals of source rows within
   each Morton-ordered packet (default 256 rows / group, 4096 rows / packet).
   q0 is removed only inside an interval. Transport retains the full tier map
   until grouping; no new IDs, clean coordinates, or reference metadata are
   sent. Source order and the existing tier map let a fresh receiver reproduce
   the grouping. Dense groups are processed together to avoid a launch loop
   per geometry group. Within-group phase ordering still changes with q0;
   this remains a discrete, biased-ST decision, not an exact gradient.

2. **Tier-independent base and personal enhancement.** Every retained point
   uses the same q1 geometry base. In dense groups the phase-reference energy
   and base-detail gain are independent of q1/q2/q3 choices. q2/q3 add bounded
   offset observations for that point. Enhancement gains are receiver-known;
   precision-weighted combining merges base and enhancement observations.
   One real slot per nonempty enhancement layer completes its energy budget.
   It is counted, not free information. These completion slots deliberately
   trade coding efficiency for bounded, identifiable amplitude.

   Fewer than 24 retained points cannot reliably populate the multiscale phase
   reference. Such a group uses a shared analog q1 reference plus the same
   personal enhancement mechanism. Its gain is known, and the base completion
   slot is not a pilot. A one-point group is supported but is not promised good
   noisy-channel precision. q0 can change a group's reference/fallback, but
   cannot move another group's boundaries.

   Attribute symbols use **per-point** tanh + RMS normalization with energy
   floor 0.1. Amplification is bounded by sqrt(10), including all-drop ST
   layouts. Above the floor, mean complex energy is one; below it, power is
   intentionally lower. `stats.json` reports actual mean transmitted energy.
   Encoder context still permits neighbors' tiers to change features; there
   is no claim of complete statistical independence between points.

3. **Mixed tiers dominate training.** With `--tier-training all`, each step
   averages three randomly skewed mixed layouts and one sampled uniform
   positive tier. Mixed layouts include q0. Uniform coverage is stochastic,
   not all three uniform tiers every step. Four sequential backward passes
   retain bounded activation memory. The mask still has one `[4]` logit vector
   per original Gaussian; no group-tier parameter is introduced.

4. **Robust attribute objectives.** `robust_v4` keeps the v6 position objective
   and persistent clean-position constraint. For attributes, use the fixed
   checkpoint statistic `u = max(1, RMS(std(log_scale)))`. Shape compares the
   traceless log-covariance difference divided by u, using the radial penalty
   `2 * (sqrt(1 + ||error||_F^2 / 9) - 1)`. Scale uses SmoothL1 on sorted
   log-scale differences divided by u. Removing trace avoids penalizing pure
   isotropic size twice; anisotropy still overlaps partly between objectives.
   The radial penalty keeps joint-rotation invariance and avoids unbounded
   quadratic residual slopes. These are engineering objectives, not weights
   reproduced from a paper. They do not bound the entire network Jacobian,
   guarantee no clipping, or guarantee perceptual quality.

## Running and verifying

Local real-PLY CPU diagnostics (new output directory required):

```bash
python scripts/test_individual_tiers_local.py \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --checkpoint output/reference_v6_bounded_local_20260916/codec.pt \
  --out output/individual_tiers_check --steps 20
```

This records paired old / wire-only / wire+loss gradient probes, mixed-layout
training, and a mask-joint attribute proxy. It does not render test cameras.
The diagnostic checkpoint is not a deployment recommendation.

Server training, fixed 10 dB, full codec parameters (not geometry-only):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_codec_individual_tiers.sh \
  /absolute/path/to/codec.pt /absolute/path/to/new_output
```

The script defaults to 3000 attribute steps and no automatic render phase.
`STEPS`, `LR`, `RENDER_STEPS`, `PLY`, `SCENE`, and `PYTHON_BIN` are configurable
environment variables. A render phase requires the dataset and CUDA renderer.
Do not infer quality from a lower loss after changing the loss definition.

Unit coverage: noiseless reconstruction; exact total geometry energy; q0
group stability; other points' geometric payload unchanged on positive-tier
upgrade; mixed dense/packed and fresh receiver agreement; all-drop finite
gradients; local attribute power; replay/checkpoint gradient agreement.

## Local results (2026-09-16, CPU)

Final diagnostic directory: `output/individual_tiers_final_local_20260916`.
Initializer: `output/reference_v6_bounded_local_20260916/codec.pt`.
Two real Truck blocks (0 and 107), 4096 points each, AWGN 10 dB, paired low
and mixed layouts. Attribute-symbol encoder gradient norms before clipping:

| Block / layout | Old | Wire changes only | Wire + robust objective |
|---|---:|---:|---:|
| 0 / low | 29.57 | 39.43 | 7.97 |
| 0 / mixed | 48.14 | 48.82 | 5.12 |
| 107 / low | 28.54 | 27.56 | 7.02 |
| 107 / mixed | 4.37 | 3.45 | 1.80 |

The improvement is primarily attributable to the objective change in these
probes, not proof that changing normalization alone fixes gradients.
Twenty subsequent mixed-layout steps had total pre-clip norm 1.54–7.15.
The attribute encoder still clipped in 19/20 steps (median 3.31, max 5.93),
so clipping has NOT been eliminated. The joint attribute proxy gave nonzero
mask gradient norm 0.01794 with beta=0; this is not a rendering evaluation.

Quality tradeoff before adaptation: low-tier position RMSE was essentially
unchanged (2.91375 -> 2.91303; 0.26807 -> 0.26803). Mixed-layout RMSE worsened
in these same probes (2.25501 -> 2.63759; 0.21511 -> 0.24514). Removing the
extra high-tier energy that formerly helped the shared reference has a cost;
fixed grouping also changes references under dropping. Do not call these
changes an overall quality improvement or launch a long run on that premise.

Intermediate directories `individual_tiers_local_20260916` and
`individual_tiers_layers_local_20260916` contain superseded development
waveforms. Do not load their diagnostic checkpoints with the final code.
The discarded constant-gain geometry variant had unacceptable position
regression; the final design retains the tier-independent normalized base.
