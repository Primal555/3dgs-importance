# Three prefixes + learned drop: full importance-allocation experiment

Entry: `scripts/train_progressive16_mask_full.sh`. This is an experimental
scene-specific categorical probability table, **not** a generalizing neural
allocation predictor and not a claim to reproduce MaskGaussian's CUDA mask
gradient estimator. The previous dense-eight-prefix experiment remains intact.

## Decisions and prior

Each original PLY row owns four trainable logits. Softmax gives probabilities
for q0/drop, q1/8, q2/16 and q3/32 complex attribute symbols. A local block can
contain all four decisions. q0 contributes no Gaussian attributes or coordinate
row to transmission or rendering, but the scene's tier-map syntax still has a
cost. Source rows/bbox/block framing remain fixed; dropped rows are not compacted
before attention. Codecs/weights are shared with the receiver in advance.

Initialization uses the existing prototype: `P(drop)=1-p`, positive probabilities
`p * [0.1, 0.3, 0.6]`. This is an engineering initialization, NOT an inferred
quality benefit. Without a historical existence prior, p=0.99 for all points;
it is a conservative high-retention/high-tier start, not learned importance.
Initialization therefore need not have any hard q1/q2 decisions.

`EXISTENCE_PRIOR=auto` reads `masks_0/masks_1` from the SAME input PLY if present.
The local MaskGaussian implementation uses log-softmax logits and column 0 for
existence, so the prior is `softmax([masks_0,masks_1])[0]`, NOT opacity or sigmoid
of an arbitrary column. If those fields are absent, the log explicitly reports
no historical prior. `EXISTENCE_PRIOR=ply` fails instead of falling back.
An explicit NPY must be probabilities matching original PLY rows. Equal length
does not prove alignment; do not copy an array from the unpruned point cloud.
Pruned/exported PLY files may no longer contain existence probabilities.

Historical existence answers "is keeping this Gaussian useful for that earlier
reconstruction/pruning objective?" It does not directly measure q1->q2->q3
improvement at the current SNR. The prior initializes selection but is neither a
fixed rule nor an extra loss forcing final choices to follow it. Shape/opacity
and rate demands can differ between points with similar existence probability.

For context, [MaskGaussian](https://maskgaussian.github.io/) learns probabilistic
existence via masked rendering. This prototype extends the decision space to
communication lengths and uses a different, score-function estimator.

## Training

1. 5000 attribute bootstrap updates: historical local-response reconstruction,
   fixed reliable 16-bit quantized XYZ with lossless delta/zlib integer coding.
2. 5000 multiview full-scene source-render MSE updates. Both stages rotate
   q1/q2/q3/mixed; mixed includes 5% random drop for exposure to missing peers.
3. 3000 allocation/joint updates. First 500 update only the table with codec
   unchanged (no codec backward); remaining 2500 update codec and table together.
   `MASK_ONLY_STEPS` is INCLUDED in `JOINT_STEPS`, not additional updates.

Defaults: fixed 10 dB AWGN, constant codec LR=1e-4, mask LR=1e-3, replay backward,
64 blocks/batch, two training views and two independently sampled masks/update.
The third stage is more expensive than an ordinary render update. Progressive
mask samples share full-slot channel noise (RNG-isolated) and views, reducing
unrelated noise in their comparison without replacing hard decisions by a
soft-opacity approximation. Two masks are still a noisy estimator at 880k points.

The task target is always the complete original PLY render. Dropping a point
does NOT remove its reference pixels from the loss. Empty scenes receive the
background-image loss, not zero loss. XYZ stays fixed for retained points.

The objective is:

`E_mask,channel,views[MSE] + beta * E_mask[C_proxy / C_full_q3]`.

- Attribute payload expectation is computed analytically from probabilities;
  its gradient penalizes expensive tiers directly.
- Actual compressed XYZ bytes and compressed packed-tier-map bytes depend on
  the sampled discrete layout. Their costs enter the REINFORCE reward with
  an independent-sample leave-one-out baseline; they are not logging-only.
- Normalize by the fixed full-q3 cost per original Gaussian. All averages use
  ORIGINAL source count, not just the number left after dropping.
- Cost uses 2 assumed net digital information bits/complex use for side streams.
  This is accounting, not simulated FEC or demonstrated error-free transmission.
- Training proxy includes XYZ framing but excludes the packet JSON/bbox/model-ID
  header. Final packet evaluation measures full metadata and actual XYZ streams.
- `BETA=0.01` is an explicit engineering starting point, NOT a hard rate cap or
  proven optimum. Zero beta supplies no explicit incentive to save bandwidth.
  Increasing beta favors savings, potentially harming quality or dropping all
  points; decreasing it favors distortion. No artificial tier quotas are imposed.

Validation/best selection uses the **hard argmax** table's image MSE plus the
same normalized rate proxy, not the expected soft rate. Inspect both: small
probability changes can leave hard deployment unchanged. Training logs record
`mask_grad_norm`, `codec_updated`, sampled counts, rate loss and sampled rewards.
During the first 500 joint steps codec grad/update=0 is intentional; mask gradients
must still exist. Global scene-level REINFORCE can remain high-variance; a
successful software test is not proof that it learns point importance well.

## Outputs and the prior-only diagnostic

Every 500 phase updates (and endpoints):

- `validation_images/<step>/mask_view*.png`: actual hard allocation rendering.
- `allocation_history.jsonl`, `charts/allocation_history.png/.csv`: hard counts,
  shares, expected counts, entropy, confidence and changed decisions.
- `allocation_latest/tiers.npy`, `probabilities.npy`, `allocation.json`: latest
  ORIGINAL-ROW deployment map and probabilities. Prior bins/counts are included
  when available. This is **last validation**, not necessarily best checkpoint.
- `charts/existence_vs_allocation.png`: original probability intervals vs final
  learned tiers, if a real prior exists. Empty bins are zeros, not evidence.
- `validation.jsonl`: all uniform tiers, mixed, mask; mask actual proxy costs.
- When a prior exists, `prior_ranked` assigns the exact same tier counts to
  points sorted by existence probability. Equal attribute payload, NOT guaranteed
  equal total cost (coordinate compression changes). View/noise comparisons are
  paired. No thresholds or quotas are claimed optimal. Ties use original order.

The script then exports the matched `codec_best_joint.pt` / `route2_best_joint.pt`
to `deployment_best/`. Its `allocation.json` and arrays are the BEST deployment,
not the latest snapshot. Do not combine a codec and table from different steps.
`FINAL_EVAL=1` (default) sends/decodes real packets on the held-out test cameras,
saving images, PSNR/SSIM and full measured packet costs under `test_best_mask/`.
Digital overhead accounting uses code-rate1/modulation-bits2 to match the assumed
2 net bits/use; reliable delivery is still assumed. Test images are not training
or selection views. Final evaluation uses the existing clipped/display metric
pipeline, while training validation records unclipped MSE; do not silently merge
these metric definitions. NPY disk payload bytes are NOT wireless bit counts.

## Launch

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git pull --ff-only --no-recurse-submodules origin main
GPU=2 # example: verify this card is available
OUT="$PWD/output/truck_progressive16_mask_full_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" PYTHON_BIN="$(command -v python)" \
  nohup bash scripts/train_progressive16_mask_full.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

Random by default. Explicit `INITIALIZATION=checkpoint INIT=/.../codec.pt` accepts
matching progressive 0/8/16/32, quantized16 delta_zlib codecs only; this loads
weights, not optimizer continuation. To skip earlier stages set both
`BOOTSTRAP_STEPS=0 RENDER_STEPS=0`. Joint stage must remain positive for this
full launcher. Change `BETA`, `MASK_LR`, `MASK_SAMPLES`, `JOINT_STEPS`,
`MASK_ONLY_STEPS`, `VALIDATE_EVERY`, `SAVE_EVERY` via environment variables.
Use `FINAL_EVAL=0` to defer the extra final test-camera run.

All results are under OUT; shell log/PID are alongside it. Stopping tail with
Ctrl+C or disconnecting the client does not stop nohup training.
