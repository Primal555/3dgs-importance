# Fully learned Gaussian JSCC mainline

This replaces the handcrafted geometric transport direction. It is an implemented
architecture, not a claim that the position-recovery problem has already been solved.
Use `train-learned` or its alias `train`. The historical training implementations
and `train-route2` entry point have been removed; joint training uses `--joint-steps`.

## Architecture and wire contract

```text
Frozen source PLY + independent per-Gaussian q in {0,1,2,3}
  -> Morton ordering / fixed source blocks (default 256; q0 does not move boundaries)
  -> learned XYZ / scale / rotation / opacity / SH embeddings
  -> shared feature fusion + FCGS-inspired sender grid aggregation
  -> learned shared symbol head
  -> per-row prefix selection k(q) + smooth power normalization
  -> packed complex symbols -> AWGN
  -> receiver unpack using q map (absent slots are masked)
  -> local received-feature attention (window 32; alternating shifted windows)
  -> parallel learned affine XYZ / scale / rotation / opacity / SH heads
  -> remove q0 rows -> physical parameter conversion -> render
```

The sender grid uses source XYZ; the receiver never receives that grid or its
bounds. Its attention keys/values come from noisy symbol features. Decoder
context is not constructed from guessed XYZ. Deterministic sequence encoding
restarts in every fixed block: no global Gaussian-ID embedding / scene lookup table.

q1/q2/q3 may coexist inside every block. A block is a context/computation unit,
not an allocation unit. q0 has zero payload, is masked out of aggregation and
attention, and is omitted from the rendered scene. q1/q2 mean lower resource
budgets, **not smaller opacity**.

All active symbols are learned from joint geometry/attribute features. There is
no position/appearance split in the rate table and no separate geometric encoder.
Separate output heads only specify parameter semantics; the receiver does not
need a header describing how many symbols were devoted to XYZ. `return_seed`
exists for old benchmark API compatibility and returns final XYZ, not another stage.

Default rates are `(0,8,16,32)` **complex** symbols per source Gaussian. They are
engineering comparison points, not established optimal budgets or guaranteed
quality levels. The tier conditions both networks; different tiers need not be
literal truncations of the very same encoded waveform. No monotonic-quality
guarantee is imposed.

For selected real/imaginary coordinates `z`, normalization is
`z / sqrt(mean_complex_energy(z) + power_floor)`. The default floor `.01`
bounds normalization gain by 10 and yields mean complex energy **at most** one.
There are no fixed-energy filler symbols or transmitted amplitude multipliers.
AWGN variance uses unit reference power; effective measured transmit energy is
reported in packet stats. This is a power constraint, not a guarantee against
large gradients elsewhere.

Metadata still contains global bbox normalization (six floats), source count,
the compressed two-bit-per-row q syntax, configuration/model identity and
channel configuration. No individual/block coordinate reference is sent.
Headers are assumed reliably received, with their size and modeled channel
uses counted. Header channel errors are not simulated; model weights are shared
in advance and excluded from payload cost. These assumptions remain explicit.

## Objective and optimization

`learned_v1` has fixed, configurable weights. It is our engineering design,
not a claimed reproduction of a paper's empirically optimal loss.

- XYZ: SmoothL1 of bbox-normalized coordinate error divided by `xyz_loss_scale`
  (default `.05`), transition beta `.1`. Decoder XYZ is affine and **not clipped**
  to `[0,1]` before training/render conversion. `.05` sets a loss scale; it is not
  a claim that 5% positional error is acceptable.
- Shape: minimum squared distance of normalized quaternions under `q` / `-q`.
  This removes quaternion sign ambiguity, but does not implement all equivalent
  scale-axis/quaternion permutations of the same covariance.
- Scale, DC and higher SH: SmoothL1 in shared standardized attribute units.
- Opacity: alpha-space SmoothL1 plus a smaller standardized-logit term.
- Projection: source-camera image-plane error and positive-depth penalty.
  Only source-in-frustum points are supervised; erroneous predicted depth cannot
  silently exclude a point. Denominators use a source-depth-relative floor to
  limit perspective amplification. Source/cameras are training targets only.
- Render task: `.8 L1 + .2 (1-SSIM)` against the frozen input PLY's rendering.
  Original photographs are shown separately and remain a different reference.

There is no inverse tiny-Gaussian covariance XYZ multiplier, logarithmic XYZ loss
tail, old seed loss, or learned loss weight able to suppress the positional term.
The normalized XYZ loss alone does **not** establish per-primitive visual accuracy;
render/projection supervision and validation are still needed.

Codec training keeps the same reconstruction objective and coefficient across
phases. Rendering/projection contributions ramp from zero to their configured
weights without resetting the network, optimizer or learning rate. The phase
changes from sampled local blocks to a full-scene objective, so the data
distribution still changes; phase-separated plots remain necessary.

Gradient clipping defaults to `none`, with finite-gradient checks before updates.
`--clip-mode global|branch --clip-norm ...` is an explicit option. Logs contain
pre/post group norms, actual Adam update norms and relative updates. Large norm
alone is not proof of gradient explosion; removing clipping is not proof it cannot
occur. No mandatory norm threshold of one is used.

## Four-way mask joint optimization

`--joint-steps N` enables the existing per-Gaussian categorical-logit table;
attention weights do not replace its importance decisions. An optional
`--existence-prior prior.npy` initializes it in **original PLY row order**.

During joint training:

1. Independently sample at least two hard four-way layouts, including q0.
2. Actually encode/decode those layouts and render only retained primitives.
3. Backpropagate conditional task + reconstruction regularization into codec weights.
4. Update categorical logits with task-cost REINFORCE and an independent-sample
   leave-one-out baseline. Task cost includes configured render/projection terms.
5. Add the analytic gradient of expected payload cost over all source rows.

No straight-through placeholder XYZ is assigned to omitted points. The score
estimator's expected gradient is checked against exact enumeration in a small
categorical example, not merely checked for being nonzero. Auxiliary parameter
loss is a codec regularizer, deliberately excluded from the mask reward, so
discarding an inaccurate row does not directly remove its auxiliary cost from
that reward. The logged total includes codec auxiliary cost; `policy_surrogate`
is a separate estimator, not a reconstruction metric.

Important limitations: scene-level score-function gradients can have high
variance, and two samples roughly double full-scene work. This implementation
does not claim their convergence on million-Gaussian scenes is established.
`beta` is a Lagrangian **expected payload** penalty, not a hard packet budget.
Actual deployment uses argmax and may have a different mean rate. Metadata cost
is reported by transmission tools but is not optimized by this rate penalty.

## Training on the server

Old reference_v6 / geometry_first weights cannot be loaded by this mainline.
Use the corresponding historical Git revision to evaluate those experiments;
the pre-cleanup implementation snapshot is `a545aa5`. Existing learned-v4
weights and packet hashes remain supported without preserving old networks.
Start fresh; reusing the old point-cloud PLY is correct.

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
OUT="$PWD/output/truck_learned_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/train_codec_learned.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

Choose an actually available GPU. The script honors `CUDA_VISIBLE_DEVICES`,
`PLY`, `SCENE`, `PYTHON_BIN`, `STEPS`, `RENDER_STEPS`, `JOINT_STEPS`, `LR`,
`BLOCKS_PER_BATCH`, `CLIP_MODE` and `CLIP_NORM`.

Defaults: fixed AWGN 10 dB, at most 3000 codec-pretraining + 1000 render steps,
32 blocks/batch, 256 Gaussians/block, attention windows of 32. These limits are
not claimed necessary training amounts. Every 100 steps the same noise seeds,
layouts and validation blocks/views are evaluated. Patience can end a phase early.
Validation blocks are excluded from local pretraining where possible; validation
views are excluded from render-training camera selection. This is still one-scene
training, not cross-scene generalization. No variable-SNR training is enabled in
this command.

The main script leaves `JOINT_STEPS=0` so codec quality can be assessed without
mask allocation hiding reconstruction errors. To train the complete stack in
the same run, set `JOINT_STEPS=1000` before launch. To start joint optimization
from a learned checkpoint, invoke `train-learned --init ... --steps 0
--render-steps 0 --joint-steps ...` with the normal PLY/source/out/device options.
`--allocation-init route2.pt` can restore its matching mask. `--init` retains the
checkpoint's model/loss configuration but starts fresh optimizers; it is not exact
training resume. Constructor flags apply only to fresh models.

Artifacts:

- `codec.pt`: final weights. `codec_best_attribute.pt`, `codec_best_render.pt`:
  best fixed-validation checkpoints for the corresponding phases. They need not
  equal the final checkpoint. Intermediate `codec_STEP.pt` are also saved.
- `loss.jsonl`: objectives, component contributions, gradients, updates, timings.
- `validation.jsonl`: fixed-condition per-tier/mixed layout XYZ RMSE and, when
  rendering is enabled, complete-scene render distortion and measured payload rate.
- `validation_images/STEP/*.png`: held-out camera, left-to-right photograph /
  input-PLY rendering / decoded rendering. These appear in render/joint phases,
  not during CPU attribute-only training.
- `charts/`: phase-separated objectives, components and gradient/update charts.
- Joint runs additionally write matching `route2.pt`, `route2_best_joint.pt`
  and `codec_best_joint.pt`. Never mix masks and codecs from different saves.

## Evaluate the codec without pruning

```bash
python benchmark_codec.py \
  --ply "$PWD/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply" \
  --checkpoint "$OUT/codec_best_render.pt" \
  --source "$PWD/data/tandt_db/tandt/truck" \
  --out "${OUT}_evaluation" --device cuda \
  --channels none awgn --snrs 10 --tiers 1 2 3 \
  --resolution 2 --save-images --hybrid-ablation
```

Use a fresh evaluation directory. `none` isolates learned bottleneck error;
AWGN adds channel noise. The existing benchmark's receiver-only packet path,
PNG panels, parameter metrics and rendering metrics work with checkpoint v4.
Read image legends: original-photo PSNR and codec-vs-input-PLY PSNR are different.
For CPU parameter checks, omit source/image/ablation flags and use `--device cpu`.

## Local verification and its limits

The test suite covers packed/batched equivalence, exact symbol counts, q0 holes,
empty windows, independent receivers, source-input leakage, learned XYZ gradients,
unclipped coordinate gradients, replay/checkpoint equivalence, exact categorical
gradient expectation, and complete three-stage orchestration with a **mock**
renderer. Old architecture-specific tests were removed with their implementations;
generic transport, replay, rendering, statistics and allocation tests were adapted
to the maintained codec. Unsupported old checkpoints are explicitly rejected.
After cleanup, 45 tests were collected: 42 passed and three CUDA-dependent tests
were skipped. Current learned-v4 checkpoint/packet identity preservation is tested.
The launch script passed `bash -n`; Python compilation and Git whitespace checks passed.

`scripts/test_learned_codec_local.py` performs real truck-PLY CPU training from
scratch on 32 spatial blocks and evaluates eight held-out blocks at fixed AWGN10.
The 400-step run is recorded in `output/learned_joint_final_cpu_20260917/results.json`.
Held-out XYZ RMSE decreased from approximately 53.00 / 34.64 / 42.92 to
20.60 / 20.79 / 20.79 scene units for q1/q2/q3. **Those final errors remain large**;
this verifies optimization and packet closure, not acceptable reconstruction.
It also does not establish a useful rate-distortion ordering yet. These values
must not be compared directly with the old experiment's loss values or different
evaluation subsets. No local CUDA renderer or full-scene render-quality validation
was available.
Across those 400 unclipped steps, total gradient norms ranged from 3.648 to
75.509; actual Adam update L2 norms ranged from .01289 to .05716. No nonfinite
gradient was observed. This does not establish stability of later CUDA render
or million-primitive categorical-mask optimization.
