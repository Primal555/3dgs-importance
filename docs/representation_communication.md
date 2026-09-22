# Representation-first Gaussian codec experiment

This is an opt-in four-module architecture, not a renamed noiseless JSCC run.
The prior Transformer research baseline remains runnable and its checkpoint
identity is unchanged. Select the new architecture with `train-representation`.

The separate [teacher-axis position experiment](teacher_axis_experiment.md)
replaces only the position objective when explicitly requested. The default
objective and original launcher remain available as the control.

## Architecture and objectives

`G -> representation_encoder -> y -> representation_decoder -> G_clean`

`G -> representation_encoder -> y -> channel_encoder -> power/channel -> channel_decoder -> y_hat -> representation_decoder -> G_comm`

- One token per Gaussian. Default latent dimension: 64 real values. This is NOT
  a transmitted rate. Actual communication uses q3, 32 complex symbols per point.
- The representation encoder retains pointwise paths and geometric multiscale
  Context. No SNR or tier embedding enters either representation module.
- The representation decoder is the plain block Transformer with multi-depth
  XYZ readout and logcov/opacity/DC/SH heads. No localization token or progressive
  geometry refinement is added in this experiment.
- Each communication adapter has separate parameters and conditional Transformer
  blocks. The existing per-point soft power constraint is retained to avoid
  changing both factorization and the channel model in the same comparison.
- Only sender-side Context sees source XYZ. Receiver uses received symbols,
  known tier/SNR and shared model/global normalization metadata. No source feature,
  neighbor graph, center, radius, or per-point coordinate side stream is provided.
- By default the spatial-response/logcov objective is unchanged in all phases. It remains
  an engineering surrogate, not image MSE. Render quality is always measured
  separately. No full-scene rendering backward or replay is involved here.

## Training phases

1. `representation`: train E_repr and D_repr from random weights. Both channel
   modules are frozen and absent from the training forward graph.
2. `adapter`: freeze E_repr/D_repr, train E_channel/D_channel through the frozen
   decoder. Frozen weights do NOT mean `no_grad` on the decoder input path.
3. `joint`: unfreeze all four. Objective is `L_comm + clean_weight * L_clean`.
   Both terms use the SAME reconstruction definition and sampled view directions.
   Latent MSE is diagnostic only. Per-objective, per-module gradient norms are
   profiled, along with actual parameter updates.

There are two conceptual stages: clean representation, then adapter warmup +
joint communication learning. These are not parameter-pretraining + render tuning.
LR is fixed at 2e-4 by default; each phase starts a fresh Adam. Clipping is disabled
unless explicitly requested. Training budgets are upper experiment lengths, not
proof of convergence. No automatic passage to communication occurs without a
declared quality threshold.

## Start with representation only (recommended first server run)

In the active maskgs environment, from the repository root:

```bash
OUT="$PWD/output/truck_representation_$(date +%Y%m%d_%H%M%S)"
mkdir "$OUT"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_representation_codec.sh "$OUT" \
  > "$OUT/console.log" 2>&1 &
echo $! > "$OUT/run.pid"
tail -f "$OUT/console.log"
```

GPU 2 is an example, NOT a claim it is currently free. Select an available GPU.
The trainer accepts only an empty new directory or one containing launcher
`console.log`/`run.pid`. It never overwrites an existing run accidentally.

Default: 5000 representation steps, 0 adapter steps, 0 joint steps. This is an
observation run, not an implicit pass criterion. Fixed test-split cameras are used
for validation every 500 steps (also at initialization and phase boundaries).
They are no longer a pristine final test set once used to make design choices.

## Enable all phases with explicit gates

```bash
MIN_CLEAN_PSNR=25 MAX_ADAPTER_PSNR_DROP=1 \
REPRESENTATION_STEPS=5000 ADAPTER_STEPS=1000 JOINT_STEPS=2000 \
CUDA_VISIBLE_DEVICES=2 bash scripts/test_representation_codec.sh "$OUT"
```

25 dB is an EXAMPLE source-render fidelity requirement, and 1 dB an EXAMPLE
allowed communication drop. Neither is a theory-derived optimum. Choose them
based on the clean scene results. PSNR here is vs the original PLY rendering,
NOT vs photographs. If a gate fails, `transitions.jsonl` and `summary.json` say
why, and the next phase does not run. Checkpoints from the completed phase remain.
One must improve/extend the representation model, not silently weaken the gate.

Continue from a successful new representation checkpoint using `INIT=.../codec.pt`
and `REPRESENTATION_STEPS=0`. This deliberately starts fresh optimizers; it is not
exact resume. Do not use checkpoints from the old end-to-end architecture.

For an exact continuation after interruption:

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m gaussian_jscc train-representation \
  --resume "$OUT/training_state.pt"
```

This restores weights, Adam, phase counters, best scores, random states and stored
arguments in the SAME directory, from the last saved step. Other CLI overrides
are ignored explicitly. Keep the same data, software and visible-device count.
Uncheckpointed trailing logs/images may already exist after an interruption;
resume recovery handles those records before continuing.

## Outputs and interpretation

- `training.json`: architecture, exact objective, gates, source fingerprint,
  disjoint fitted/held-out Morton blocks and validation cameras.
- `loss.jsonl`: objectives, gradients before clipping, clip factor, module updates,
  per-objective gradient contributions and timing/peak allocated GPU memory.
- `validation.jsonl`: clean + communication held-out reconstruction/XYZ metrics,
  latent error. During phase 1 the communication adapters are untrained; poor
  communication curves at this point are expected, NOT failed representation.
- `render_validation.jsonl`: source- and photo-referenced MSE/PSNR/SSIM/L1 for both
  paths, source PLY baseline, and communication PSNR drop. Full-scene render
  validation includes fitted points; this is not unseen-scene evidence.
- `images/000500/view_00/{photo,source,clean,communication,comparison}.png`:
  exact fixed-camera comparisons, every 500 steps by default.
- `codec_<step>.pt`, `codec_<phase>.pt`, `codec_best_<phase>.pt`, `codec.pt`:
  weights for existing transport/evaluation APIs; best is selected by held-out
  spatial-response loss, NOT claimed best PSNR. `training_state.pt` is resume state.
- `charts/`: phase-separated objectives, module gradients, clean vs communication
  reconstruction and render quality. Everything stays within one run directory.

Global feature statistics and bbox use the supplied scene as shared model metadata;
the held-out blocks are not involved in weight updates, but this is not a claim
of fully unseen-distribution evaluation. Fixed q3/noiseless is the first task;
AWGN at fixed SNR is supported subsequently. Dynamic rates/masks are not trained
or claimed robust by this experiment.

CPU checks omit `--source`, use small blocks/hidden dimensions and explicit
`--max-clean-loss` / `--max-adapter-loss-ratio` gates if communication is requested.
These gates are solely structural diagnostics and cannot certify rendering quality.

## Local verification (2026-09-22)

`scripts/check_representation_local.py` samples real, spaced Morton regions from
the truck PLY. A 32-region / 8192-point CPU run used hidden=48, latent=64, 256-point
blocks, fixed q3/no noise, LR=2e-4 and 300+100+100 steps. Four blocks (1024 points)
were held out from weight updates. Its deliberately permissive diagnostic gates
exercise all phases; they do NOT assert readiness for deployment.

| Boundary | Clean held-out loss | Communication held-out loss | Clean XYZ RMSE (world units) |
|---|---:|---:|---:|
| Initial | 17.438 | 18.492 | 18.531 |
| Representation 300 | 11.420 | 16.276 | 18.213 |
| Adapter 100 | 11.420 | 13.058 | 18.213 |
| Joint 100 | 10.841 | 12.925 | 17.504 |

The frozen representation path stayed unchanged during adapter training, and
all three phases had finite gradients. Location improvement remains limited;
this does not establish success in rendering or superiority over the previous
codec. There was no CUDA rasterizer on this machine. Image-file/metric integration
is unit-tested with a mocked renderer; real PSNR/SSIM and GPU memory must be
verified on the server. Detailed local artifacts are under the ignored
`output/representation_cpu_check_20260922/` directory, not committed.
