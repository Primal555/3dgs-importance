# Progressive 16-bit XYZ codec: restore inexpensive attribute initialization

Entry point: `scripts/train_progressive16_two_stage.sh`.

## What is restored, and what stays unchanged

Stage A restores `gaussian_jscc/local_response.py` from commit `d74bf75` (first
introduced at `37e4d35`), not the later learned-center/logcov variants. Predicted
and source Gaussians share a center. Their isolated orthographic RGB responses
are compared over four random directions, source/prediction footprint probes,
and black/white backgrounds. The source and adaptive probes are detached.
The loss is the average black/white response MSE, not a manually weighted sum
of rotation, scale, opacity and SH losses. The historical footprint floor and
probe sampling are still engineering choices; this is not a full renderer.

Stage A samples 64 local blocks per update with ordinary autograd. It trains
the shared encoder, symbol projection, decoder and attribute heads through
the noisy channel. It does not do a full-scene render at every update. It does
not supervise XYZ, simulate scene occlusion or perspective, or guarantee unique
recovery of every original attribute. Both fixed-block local validation and
held-out full-scene render validation are recorded at validation intervals.

Stage B carries the learned weights forward, clears Adam moments, sets the
render LR, and optimizes **only** full-scene multiview RGB MSE using replay.
It does not retain a weighted local auxiliary. The source PLY renders remain
the reference; photographs are a separate evaluation reference. XYZ remains
fixed in both phases. `joint_steps=0` refers to the unused mask-optimization
phase, not a missing end-to-end update: encoder and decoder train together.

Both phases retain progressive 8+8+16 symbol layers, SNR conditioning, and
per-Gaussian mixed tiers. The reliable compressed 16-bit XYZ path is unchanged.
The new launcher's default LR remains the current **1e-4 in both phases**,
not the historical run's 2e-4. Default budgets are 5000+5000, with no automatic
LR decay or early stop. These defaults leave room to observe the new trajectory;
the old run's plateau location is not imposed as a stop rule.

## Logs and checkpoints

Everything is written into one experiment directory:

- `loss.jsonl`: phase name, global `step`, local `phase_step`, phase-specific
  objective label, gradient/update measurements, timing and LR.
- `bootstrap_validation.jsonl`: fixed held-out blocks' local-response loss.
- `validation.jsonl` and `validation_images/`: full-scene quality in BOTH
  phases, every 500 updates by default, with paired symbol noise and prefix gains.
- `codec_best_bootstrap.pt`: selected by full-scene validation MSE during
  Stage A, not by its surrogate loss. Recorded for inspection; the transition
  uses the last Stage A weights (no silent best-checkpoint restoration).
- `codec_end_bootstrap.pt`, `codec_best_render.pt`, `codec_end_render.pt`,
  regular numbered checkpoints, `codec.pt` (last weights).
- `charts/`: separate phase objective axes, local validation curve, scene
  PSNR/SSIM and prefix-gain plots. Local-response MSE is not labeled SmoothL1.

Adam state is reset on the phase transition, as in the historical procedure.
Checkpoints still save codec weights, not exact-resume optimizer/RNG state.
Local validation is a surrogate diagnostic; it is not a promise of scene fidelity.

## Background server launch

Select a genuinely free GPU; GPU 2 below is an example, not a current availability claim.

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c http.version=HTTP/1.1 -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main || exit 1
OUT="$PWD/output/truck_progressive16_two_stage_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 INITIALIZATION=random BOOTSTRAP_STEPS=5000 RENDER_STEPS=5000 \
  nohup bash scripts/train_progressive16_two_stage.sh "$OUT" > "${OUT}.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "${OUT}.pid"
printf 'PID: %s\nOutput: %s\n' "$TRAIN_PID" "$OUT"
tail -f "${OUT}.log"
```

`Ctrl+C` exits `tail`, not the background training. To explore a cheaper render
budget use `RENDER_STEPS=3000`. `LR` and `RENDER_LR` explicitly override each
phase's fixed rate. To reproduce the historical learning rates (but not its old
coordinate precision/encoding), specify `LR=0.0002 RENDER_LR=0.0002`.
Inherited `INIT` is ignored in random mode. Explicit checkpoint initialization
requires matching progressive/16-bit/delta_zlib metadata and is not exact resume.

## Phase progress analysis

Run after training, using a new output folder:

```bash
python scripts/analyze_training_phases.py --training "$OUT" --out "$OUT/phase_analysis" --window 500
```

This leaves original logs and weights untouched. It writes phase-local PSNR and
window-gain plots, exact checkpoint/window CSVs, and timing/best-checkpoint JSON.
It describes diminishing gains, not an automatic convergence detector.
See [September 19 measured results](two_stage_history_20260919.md).
