# Mild gradient accumulation, without reducing optimizer updates

Start a fresh random run using `scripts/test_center_absolute_accum.sh`.
Everything from the previous soft-update experiment is unchanged except
`--center-accumulation-steps 2`. There is no block-centroid structure and no
additional objective. Decoder self-only scope and affine XYZ readout remain.

| Setting | Previous soft run | New accumulation run |
| --- | --- | --- |
| Optimizer updates | 5000 | **5000**, not 2500 |
| Microbatches per update | 1 | 2 |
| Blocks per microbatch | 32 | 32 |
| Sampled blocks per update | 32 | 64 |
| Base LR | 1e-4 | 1e-4, no linear scaling |
| LR decay | second half to 2e-5 | identical, by optimizer update |
| Validation/images/checkpoint interval | 500 updates | 500 updates |

Two separately sampled microbatches contribute to the same gradients. Compute
the original loss mean over **all valid points across both batches**, accounting
for partial blocks; do not sum two mean gradients at full weight. Backpropagate
each microbatch immediately, without retaining both computation graphs. Clip
or softly limit updates only after accumulation, and step Adam once. Its
moments and counters advance once per outer iteration, never per microbatch.

Default memory for activations stays near one microbatch; compute for center
forward/backward is roughly doubled, not free. There are 10000 microbatches,
320000 sampled blocks, at most 81,920,000 point observations for 256-point
blocks (partial blocks reduce this). Sampling is with replacement; these are
NOT unique Gaussians. Compare both equal update budgets and processed-point /
elapsed-time budgets: a gain cannot automatically be credited to averaging
rather than the extra data/computation. No claim that two is universally optimal.

`CENTER_ACCUMULATION_STEPS=1` gives the existing soft-update baseline. Defaults
never ramp to 4/8/16, reduce the final update count, or multiply LR. B/C phases
are unchanged. Accumulation is incompatible with the legacy loss-rejection
guard, so no step is approved on just the final microbatch by mistake.

## Observability and recovery

- Console reports `accum` and valid `points/update`.
- `loss.jsonl`: weighted loss, per-microbatch point counts/losses, sampled block
  IDs (with probes enabled), cumulative actual optimizer calls and point work.
- `summary.json` / training checkpoints: `center_work` totals and counting start.
- `charts/center_training_effort.png`: loss and center-only PSNR by processed
  points, complementing the existing curves by optimizer step.
- Drift diagnostics remain before/after one **complete** accumulated update.
- Checkpoints are taken only at optimizer boundaries. Interrupted microbatches
  are replayed from the last saved weights/Adam/RNG; never resume half a sum.
- Exact resume restores the saved accumulation setting and schedule. Historical
  resumes without this option use one microbatch; work counts begin at the first
  newly logged update and disclose that step, not invented historical counts.

## Server launch

Choose an available GPU (2 below is only an example):

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_center_absolute_accum2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_center_absolute_accum.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

Judge XYZ using `center_only`; attributes remain untrained in `full`. Outputs
and logs stay in this run directory. Old experiments are not overwritten.
