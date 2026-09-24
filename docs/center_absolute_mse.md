# World-coordinate MSE ablation

`scripts/test_center_absolute_mse.sh` changes only the center training loss
relative to the two-microbatch soft-update experiment. Old entry points keep
their distance objective for reproducibility. No new model layers, targets,
geometry weights or position side channels.

Old: `mean(sqrt(sum(world_error**2) + tau**2) - tau)` over points.

New: `sum(world_error**2) / (3*N)` (mean over valid points **and XYZ axes**).

`world_error = (predicted_unit_xyz - source_unit_xyz) * geometry.span`.
This restores physical world-coordinate units; it does NOT divide by scene
extent. There is no square root, smoothing, shape scaling, clipping or new
coefficient in MSE. Targets are detached. Reductions compute in float64 and
return the model dtype, as in the existing distance objective.

MSE increases the influence of large errors and its gradient tends to zero
near a correct prediction. It may help outliers or fine convergence, but may
also let rare distant points dominate training; no PSNR gain is guaranteed.
Large initial MSE/gradients alone do not establish numerical explosion. New
loss values have squared-world units and cannot be compared numerically with
the old distance curve. Validation logs include world MSE, RMSE, mean distance,
and labelled selected loss. Images and center-only PSNR remain comparable.

## Unchanged controls

- Random initialization, seed 42, per-point absolute XYZ Transformer/self-only
  decoder with affine readout; 32 center latent dimensions.
- 5000 optimizer updates, two sequential microbatches of 32 blocks each.
- Gradients averaged over all valid points, including partial blocks.
- Continuous Adam, no loss-rejection guard or automatic momentum restart.
- LR 1e-4 for the first half, cosine decay to 2e-5 in the second half.
- Same historical update-size protection; no new gradient clipping.
- No attribute/joint stages. Images, PSNR/SSIM and checkpoints every 500 updates.

Both training accumulation and validation use the selected loss. It is recorded
in training.json/checkpoint arguments and loss.jsonl (`world_center_mse`), and
the plot is labelled accordingly. Exact resume retains the stored loss; old
checkpoints without this setting retain `distance`. Use a fresh output directory
for this ablation, not `--resume` on the old experiment.

## Launch

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_center_absolute_mse_accum2_$(date +%Y%m%d_%H%M%S)"
# GPU 2 is an example; first check availability.
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_center_absolute_mse.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

Expected startup: `Center loss: mse`, `2 microbatches/update`, and
`Joint directional step guard: False`. Assess `center_only`, not `full` with
untrained attributes. Results remain together in the selected run directory.
