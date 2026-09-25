# Fixed-XYZ color/shape/opacity swap diagnosis

This is evaluation only, using the finished checkpoint, not a new training run.
Source attributes are oracle interventions, NOT a deployable decoding path.
No model state, training code, learning rate, loss or old output is changed.

The default checks q3, the original four held-out cameras and two noise draws.
For each tier/trial it decodes once, then swaps raw attributes in row-aligned
Morton order. All eight variants preserve exactly the same delivered XYZ:

| Variant | Attributes replaced with source values |
| --- | --- |
| received | None (baseline) |
| source_color | DC and all higher-order SH |
| source_dc | DC only |
| source_sh | Higher-order SH only |
| source_shape | Scale and rotation |
| source_opacity | Opacity |
| source_shape_opacity | Scale, rotation and opacity |
| source_attributes | All non-XYZ attributes (quantized-XYZ control) |

The source reference image uses original XYZ. The last control measures how
close the image can get by replacing all attributes while retaining delivered
XYZ; it is not expected to have exactly zero error against the original PLY.

If replacing color repairs the hue, color prediction is implicated. If replacing
shape/opacity repairs it, compositing is implicated. Both may contribute or even
compensate for one another: these effects are NOT additive causal percentages.
Do not use PSNR alone to judge hue. Inspect the matched images and RGB error
metrics. Signed per-channel biases can cancel across regions; R-G/G-B error RMSE
is also reported to separate chromatic errors from equal-RGB brightness shifts.
It is not perceptual Delta-E. No white balance, rescaling or image alignment is
applied. Metrics use unclipped renderer RGB, except SSIM (display-clipped).

## Server command

Choose a genuinely available GPU; 2 is only an example.

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c http.version=HTTP/1.1 -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
TRAIN_DIR="$PWD/output/truck_quantized16_zlib_gpu2_20260924_224932"
DIAG_OUT="$TRAIN_DIR/color_ablation_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/render_color_ablation.sh "$TRAIN_DIR" "$DIAG_OUT" > "${DIAG_OUT}.log" 2>&1 &
DIAG_PID=$!
printf '%s\n' "$DIAG_PID" > "${DIAG_OUT}.pid"
tail -f "${DIAG_OUT}.log"
```

`Ctrl+C` exits tail only. All new results are inside the original run directory.
Existing directories are never overwritten. To diagnose other tiers use:

```bash
CUDA_VISIBLE_DEVICES=2 python -m gaussian_jscc.color_ablation --training "$TRAIN_DIR" --out "$TRAIN_DIR/color_all_tiers" --tiers 1 2 3 --trials 2
```

Results: `comparisons/q3_view00.png` ... `q3_view03.png` are 3x3 sheets;
`panels/q3/` contains full-resolution four-column panels (photo, source, swapped
image, absolute error x4). Images are saved for trial 0, metrics for both trials.
`summary.csv/json` aggregate the per-view/trial `metrics.jsonl`; `manifest.json`
records input paths, model hash, controls and interpretation boundaries.

Batch grouping, resolution, held-out view names, channel, SNR and seed schedule
come from training.json, reproducing the old validation settings. The default
checkpoint is codec.pt (last), not silently the best. RGB outputs may differ
slightly across library/device versions. PLY count/degree and camera names are
checked; the user must supply the actual original PLY, not another same-size file.
CLI `--ply` / `--source` or launcher `PLY` / `SCENE` overrides support relocation.

Real images require CUDA plus the project's rasterizer. CPU tests verify swaps,
metrics and workflow with a synthetic renderer; they are NOT scene-render evidence.
