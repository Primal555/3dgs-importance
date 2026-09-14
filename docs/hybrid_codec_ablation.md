# Gaussian JSCC hybrid parameter ablation

`benchmark_codec.py --hybrid-ablation` reuses each decoded Gaussian scene to render four
row-aligned parameter combinations:

| Name | XYZ | Non-position attributes | Interpretation |
|---|---|---|---|
| `reference` | source | source | input PLY upper reference |
| `position_error_only` | decoded | source | degradation caused by decoded positions |
| `attribute_error_only` | source | decoded | degradation caused by decoded opacity, scale, rotation and SH |
| `received` | decoded | decoded | complete codec result |

All source rows are Morton-ordered before columns are mixed with decoder rows. This is
essential: rendering is permutation invariant, but a hybrid Gaussian row is not.
The ablation retains every Gaussian and does not invoke the learned four-tier allocator.

## Server command

Use a new output directory. A compact noiseless diagnosis is usually the most useful
starting point because it isolates the codec bottleneck from channel noise:

```bash
PROJECT=/data/home/zhangyueheng/projects/3dgs-importance
PLY="$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply"
SCENE="$PROJECT/data/tandt_db/tandt/truck"
INIT="$PROJECT/output/truck_codec_attr_retrain/codec.pt"
OUT="$PROJECT/output/truck_codec_hybrid_none_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=0 python -u benchmark_codec.py \
  --ply "$PLY" \
  --checkpoint "$INIT" \
  --source "$SCENE" \
  --out "$OUT" \
  --tiers 1 2 3 \
  --snrs 10 \
  --channels none \
  --trials 1 \
  --resolution 2 \
  --device cuda \
  --hybrid-ablation \
  --save-images
```

For a multi-SNR AWGN curve, replace the condition arguments with:

```bash
--snrs 0 5 10 15 20 --channels awgn --trials 1
```

## Outputs

Each condition directory contains:

- `hybrid_ablation.json`: mean and per-view metrics plus exact variant definitions;
- `metrics.json`: the same render metrics consumed by the general benchmark;
- `views/*.png` when `--save-images` is enabled. Panels are ordered as ground truth,
  input PLY, position-error-only, attribute-error-only, and fully decoded;
- `stats.json`: flattened mean metrics together with channel use and parameter errors.

The benchmark root contains `results.json` and automatically generates
`charts/hybrid_ablation_quality_vs_snr.{png,svg}`. The chart has one column for each
error isolation and rows for PSNR and SSIM. `evaluation_chart_data.csv` contains the
same aggregated values for later plotting.

Hybrid mode performs the channel transmission and decoding only once per condition.
It renders four Gaussian variants instead of the usual two, so its rendering portion
takes approximately twice as long. LPIPS remains optional and is enabled by `--lpips`.
