# Evidence-driven allocation test

Independent entry: `gaussian_jscc.evidence_allocation`; launcher: `scripts/test_evidence_allocation.sh`. Existing fixed-beta and budget entries keep their defaults/checkpoint identities.

## What changes

- Two distinct targets per block. First: least evaluated, then oldest; second: image-gradient influence × conditional-tier uncertainty × past variation × age. These scores schedule computation only. Invisible points keep the basic coverage path; visibility in one window never defines global importance.
- Each candidate forward changes **one** target in an independent block. The two other positive actions are decoded with the same noise as the actual scene. No simultaneous multi-target attribution within a block.
- Centered normalized image-action costs are averaged over fresh observations, with 1000-step half-life. Current analytic rate differences are added after averaging. A block's old evidence is invalidated when mean total-variation drift of its probability distribution exceeds 0.1. Stochastic masks themselves do not invalidate it. Evidence is not replayed as independent measurements.
- Eight supplemental finite interventions per update by default: up to six positive actions and two keep flips, when available. High influence, ambiguous RD sign and rotating coverage share the positive checks. Actual finite costs replace selected training action costs. Keep changes refresh semantic context; retained neighbors are recomputed. All q0 gates still receive baseline feedback every update.
- Baseline images are cached during the normal backward. Finite MSE change uses `mean(2*(base-reference)*delta + delta**2)` with float64 accumulation, unchanged pixels cancel before averaging. Base loss/view records and training RNG remain untouched.

The half-life, distribution drift threshold, candidate/correction counts and independent LR defaults (keep 0.1 / tier 0.4) are exposed engineering parameters, not literature-optimal constants. No per-point loss magnitude normalization, forced tier proportions, hard communication budget or joint finetuning.

## Server start (Bash)

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948"
OUT="$PWD/output/truck_evidence_beta0p1_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_evidence_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

The launcher accepts the Mip360 pipeline root and resolves its `training/` endpoint. It also accepts an explicit completed training directory. Use an available GPU; this example does not detect ownership or reserve GPU2. All source camera splits/statistics are checked. Heldout scenes cannot be fitted. Paths can be relocated with the existing `--manifest` option, preserving PLY identity.

Exact resume: use the **same** `RUN`, `OUT` and all arguments with `RESUME=1`. Old fixed-beta checkpoints are not evidence initializers. The codec source remains fixed; the new probability table starts fresh.

## Independent controls

- `--candidate-rounds 1 --candidate-policy round_robin`: one coverage-only target.
- `--no-evidence-memory`: use just fresh centered action costs.
- `--no-finite-feedback`: no supplemental decode/render corrections; full q0 gates remain.
- `--tier-lr 0.1`: explicitly match the older positive-tier step size.
- `--finite-samples 4`: reduce finite-render workload.
- `--no-uniform-controls`: skip final fixed q2/q3 test renders if already measured with this exact codec/protocol.

Candidate batching amortizes the baseline but adds decode work. Finite corrections add full-scene renders; there is no guaranteed wall-clock speedup. Compare candidates/second and downstream quality at matched measured communication cost, not global step counts alone.

## Output

- `metrics/loss.jsonl`: candidate throughput, coverage, effective evidence, q0 feedback count, finite corrections/sign disagreements, cost proxies and per-component timing.
- `metrics/audit.jsonl`: supplemental **training-applied** finite feedback. Zero predictions are separate; they do not count as informative sign agreement.
- `comparison/evidence.json`: selected-point confidence and ten-draw seed sensitivity. Seed comparisons are numerical tier comparisons, not additional rendered quality claims.
- `comparison/uniform_controls.csv`: learned initial/selected/latest and same-codec fixed q2/q3, including measured total complex uses, source/photo PSNR, SSIM, LPIPS and RD scores.
- Existing charts, selected tier/probability arrays, initial/final panels and tier maps remain in the same output folder. Calibration renders save metrics only. An identical latest map aliases selected images.
- Exact checkpoints contain the full evidence history/context, finite-check coverage, scheduler and RNG; incompatible settings are rejected on resume.

CPU directed tests use a real small codec and a differentiable toy rasterizer. Server CUDA rendering speed, quality gains and memory peaks must be measured on the actual scene.
