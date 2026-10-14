# Frozen-semantic channel capacity control

Run baseline two-layer pointwise channel MLPs versus opt-in residual channel
MLPs on the **same completed clean-color semantic checkpoint**. Never copy
its untrained channel weights. The semantic encoder, semantic decoder and
attribute mean/std remain bit-identical; gradients still traverse the frozen
semantic decoder into the channel decoder during rendering.

Residual variant: hidden 192; three pointwise residual blocks at EACH end;
each block uses hidden-to-2hidden-to-hidden GELU MLP and bounded affine
SNR conditioning (decoder also includes the existing tier condition).
Final residual weights initialize at std 0.001. No LayerNorm, neighbor mixing,
new symbols, position prediction or new side-information. Baseline remains
hidden 96 and unchanged state-dict/config syntax. This is a joint width/depth/
conditioning upgrade, not an isolated depth ablation or a parameter-matched test.

Both arms: reliable 16bit XYZ, 4/12/24 complex symbols, same per-point
per-segment power normalization and floor, same q1/q2/q3/mixed cycle,
10 dB AWGN, no pruning/allocator. 2000 real updates latent MSE followed by
2000 real updates original PLY render RGB MSE + 1 * latent MSE; lr 1e-4.
Adam resets between these two phases as before. Both semantic modules stay
frozen throughout (unlike ordinary modular communication render training).
Seed is reset after model/scene construction so corresponding arms use the
same minibatches, training cameras, tier actions and noise draws. Suite resume
retains optimizer, progress and RNG and skips completed arms.
The initializer retains the teacher's batch grouping settings; do not override
`--blocks-per-batch` to a different value in this control.

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/truck_clean_color_gpu1_20261014_103802"
OUT="$PWD/output/truck_channel_capacity_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_channel_capacity_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log" "$OUT/baseline.log"
```

The suite runs baseline then residual on one GPU. To follow the second arm:
`tail -n 40 -F "$OUT/residual.log"`. Select an actually available GPU.
To resume: repeat identical launch options/output with `--resume-suite`.

Outputs stay in ONE root, with baseline/residual subdirectories. Each includes
loss/validation/test JSONL, exact checkpoints and initial/final representative
images. Final numeric tests cover all heldout test cameras at 0/5/10/15/20 dB
with two matched trials plus finite-noiseless and clean paths. Comparison
contains metrics CSV (including colors), quality/rate and enhancement plots,
per-tier latent/color/PSNR curves and tier-separated latent training curves.
Summary audits semantic state against the teacher's SHA256; altered semantic
weights/statistics abort comparison. clean must remain stable here: no semantic
adaptation is permitted. Do not judge gains only by q3 or enhancement gaps.

This test cannot prove improved noise tolerance locally: local integration tests
use real PyTorch channel paths and a synthetic differentiable renderer; actual
Truck CUDA rendering and performance must be checked on the server.
