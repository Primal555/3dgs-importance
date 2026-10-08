# Fixed-histogram allocation control

This is a one-shot diagnostic, not another training pipeline. It loads the
matching `codec_best_joint.pt` and `route2_best_joint.pt`, validates the source
PLY identity, reconstructs the learned ten-draw deployment map, and shuffles
that map across all original rows (including q0).

Both arms retain exactly the same q0/q1/q2/q3 counts and attribute payload.
Compressed coordinate and tier-map bytes can differ, so total costs are measured
separately with the same cost meter used during allocation validation. These
costs exclude complete packet framing, FEC/retransmissions and shared weights.

The comparison preserves Morton blocks, training codec batch size, held-out
validation cameras, checkpoint compression settings and operating SNR/channel.
Full point/symbol-slot noise is paired across arms using the original validation
seed schedule. No optimizer is created and model identity is checked afterwards.

Server command, with an available GPU:

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/test_allocation_shuffle.sh \
  output/truck_hierarchical_mask_gpu2_20261008_110441
```

Defaults are one shuffled map (`SHUFFLE_SEED=2026`) and two channel trials
(`TRIALS=2`). The existing output directory is never overwritten. An optional
second positional argument specifies a different output directory.

Outputs in `TRAINING_DIR/allocation_shuffle/`:

- `summary.json`: both quality/cost summaries and paired learned-minus-shuffled
  deltas. Positive PSNR/SSIM favors learned; negative MSE favors learned.
- `metrics.csv`: individual source-PLY/photo metrics by view and channel trial.
- `images/`: learned and shuffled render panels for each view, first trial only.
- `manifest.json`: codec identity, original counts, views, seeds and assumptions.
- `learned_tiers.npy`, `shuffled_tiers.npy`: maps in original input PLY row order.

One random permutation is only a quick descriptive control, not statistical
significance or generalization evidence. If learned quality is higher, compare
actual total costs too before calling it an equal-rate advantage. This control
jointly tests deletion and positive-tier assignment; it does not isolate their
individual contributions. `--ply`/`--source` overrides are available via
`python -m gaussian_jscc.allocation_shuffle` for relocated datasets.
