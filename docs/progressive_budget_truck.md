# Increased symbol-budget experiment

Run `scripts/test_progressive_budget_truck.sh CLEAN_COLOR_RUN NEW_OUT` with an
available `CUDA_VISIBLE_DEVICES` and the `maskgs` environment activated.
Use `truck_clean_color_gpu1_20261014_103802` as the semantic teacher, not the
previous communication run. The script copies only the completed semantic
encoder/decoder and attribute statistics; channel weights and Adam start fresh.

## Controlled settings

| Setting | Existing experiment | Increased-budget experiment |
|---|---|---|
| Cumulative complex symbols q1/q2/q3 | 4 / 12 / 24 | 8 / 24 / 48 |
| New symbols per layer | 4 / 8 / 12 | 8 / 16 / 24 |
| Semantic latent y | 48 real values | unchanged |
| Channel architecture | 3 residual MLP blocks, width 192 | unchanged depth/width |
| Semantic modules | frozen | frozen |
| Training | 4000 latent-MSE steps | 4000 latent-MSE steps |
| SNR / LR | 10 dB / 1e-4 | unchanged |
| Position delivery | reliable per-axis 16bit, delta-zlib | unchanged |

The shared channel encoder produces one maximum-length signal, and each tier
receives its prefix. The decoder recovers y1, then adds corrections y2 and y3.
The budget changes only the channel encoder output width and decoder segment
input widths, not y or the semantic network. Consequently this is a bandwidth
experiment, **not a parameter-count-matched architecture comparison**.
Per-complex-symbol power is unchanged, so longer prefixes also use more total
energy at the same per-symbol SNR. No monotonic quality guarantee is imposed.

## Outputs and interpretation

One output root contains the suite plan, `progressive.log`, comparison plots,
and the `progressive/` training run: checkpoints, loss/validation/test JSONL,
initial/final render images, and completion metadata. Intermediate validations
save numbers, not images. Render evaluation remains enabled, but does not update
weights; no render-training phase or additional comparison arm runs.

Compare to the previous 4/12/24 **4000-step** run with the same teacher, separately
for q1/q2/q3: clean PSNR, finite-noiseless PSNR/NMSE, and 10 dB PSNR/NMSE. Clean
must remain identical. If finite-noiseless recovery improves, limited channel
representation capacity is implicated; if only noisy recovery improves, extra
symbols primarily help robustness. These are diagnostic indications, not proof
that budget was the sole bottleneck. Do not interpret saved float32 signal-file
size as over-the-air cost; report actual complex-symbol counts plus the unchanged
reliable XYZ side-stream accounting.

The `--rates Q1 Q2 Q3` option is recorded in plans and checkpoints and is also
supported by the read-only suffix-shuffle probe. Defaults remain 4/12/24.
Exact resume requires the same rates and original output directory; use
`--resume-suite` on the same script for an interrupted run. Never resume an old
4/12/24 communication checkpoint with the new budgets.
