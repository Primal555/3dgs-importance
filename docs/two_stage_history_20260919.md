# September 19 two-stage run: measured diminishing returns

Source: `output/truck_two_stage_random_replay_b64_20260919_163725/`.
Read `training.json`, `loss.jsonl`, `bootstrap_validation.jsonl`, and
`validation.jsonl`; no retraining or original-file edits. Analysis script:
`scripts/analyze_training_phases.py`, window 500. Local output:
`phase_analysis_20260924/{summary.json,checkpoints.csv,window_gains.csv,phase_progress.png}`.

The run used 883438 Gaussians, reliable 12-bit XYZ, the adaptive (not progressive)
shared codec, fixed 10 dB AWGN, 64 blocks, LR 2e-4 in BOTH phases, no clipping,
and two training views per render update. Validation: four fixed camera views,
two noise trials, and four layouts (q1/q2/q3/mixed). Values below are mean
source-render PSNR over those observations/layouts, not PSNR from pooled MSE
and not photo PSNR. Global 7000 = render-local 2000.

## Stage A: local-response attribute initialization

| Updates within stage | Source PSNR (dB) | Held-out local-response MSE |
| ---: | ---: | ---: |
| 0 | 9.952 | 0.076695 |
| 500 | 16.813 | 0.016853 |
| 1000 | 17.563 | 0.014869 |
| 1500 | 18.494 | 0.013987 |
| 2000 | 19.709 | 0.013048 |
| 2500 | 20.578 | 0.010877 |
| 3000 | 20.729 | 0.009704 |
| 3500 | 20.751 | 0.009394 |
| 4000 | 20.775 | 0.009060 |
| 4500 | 20.407 | 0.008991 |
| 5000 | 21.202 | 0.008859 |

Interpretation: an evident slowdown around **2500-3000**, not complete
convergence. From 3000 to 5000 the endpoint improves ~0.473 dB and local MSE
falls another ~8.7%. The best observed Stage A source-MSE checkpoint is 5000.
The temporary fall at 4500 is not evidence that further training is useless.
Adjacent 500-step window mean PSNR gains after 3000 are +0.090, +0.162,
-0.042, +0.183 dB: diminishing but still nonzero/noisy improvement.

Recorded training updates average 0.03970 s; 5000 sum to 198.52 s. The last
2000 updates cost roughly 80 s at that average. This excludes full-scene
validation, image/checkpoint I/O, initialization and other wall-clock overhead.
Thus retaining 5000 inexpensive updates is reasonable despite the slowdown.

## Stage B: full-scene render optimization

| Updates within stage | Global step | Source PSNR (dB) |
| ---: | ---: | ---: |
| 0 | 5000 | 21.202 |
| 500 | 5500 | 22.514 |
| 1000 | 6000 | 22.854 |
| 1500 | 6500 | 22.942 |
| 2000 | 7000 | 23.389 |
| 2500 | 7500 | 23.379 |
| 3000 | 8000 | 23.418 |
| 3500 | 8500 | 23.456 |
| 4000 | 9000 | 23.352 |
| 4400 (best source MSE) | 9400 | 23.671 |
| 4500 | 9500 | 23.551 |
| 5000 | 10000 | 23.508 |

The first 500-1000 steps yield the largest gains; subsequent improvement is
slower. A clearer plateau-like regime appears at **3000-3500**. To avoid
over-interpreting single checkpoints, mean PSNR over the last 500-update bins:

| Window | Mean source PSNR | Gain vs previous 500-update window |
| --- | ---: | ---: |
| 2501-3000 | 23.2954 | +0.1413 |
| 3001-3500 | 23.3896 | +0.0942 |
| 3501-4000 | 23.4445 | +0.0548 |
| 4001-4500 | 23.4765 | +0.0320 |
| 4501-5000 | 23.4983 | +0.0219 |

These are descriptive gains from a single run, not independent significance
tests. Best source MSE is 0.00441737 at render step 4400; final MSE is
0.00459990. Best checkpoint is preferable to blindly taking the last weights.
Final photo PSNR averaged across layouts is 21.031 dB, not 23.508 dB.

Mean render-update time is 2.35827 s, median 2.33857 s. 5000 updates sum to
11791.36 s (~3.28 hours), excluding validation/I/O. At this rate the final 2000
updates cost ~79 minutes for modest refinement. Stage A does not make each
full-scene render update faster; it supplies a cheap initialization.

## Application to the new run

Practical historical trade-off: **5000 attribute + 3000 render** for quicker
iteration; **5000 + 5000** to observe late refinement and select the best model.
The new launcher keeps 5000+5000 by default, with explicit step overrides.
It now uses 16-bit coordinates, progressive prefixes and LR 1e-4, so neither
the old timing nor these plateau steps are guaranteed to transfer unchanged.
No automatic early stop or LR reduction is introduced based on this one run.
