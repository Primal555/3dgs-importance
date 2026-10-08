# Full 0/4/12/24 experiment

This is a new random-initialized run, not a reinterpretation of an 8/16/32
checkpoint. The rate sweep motivates these candidate lengths; it does not
prove they are optimal, or that all important points can dispense with 32.
The old eight-prefix sweep used 4-symbol normalization layers, so its uniform
quality numbers cannot be promised for this new 4+8+12 codeword.

## Pipeline and defaults

| Stage | Updates | What changes | Objective |
|---|---:|---|---|
| Attribute bootstrap | 5000 | Codec | Historical isolated local-response RGB MSE |
| Scene rendering | 5000 | Codec | Two-view original-PLY render MSE |
| Allocation only | 2000 | Keep and conditional-tier probability tables | Render MSE + rate penalty |
| Joint fine-tuning | 1000 | Codec and allocation | Same render MSE + rate penalty |

The last two stages are one `--joint-steps 3000` schedule with
`--mask-only-steps 2000`, preserving allocation Adam moments at unfreezing.
The codec follows the existing phase transition policy: bootstrap -> render
clears codec Adam moments; render -> allocation/joint preserves them. Phases
carry the last executed weights forward, not a silently restored best model.
Stage lengths are engineering starting points, not convergence guarantees.

Fixed conditions: AWGN 10 dB, reliable 16-bit XYZ with delta/zlib level 6,
block size 256, 64 blocks/batch, replay backward, no clipping. Codec LR is
fixed at 1e-4 in all stages. Keep LR 0.01; conditional-tier LR 0.001;
allocator Adam epsilon 1e-15. Mixed codec layouts independently drop 5% of
points; uniform codec layouts keep all points. q0 learning retains all
sampled q0 counterfactual feedback and independent keep/tier hard-Gumbel
branches. No quotas, q0 cap, or new rejection rules are introduced.

Only retained points send coordinates. Deployment still uses ten categorical
draws: drop only if all ten are q0; otherwise choose the modal positive tier.
When a PLY lacks `masks_0/masks_1`, initialization is keep=0.99 and equal
conditional positive tiers, not an inherited MaskGaussian importance prior.

## Rate penalty uses the actual 24-symbol maximum

The denominator is the measured full-retention total cost at the actual
highest tier, 24 symbols. There is no separate 32-symbol reference or
cross-version penalty alignment:

```
--beta 0.01
D = 24 + measured_full_retention_XYZ_and_tier_map_uses_per_point
L_rate = beta * expected_payload_and_side_proxy / D
```

Costs here are per original Gaussian. Thus an all-q3 scene has normalized
cost 1. With beta unchanged, the penalty per communication use is somewhat
stronger than under the former full-32 denominator. This is intentional,
not a strict matched-penalty comparison. Actual accounting charges the new
prefix lengths and measured XYZ/tier-map cost. It is not a hard budget or
a differentiable zlib model.

`training.json` records the actual full-tier payload, denominator,
and beta/denominator. Allocation loss rows also record the denominator and
`allocation_stage` (`allocation_only` or `joint_finetune`).

## Server launch

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git pull --ff-only --no-recurse-submodules origin main
GPU=2  # select a currently available GPU; this is not an availability check
OUT="$PWD/output/truck_rates41224_full_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/train_progressive16_rates41224_full.sh \
  "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

`Ctrl+C` exits tail only. SSH disconnection does not stop the nohup job.
The launcher ignores inherited INIT/ALLOCATION_INIT and never reuses old
weights. Set PLY/SCENE for another scene; this remains per-scene training,
not multi-scene shared-codec training. Optional stage overrides are
BOOTSTRAP_STEPS, RENDER_STEPS, ALLOCATION_STEPS and JOINT_FINETUNE_STEPS;
all four must be positive. Final test is enabled by default; FINAL_EVAL=0
explicitly skips it. TEST_TRIALS defaults to 2.

## Results and interpretation

All outputs are subdirectories of OUT. Attribute/render validation defaults
to every 500 updates; allocation/joint validation defaults to every 100.
The frozen-allocation boundary always validates and saves a matched pair,
even if its step is not a regular checkpoint/validation interval.

- `loss.jsonl`: phase, allocation substage, codec_updated, image/rate terms,
  sampled counts, timings and gradient diagnostics.
- `allocation_history.jsonl`: actual ten-draw q0..q3 counts/shares, changed
  points, expected probabilities and payload; not argmax-only deployment.
- `validation.jsonl`, `validation_images/<step>/`, `charts/`: fixed validation
  views, PSNR/SSIM against original PLY and photos, uniform/mixed/learned maps,
  measured XYZ and tier-map proxy costs during allocation/joint.
- `codec_end_allocation.pt` + `route2_end_allocation.pt`: pre-finetune pair;
  exported to `deployment_end_allocation/`.
- `codec_best_joint.pt` + `route2_best_joint.pt`: matched pair selected by
  actual deployment MSE + rate penalty across BOTH allocation and joint
  stages. Best may precede joint fine-tuning; never assume fine-tuning helps.
  Exported to `deployment_best_joint/`.
- `test_best_uniform/`, `test_best_mask/`: final held-out test-camera results
  with the same selected codec. Tests charge packet metadata and XYZ, save
  images, and report PSNR/SSIM against source PLY and photos. They still assume
  reliable digital side delivery at 2 net bit/use, without simulated FEC or
  packet errors. Shared codec weights are excluded. Disk float32 I/Q array
  bytes are NOT over-the-air JSCC bit costs.

Compare total channel uses versus quality, not total loss against an old
normalization. Quantify any texture/SSIM losses as well as average PSNR.
Changed budgets require retraining; these results are not a strict paired
allocation-only comparison with the old codec. Local CPU integration tests
use a synthetic renderer and do not establish real-scene quality or GPU memory
requirements.
