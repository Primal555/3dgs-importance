# Channel normalization controls implementation plan

> **For agentic workers:** execute inline; the user approved the experiment design and requested implementation in the current repository.

**Goal:** compare point, statistical and original-block power normalization under noiseless and 10 dB channel-only training.

**Architecture:** keep one frozen semantic teacher, recreate identically seeded channel networks and fresh Adam for each arm. Opt-in configuration preserves historical checkpoint identities. A standalone suite owns deterministic sampling, calibration, original-block diagnostics, rendering, exact resume and comparison artifacts.

**Tech Stack:** PyTorch, existing modular codec/checkpoints, matplotlib, Bash launcher.

**Spec:** approved in-chat design immediately preceding the implementation request.

## Global constraints

- Rates 8/24/48, 4000 optimizer updates per arm, LR 1e-4; pure original-y MSE.
- No learned allocator, semantic updates, rendering gradients, or receiver-only access to source energy.
- Uniform positive q1/q2/q3 cycle, independent of noise RNG; original Morton blocks never regrouped.
- Training statistical scales use detached EMA (momentum .01); initial and each validation calibration use the full fixed teacher population, eval freezes buffers. This calibration is an explicit transmitter update, not free sample-dependent receiver metadata.
- AWGN variance per real component is 10^(-10/10)/2, independent of instantaneous point power. Record actual per-segment/prefix power and flag incomparable budgets.
- Initial/final images only; intermediate numerical validation; all outputs under one suite directory, per-arm subdirectories/logs.
- Preserve unrelated worktree changes. User's standing instruction authorizes commit/push after verification; no local full-scale training.

## Review focus

1. Historical point-normalized state/config hashes and forwards must remain unchanged.
2. Padding, empty blocks and regrouping minibatches must not change valid-point block normalization.
3. Statistical eval/calibration must not update weights or consume training RNG; calibration state must survive checkpoint reload.
4. Noisy/noiseless arms must share initialization, teacher, tier/data sequence; only noise/normalizer may differ.
5. Resume must restore Adam, RNG, statistical buffers and cursor without repeating calibration or double-counting updates.

### Task 1: normalization modules and compatibility

- [x] Write failing tests for amplitude preservation, original-block independence, padding, frozen evaluation and reload.
- [x] Add `channel_power_mode` and optional EMA momentum to modular config, omitting default fields from historical serialization; integrate a non-default normalizer in ChannelEncoder.
- [x] Verify focused normalization and existing modular/checkpoint tests.

### Task 2: six-arm suite and measurements

- [x] Write failing CPU integration test for identical initialization/frozen teacher, training, calibrated validation and exact resume.
- [x] Implement suite, original-block diagnostics, full-population power calibration/measurement, comparison CSV/plots, and checkpoints.
- [x] Verify deterministic CPU fixture with synthetic rasterizer; verify real source dry-run.

### Task 3: launch and handoff

- [x] Add fail-fast Bash launcher with per-arm selection, dry-run and resume; document inputs, costs and calibration protocol.
- [x] Run relevant tests, syntax checks and independent review; stage only scoped files for commit/push.

## Execution record

- Initial inspection: historical per-point/per-segment floor is .01; all semantic modules and original context can be reused.
- Ruling: use the full scene's frozen teacher latents for statistical calibration and power accounting, cached on CPU in original blocks. This avoids optimistic subset-only power claims and repeated semantic forward costs; extra host memory is about 4*N*latent_dim bytes plus XYZ/masks.
- Worktree: user previously explicitly chose adding files in the current repository; unrelated edits preserved.
- Verification: `python -m unittest discover -s tests -p 'test_modular*.py' -q`: 71 tests, OK. `python -m unittest discover -s tests -p 'test_channel*.py' -q`: 59 tests, OK. The latter includes normalization and six-arm synthetic integration tests. No full-scene local training or CUDA quality claim.
- Real source preflight: completed 6000-step Truck checkpoint, SHA256 `5bb461633188a0f0f1ea69409670570098e5155e6f35f0d2c7d362165743b161`; six matched jobs, rates8/24/48, no output created by dry-run.
- Bash syntax and missing-GPU fail-fast checks passed. Python compile checks passed.
- Independent final reviewer found P2: interruption before initial evaluation completed left no resumable checkpoint. Added a reproducing failing test, then persisted step0 before camera/caching/evaluation and replaced partial initial metric rows on replay; RED to GREEN verified, including bitwise equal final weights after early interruption.
