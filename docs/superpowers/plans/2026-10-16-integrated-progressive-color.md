# Integrated Progressive Color Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an isolated, runnable one-pair Gaussian JSCC experiment with 5000 attribute updates and 1000 render updates.

**Architecture:** Reuse the existing learned_joint progressive residual codec and packet format. Add independent training, exact checkpoint continuation, scene validation, evaluation and plotting modules; do not change the modular experiment or old training defaults.

**Tech Stack:** Python, PyTorch, existing CUDA rasterizer on the server, unittest, Bash.

**Spec:** `docs/superpowers/specs/2026-10-15-integrated-progressive-color-design.md`

## Global Constraints

- Truck only; random initialization; reliable 16bit XYZ; cumulative complex budgets 0/12/24/48.
- Fixed 10 dB AWGN; LR=1e-4; no clipping; 5000 bootstrap then 1000 render updates.
- Every fourth bootstrap update uses unclipped direction RGB; objective-local tier counters avoid schedule aliasing.
- Keep all existing architectures, checkpoints and scripts unchanged.
- Only initial/final images; numeric validation and checkpoints every 500; replay rendering.

## Review Focus

- Momentum must not move non-color heads in ANY residual stage on a color update: Task 1 tests.
- A checkpoint at the last bootstrap update must transition once, not replay or skip render updates: Task 2 tests.
- Changed dataset/split/settings on resume must fail rather than silently alter the experiment: Task 2 tests.
- Unsent enhancement slots and q0 cannot leak into earlier predictions; receiver must not access source: Task 1/3 tests.
- Validation must not consume training RNG or compare tiers using shifted noise slots: Task 3 tests.

## Task 1: Scheduling and noisy color/attribute updates

Files: create `gaussian_jscc/integrated_training.py`, `tests/test_integrated_progressive.py`.
Interfaces: `update_spec(phase, phase_step, color_every=4) -> (objective, layout)`;
`bootstrap_step(model, features, q, geometry, centers, *, color=False, color_directions=8)`;
`make_optimizer(model, lr)`; `save_state(...)` / `restore_state(...)`.

- [x] Write failing tests for the exact 5000-step objective/tier count table, noisy end-to-end color gradients, all-stage non-color Adam protection, and full-prefix/padding boundaries.
- [x] Run tests and verify missing-feature failures.
- [x] Implement independent objective counters, reuse local_response/directional_color objectives, and clear non-color output-head gradients after color backward.
- [x] Verify finite gradients, fixed XYZ/statistics, replay equivalence and checkpoint optimizer/RNG roundtrip.

## Task 2: Isolated two-stage executable

Files: create `gaussian_jscc/integrated_scene.py`, `gaussian_jscc/integrated_experiment.py`; expand tests.
Interfaces: scene metadata loader accepts old and modular source directories; executable accepts `--run`, `--out`, `--resume`, phase budgets, manifest/device/LR/interval overrides.

- [x] Write failing CPU pipeline tests using real model/channel/Adam, replacing only CUDA rendering and camera I/O.
- [x] Implement source identity/split verification, random initialization, two-stage loop and fresh Adam at transition, exact same-directory resume and settings validation.
- [x] Verify interrupted bootstrap, exact transition and interrupted render match uninterrupted weights, optimizer state, layout and noise streams.
- [x] Verify source checkpoint is never read as initialization; changed objective/budget/scene is rejected and incomplete runs lack complete.json.

## Task 3: Observability and deployment handoff

Files: create `gaussian_jscc/integrated_evaluation.py`, `gaussian_jscc/integrated_plots.py`, `scripts/test_integrated_progressive_color_truck.sh`, `docs/integrated_progressive_color_truck.md`; expand tests.
Interfaces: `evaluate(model, scene, *, out, step, phase, trials, seed, save_images)` returns per-path/per-tier rows; `plot_run(out)` reads saved logs only.

- [x] Write failing tests for RNG-preserving paired evaluation, measured XYZ/metadata costs, finite-noiseless labels, endpoint-only images and receiver-only packet roundtrip.
- [x] Implement full-scene q1/q2/q3/mixed evaluation, attribute/color metrics, prefix gains, final residual correction diagnostics and plots.
- [x] Add Bash launch/resume guards and documented nohup commands; smoke-test actual script parsing.
- [x] Run the complete unittest suite, request fresh read-only review, fix blocking findings with regression tests, commit only task files and push main.

## Execution record

The user requested complete implementation and push after reviewing the design. Execute in this session; preserve unrelated worktree changes. CPU checks are implementation evidence only, not real CUDA quality claims.

- Base commit: `2093618`. No existing modular or one-pair implementation files were changed.
- Initial RED: five tests could not import the missing independent modules; then scheduling/noisy-color/replay checks passed before the executable was added.
- Pipeline GREEN: interrupted bootstrap, phase boundary and interrupted render reproduced the uninterrupted result; source dummy checkpoint remained unread and unchanged.
- Fresh read-only review found three resume/runtime gaps. Regressions reproduced two assertion failures and one missing-prior exception before fixes.
- Fixed camera/image/accounting identity verification, unused-prior reloading in final diagnostics, and completed/older-checkpoint rewinds. Nine focused tests passed after fixes; final full-suite validation is recorded with the delivered commit.
- Final verification: `python -m unittest discover -s tests` — 336 tests, 4 skipped, zero failures (358.953 seconds); Bash syntax and actual launch-script `--help` smoke checks passed.
- Review limitations: actual CUDA rasterizer, GPU memory/runtime and Truck quality require server execution; no CPU convergence claim is made.
