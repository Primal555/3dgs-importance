# Fixed Beta Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed-beta rate-distortion allocation mode with no communication budget or adaptive price.

**Architecture:** Reuse the frozen-codec mixed-neighborhood kernel and current experiment lifecycle. Introduce a validated fixed-RD objective, mode-specific selection and checkpoint identity, then extend reporting and an explicit-beta launcher. Keep budget-mode CLI/checkpoints compatible.

**Tech Stack:** Python, PyTorch, unittest, Bash, Matplotlib.

**Spec:** `docs/superpowers/specs/2026-10-24-fixed-beta-allocation-design.md`

## Global Constraints

- 2000 allocation updates; 4 views/window; frozen codec/statistics; no joint training.
- Rates 0/8/24/48; SNR10; reliable16bit XYZ and binary16 scales unchanged.
- Mandatory explicit finite beta >= 0; mutually exclusive with budget arguments.
- Fixed D_ref from initial q3 calibration and C_ref from actual full-q3 delivery; rank by calibrated actual-deployment RD.
- Preserve all-q0 feedback, candidate rotation, same-update reuse, ten-draw deployment and source/test boundaries.
- Use current repository per user's declared preference; preserve unrelated dirty files. No full historical training regression per approved scope; run directed and compatibility tests.

## Review Focus

1. beta=0 must not fall into budget mode through a truthiness check.
2. A cheaper but worse image can win RD; smallest MSE alone must not determine selection.
3. Expected/sample proxy improvements must not determine deployed best_rd.
4. Resuming with changed beta, mode or references must fail without rewriting output.
5. Pure-budget legacy checkpoints and reports must retain their existing behavior and fields.

### Task 1: Fixed objective and experiment lifecycle

**Files:** create `gaussian_jscc/allocation_objective.py`, `tests/test_fixed_beta_allocation.py`; modify `gaussian_jscc/budgeted_allocation.py`, `gaussian_jscc/budgeted_allocation_kernel.py`, `gaussian_jscc/budgeted_allocation_state.py`.

**Interfaces:** `FixedRateDistortion(beta,d_ref,c_ref).score(distortion,cost)` returns normalized RD; `candidate_key(row)` ranks measured candidates by `(score,source_mse,total_channel_uses)`. Kernel accepts `rate_normalizer` while retaining legacy `budget` keyword. State identity follows the mode-specific plan version.

- [x] Write objective/CLI/end-to-end tests: hand-computed RD values, invalid beta, beta0 without budget, frozen weights, best_rd, exact resume, reference preservation and changed-beta rejection.
- [x] Run `python -m unittest discover -s tests -p test_fixed_beta_allocation.py -v`; expect failures for absent fixed-beta support before implementation.
- [x] Implement fixed objective, explicit mode parsing, no adaptive controller in RD, actual-cost calibration selection, saved best_rd/reference/beta and resume validation. Keep budget branch intact.
- [x] Run new tests plus `test_budgeted_allocation.py`, `test_budgeted_allocation_kernel.py`, `test_allocation_mirror.py`, `test_allocation_budget.py`; expect all pass.
- [x] Commit Task 1 implementation and tests.

### Task 2: Reports, launcher and usage

**Files:** modify `gaussian_jscc/budgeted_allocation_report.py`, `docs/budgeted_allocation.md`; create `scripts/test_rate_distortion_allocation.sh`, `tests/test_fixed_beta_allocation_report.py`.

**Interfaces:** RD final result/validation rows expose `objective_mode,beta,d_ref,c_ref,rd_score`; measured-cost event records continue in `metrics/control.jsonl` with a cost-refresh event and no adaptive-price update.

- [x] Write report tests with literal measured RD scores, tamper rejection and optional LPIPS missing; real Bash launcher tests for mandatory beta, budget conflict, dry run, resume flag and exit status.
- [x] Run `python -m unittest discover -s tests -p test_fixed_beta_allocation_report.py -v`; expect failures for missing RD report/launcher behavior.
- [x] Implement mode-aware score/cost/quality plots and CSV, report integrity checks, explicit-beta launcher, documented server commands and engineering-vs-paper references.
- [x] Run all fixed-beta and budgeted tests, mirror/budget/reuse compatibility tests, Python compilation, `bash -n` and `git diff --check`; expect pass.
- [ ] Perform independent whole-change review using requesting-code-review, address Important/Critical findings with failing regression tests, then commit and push to origin main under user's standing instruction.

## Execution choice

User explicitly approved implementation of the presented short design and has previously requested current-repository work and directed verification. Execute inline without another design/plan approval pause; final independent review only. Do not run full CUDA training locally.
