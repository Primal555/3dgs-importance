# Evidence allocation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a frozen-codec allocation experiment with useful two-target evaluations and training-effective finite-render corrections.

**Architecture:** Optional evidence scheduler and paired render task integrate into the existing kernel/lifecycle; old entry remains compatible. New entry owns the independent switches and engineering defaults.

**Tech Stack:** Python, PyTorch, unittest/pytest, Bash launcher.

**Spec:** docs/superpowers/specs/2026-10-26-evidence-allocation-design.md

## Global Constraints

Frozen codec; source MSE/fixed beta; 0/8/24/48 at 10dB; all q0 gate feedback; single train/ten deploy; no quotas/joint phase; preserve unrelated worktree files.

## Review Focus

- All-closed and partial blocks must retain q0 feedback without positive candidates.
- Counterfactuals change one point and preserve noise; keep changes refresh context.
- Finite feedback replaces an action cost without corrupting base view losses.
- Resume exactly restores evidence, selection coverage and RNG, rejecting changed settings.
- Non-finite values, low image changes, empty intervention sets and frozen model identity.

### Task 1: Evidence state and precise finite MSE

**Files:** new `gaussian_jscc/allocation_evidence.py`, `gaussian_jscc/allocation_finite.py`, `tests/test_allocation_evidence.py`.

**Interfaces:** EvidenceScheduler supplies independent column rounds, centered cost accumulation and complete state_dict; PairedMaskedRenderTask supplies cached delta_mse(scene).

- [x] Write tests for coverage/demand uniqueness, aging/context invalidation/resume, analytic rate preservation and high-precision finite differences; run RED.
- [x] Implement state and paired render task; run GREEN.

### Task 2: Kernel, entry and server handoff

**Files:** modify budgeted kernel/entry/state only behind the independent mode; new evidence entry, launcher, docs and integration tests.

**Interfaces:** optional evidence scheduler in kernel; distinct checkpoint family and parser defaults, exact lifecycle reuse.

- [x] Write tests for two independent rounds, finite costs entering updates, all-q0 behavior, RNG/model preservation, lifecycle exact resume and switches; run RED.
- [x] Implement integration, diagnostic records, launcher and documentation; run targeted tests GREEN.
- [x] Run relevant compatibility tests, attempt project suite, shell syntax and dry-run against available source; report limits.
- [x] Independent whole-change review; fix findings and verify; commit and push scoped files.

## Execution record

- User approved implementation of the preceding design discussion and previously selected in-place work. No unrelated files changed/staged.
- Pre-flight interfaces: scheduler provides rounds/state/memory to kernel; paired task provides finite differences to corrections; lifecycle persists scheduler state. No conflicting contracts found.
- RED: five initial unit tests failed on missing evidence modules; two kernel tests failed on one-round count/missing finite fields; entry tests failed on absent independent entry. All passed after implementation.
- Final directed/compatibility command: `PYTHONPATH=tests python -X utf8 -m unittest test_allocation_evidence test_evidence_allocation_kernel test_evidence_allocation_entry test_fixed_beta_allocation test_budgeted_allocation test_budgeted_allocation_kernel test_allocation_mirror test_fixed_beta_allocation_report test_budgeted_allocation_report -q` — **46 tests passed**.
- Independent reviewer: no Critical/Important findings. Two minor findings were fixed with reproducing RED tests: lazily aged evidence diagnostics and demand ranking using just-invalidated variance. No deferred findings.
- Bash syntax and actual launcher `--help` execution passed; all changed Python modules compile. Real completed Mip360 endpoint validation passed: 11250 updates, nine trained scenes, 0/8/24/48, successive channel, reliable16bit.
- Full `unittest discover -s tests -q` was attempted, then this task's own process was stopped while running historical experiments; no full-suite pass claimed. Actual-scene CUDA speed/quality and end-to-end local dry-run requiring server-only pruned PLY paths remain server verification.
