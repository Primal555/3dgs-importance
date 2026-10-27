# Structured Evidence Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline execution of this approved plan.

**Goal:** Implement mixed-distortion continuous-history allocation without changing historical experiments.
**Architecture:** Explicit opt-in hooks in the frozen-codec engine; separate objective, history, correction and entry modules.
**Tech Stack:** Python, PyTorch, pytest, Bash; CPU toy rasterizer verification.
**Spec:** docs/superpowers/specs/2026-10-27-structured-evidence-allocation-design.md

## Global Constraints

User approved direct implementation after review, in the existing repository. Preserve unrelated dirty files. Freeze codecs. No quotas, hard budget, changed deployment rule, or migration of historical families. Run directed affected tests, no full historical regression/full-scene training. One fresh final code review. No remote push without authorization.

## Review Focus

Objective/calibration/selection consistency; stable differences; one observation per action; q0 readiness; full gate feedback; exact resume; legacy compatibility.

### Task 1: Mixed objective and paired rendering

Files: allocation_distortion.py; tests/test_structured_allocation_distortion.py.
- [x] RED: SSIM definition, paired perturbations, calibrated/manual/MSE objectives and component gradients.
- [x] GREEN: fixed calibrated objective and per-camera component render task.
- [x] Verify focused tests; commit task files.

### Task 2: Continuous histories and bounded updates

Files: structured_evidence.py; tests/test_structured_allocation_history.py.
- [x] RED: drift preservation, decay/replay, per-tier keep readiness, exact diagnostics and log-odds bounds.
- [x] GREEN: component histories, refresh selection, risk scheduling, bounded mirror step.
- [x] Verify focused tests; commit task files.

### Task 3: Experiment integration and exact correction

Files: structured_allocation_finite.py, structured_evidence_allocation.py; opt-in budgeted_allocation*.py hooks; tests/test_structured_allocation_integration.py.
- [x] RED: real small codec/toy render, replacement, uncapped q0, consistent selection, exact-resume validation.
- [x] GREEN: wire objective, history, correction and bounded updates into frozen-codec lifecycle.
- [x] Verify new and affected legacy tests; commit task files.

### Task 4: Launcher, reporting and final review

Files: scripts/test_structured_evidence_allocation.sh; docs/structured_evidence_allocation.md; launcher/report tests.
- [x] RED/GREEN: launcher arguments, dry-run, numerical reports.
- [x] Verify CLI, shell syntax, directed affected tests and diff hygiene.
- [x] Fresh final reviewer; fix substantial issues with reproducing tests.
- [x] Commit and report implementation plus local/server verification limits (60 directed tests and 10 subtests passed; no full-scene training or remote push).

## Review resolution

Fresh reviewer identified one important action-coverage parity lock and two minor issues (nonrotating remainders and scene-sized snapshot cloning during restore). Each was reproduced by a failing test, fixed, and accepted by the same reviewer. Coverage now uses branch/action-specific exact counts; remainder groups rotate by step; restore validates metadata without snapshot cloning. No unresolved review findings. Full-scene CUDA speed/memory and quality remain server measurements.
