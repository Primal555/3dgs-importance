# Modular multi-scene random training implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Train the approved four-module SSIM backbone from random weights on three pruned scenes.
**Architecture:** Reuse existing modular training/evaluation/checkpoint mechanics; add independent scene orchestration and compact per-scene/macro reporting. No existing Truck behavior changes.
**Tech Stack:** Python, PyTorch, existing CUDA rasterizer, unittest, Bash.
**Spec:** `docs/modular_multiscene.md`.

## Global constraints

- Only selected train-role scenes fit statistics or gradients; no historical weights or allocation tables.
- Every scene switch binds its own geometry without rebuilding shared Adam.
- Stage budgets are PER SCENE; semantic decoder LR is 1e-4 early and 1e-5 only in joint.
- Preserve unrelated user changes and existing entry points; targeted CPU tests, no local full-scene training.

## Review focus

1. Mid-round and phase-boundary resume preserve scene cursor, optimizer and RNG.
2. Heldout/missing scenes cannot silently enter statistics or fall back to raw PLYs.
3. Camera partitions and scene identity must match on resume.
4. Macro metrics give scenes equal mass and do not merge incompatible paths/layouts.
5. Failed launches do not overwrite an output or silently reuse Truck weights.

### Task 1: Scene/config preparation and balanced schedule

**Files:** new `gaussian_jscc/modular_multiscene.py`, `configs/modular_multiscene_ssim.json`, `tests/test_modular_multiscene.py`.
**Interfaces:** reuse Scene/manifest/shared_statistics; expose parser(), prepare_inputs(args), phase_lrs(args,phase), main(args,max_updates=None).

- [x] Write tests for selected scene roles, missing paths, random initialization, shared statistics, per-scene schedule and phase-specific LRs.
- [x] Run `python -m unittest discover -s tests -p test_modular_multiscene.py`; observed missing module failures.
- [x] Implement fresh model construction and per-scene round robin with one shared Adam per phase.
- [x] Run the same tests; passed.

### Task 2: Resume, evaluation and output

**Files:** above trainer/tests; new `gaussian_jscc/modular_multiscene_report.py`, `gaussian_jscc/modular_multiscene_state.py`.
**Interfaces:** existing modular checkpoints carry nested scene records; existing evaluate_paths writes per-scene validation/test rows; report reads these and root losses.

- [x] Add tests for exact mid-round and boundary continuation, objective/identity mismatch, per-scene/macro metrics and initial/final images only.
- [x] Observe failing tests before implementing these behaviors.
- [x] Implement atomic progress/checkpoints, log repair, scene-tagged evaluation and separate/macro curves.
- [x] Run targeted tests plus existing modular training/checkpoint/evaluation tests; 29 passed including launchers.

### Task 3: Launch and handoff

**Files:** new `scripts/train_modular_multiscene_ssim.sh`; expand spec with launch/resume instructions.

- [x] Add executable launcher tests using a recording Python command (GPU requirement, positional arguments, resume forwarding).
- [x] Observe failures; implement script and document server launch commands.
- [x] Verify Python CLI help, Bash syntax, targeted tests and scoped git diff.
- [x] Independent final review of new files; two important findings reproduced and fixed with regression tests (camera content identity; interrupted evaluation recovery).
- [x] Commit scoped changes and push to origin main as user requested; implementation pushed as `954fb06` and `8e6b01c`.

## Verification record

31 targeted tests passed after the review fixes (39.801s): new multi-scene/launcher tests plus existing modular training, checkpoint, evaluation and SSIM launcher tests.
Architecture config matched the actual Truck SSIM metadata with no differences.
Bash syntax and Python CLI help passed. No full local scene training or repeated whole-project regression suite was run; synthetic CPU render results do not establish CUDA quality or convergence.
