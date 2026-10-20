# Power normalization follow-up implementation plan

> **For agentic workers:** execute inline in the current repository, as authorized by the user's approval of all three experiments.

**Goal:** diagnose remaining attribute damage, continue matched point/block channel recovery by 4000 updates, then jointly finetune each for 1000 updates.

**Architecture:** a new suite consumes the two completed `*_awgn10` children of the power-controls run. It uses the existing checkpoint format, cached frozen teacher, replay render training and evaluation infrastructure. Shared evaluation/report helpers provide independent source-run protection, attribute ablations, paired 32-test-view endpoints, 4-view intermediate validation, diagnostics and exact restart.

**Tech Stack:** PyTorch, existing CUDA rasterizer on server, synthetic differentiable rasterizer for CPU checks, matplotlib, Bash.

**Spec:** approved in-chat proposal immediately before the user's implementation request.

## Global constraints

- Groups: point/block only; same original teacher, source scene, split and 8/24/48 symbols; reliable 16bit XYZ.
- Diagnostic replacements: normal / clean DC+SH / clean scales+rotation / clean opacity. Only initial-model diagnostic; never train using substituted attributes.
- Channel continuation: +4000 actual updates at 1e-4, retain weights/Adam; original-y MSE; both semantic modules frozen.
- Joint: +1000 actual updates; channel LRs1e-4, semantic decoder LR1e-5; semantic encoder remains frozen. Preserve channel Adam moments, new decoder state starts fresh.
- Joint objective: original PLY render MSE + (1-SSIM)/3 + latent MSE; replay; q1/q2/q3/mixed independent schedule. Test images never enter training.
- Baseline, end-channel, final: the SAME up-to-32 actual test cameras and four paired AWGN trials, no training-camera fallback; intermediate validation uses original heldout views with a distinct scope.
- No change to existing experiments or statistical normalization; no source writes. All outputs under one parent, images only initial/final plus q3 diagnostic panels.
- Checkpoints persist before expensive evaluation; exact resume retains cursor/Adam/RNG, replaces incomplete endpoint metrics, skips completed ablation.
- Standing user instruction: commit and push after focused verification, no local full-scene training.

## Review focus

1. Correct raw attribute slices; XYZ and non-replaced attributes never change; replacements use clean teacher, not source PLY.
2. No optimizer reset or semantic target drift across continuation, and channel moments survive joint transition.
3. Test set disjoint from training; endpoint/validation means are not mixed into one unlabelled curve.
4. Resume at initial evaluation, channel/joint transition and within joint matches uninterrupted weights and moments.
5. Point/block sources must match by semantics/config/scene; output must not overwrite or contain a source run.

### Task 1: evaluation and ablation

- [x] Failing tests for replacement slices, source immutability, original block decoding and real test-set selection.
- [x] Implement evaluation helper using existing decode/render/metric paths; bounded image output and atomic results.

### Task 2: matched staged continuation

- [x] Failing test for inherited Adam, newly trainable decoder, frozen encoder and boundary/interruption resume.
- [x] Implement read-only source plan, worker, checkpointed evaluation cursors, deterministic sampling, cached channel updates and replay joint updates.

### Task 3: launch, comparison and handoff

- [x] Implement CSV/plots separating test endpoints from validation; source vs photo metrics and attribute replacement gains.
- [x] Add launcher/docs, run focused CPU tests and real-source dry-run, independent review.
- [ ] Commit/push scoped changes under the user's standing instruction.

## Execution record

- Existing unrelated edits remain excluded. Source checkpoints are version4 and contain channel Adam state; source phase cursor4000.
- Test exposure is evaluation-only; repeated endpoint comparisons are for this ablation study, not final model selection evidence on an untouched test set.
- Focused verification: 3 follow-up integration/unit tests, 5 power-normalization tests and 2 checkpoint tests passed. CPU integration uses a synthetic differentiable rasterizer, not evidence of real-scene quality.
- Real source `truck_power_controls_20261020_184354` passed read-only checkpoint/config/teacher pairing dry-run; no PLY loading or output creation. Bash syntax and Python compilation passed.
- Independent review identified evaluation-before-checkpoint risk. Failure injection reproduced stale step6 instead of7; persist-before-evaluation cursor fixed it, and interrupted evaluation resumed to the same final weights. Channel/joint boundary and within-joint resume also preserve Adam state.
- Single-arm summaries now leave pairwise power matching unassessed instead of incorrectly reporting a match.
- No full-repository regression or local full-scene training, consistent with the user's preferred bounded verification.
