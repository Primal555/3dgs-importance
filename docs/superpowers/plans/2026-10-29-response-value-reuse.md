# Response-value candidate reuse

## Spec

Speed up the existing response-value allocator using current-state and semantic
encoding reuse. Preserve all four complete-block counterfactual outputs, paired
noise slots, labels, feature definitions, coverage and training budgets. Existing
pretraining checkpoints must resume allocation with unchanged family/settings.
The frozen successive codec is the fast-path target; other codecs retain the
reference path. No changes to codec architecture or serialized schemas.

## Global Constraints

Work in the current checkout as requested. Preserve unrelated changes. Execute
inline with TDD, one independent final review and targeted verification. Do not
run the historical whole-repository suite. Keep channel forwards at their
original full batch shape to preserve paired AWGN/fading and block context.

### Task 1: Implement exact candidate and semantic reuse

Interfaces: `decode_candidates` gains optional ephemeral semantic cache and
counter arguments. `ResponseValueWindow` caches actual semantic output inside
one window. Four absolute tier outputs remain the public generator contract.

1. Add parity tests against the old four-forward implementation, including
   PointNet context, q0, mixed tiers, inactive rows, padding, noise and caller RNG.
   Require three candidate forwards and no semantic refresh for unchanged masks.
   Run targeted tests. Expected: new speed assertions fail before implementation.
2. Use three relative tier shifts per row at full batch shape and reassemble
   four absolute-tier outputs. Reuse present/absent semantic masks, refreshing
   the mask whenever the selected point changes presence. Account actual calls.
3. Run tests again. Expected: exact outputs and original RNG, reduced counts.

### Task 2: Verify checkpoint compatibility and document handoff

Interfaces: Task 1's ephemeral cache must not enter plans/checkpoints. Existing
reference pretraining can resume optimized allocation through the same launcher.

1. Add old-reference pretrain to optimized allocation resume test; compare final
   learner, table, scheduler and deployment with reference continuation.
2. Run response tests and relevant allocation compatibility tests, syntax checks
   and whitespace checks. Expected: all targeted checks pass.
3. Document scope and counts, measured small-block runtime, and unchanged server
   resume command. Obtain independent review, address important findings with
   failing tests, then commit and push scoped files.

## Review Focus

Paired noise slots for mixed current tiers; full neighbor changes and block
scale frames; q0-dependent semantic context; inactive/padded rows; stale cache
scope; numerical parity with PointNet and neighbor recovery; no checkpoint or
plan identity changes; truthful counters and runtime claims.
