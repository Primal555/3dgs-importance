# Deployment-aligned joint training and prefix preservation

User selected direct implementation of both improvements in the current workspace.
Preserve unrelated edits. No separate A/B runs or extra design approval.

1. RED/GREEN: exact ten-draw deployment marginals, replayable hierarchical sampler,
   nonzero existence surrogate gradients (including several retained draws).
2. Integrate chunked deployment-cost gradients and compact replay seeds into the
   existing per-point masked rendering feedback; preserve all q0 counterfactuals.
3. RED/GREEN: same-view cyclic uniform-prefix anchors every four joint updates;
   average codec image gradients, leave mask image/rate gradients unchanged.
4. Load paired end-allocation codec/tables for joint-only initialization. Keep
   fresh Adam semantics explicit and preserve the existing rate normalizer.
5. RED/GREEN: initial/final-only panels, including random mixed versus learned
   deployment; full intermediate metrics, PNG charts, compact review archive.
6. CPU synthetic integration and complete regression suite; review; commit only
   scoped changes and push main. CUDA quality remains a server experiment.

Tests run with the existing offline uv dependency cache (PyTorch 2.14.1 CPU);
the recorded old Anaconda interpreter paths no longer exist. The current
workspace is explicitly requested; no new worktree or destructive rollback.

## Verification and review

- RED/GREEN verified missing deployment sampler/marginals, prefix helper,
  endpoint panel controls and joint-only initialization.
- Exact marginals checked against enumeration and 60,000 deployment samples;
  replay/checkpoint gradients agree with AWGN, including codec gradients.
- Reviewer caught repeated full-table index gradients in the rate chunks.
  Local score leaves plus slice accumulation fixed it; chunk-size and SNR-slope
  equivalence tests pass, with only one full-table image backward hook.
- Compact and full panel policies preserve identical validation metrics;
  synthetic initialization/final evaluation and review.zip generation pass.
- Regression suite: 152 tests, 3 platform-dependent skips. No real CUDA quality
  claim; GPU runtime/memory, rasterizer fidelity and actual scene improvements
  remain server experiments. Shell syntax and git whitespace checks passed.
- Fresh read-only reviewer: no unresolved Critical or Important findings.
  CUDA-only behaviors explicitly declined, not silently treated as verified.
- Current checkout/main and standing user push instruction govern integration;
  unrelated preexisting edits and downloaded external projects are excluded.
