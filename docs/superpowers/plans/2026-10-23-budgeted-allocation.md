# Budget-constrained Gaussian Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task in the current session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在明确总通信预算下训练逐点档位概率表，并以真实十次采样部署的成本和校准画质选择结果。

**Architecture:** 新增独立 `gaussian_jscc.budgeted_allocation` 入口；冻结 successive 的全部编解码器与统计量。以实际混合邻域的单点候选、完整 q0 门控反馈和多视角流式梯度估计局部收益，用镜像更新训练表，用真实部署成本驱动非负预算价格。

**Tech Stack:** Python、PyTorch、NumPy、现有 CUDA Gaussian rasterizer；CPU unittest 使用真实小模型和可微合成渲染器。无需新增依赖。

**Spec:** `docs/superpowers/specs/2026-10-23-budgeted-allocation-design.md`（用户已要求按该文档实现）

## Global Constraints

- 独立实验入口，保留 `successive_allocation` 及其 checkpoint、日志格式和训练行为。
- q0/q1/q2/q3 = 0/8/24/48 个累计复符号、可靠 16bit XYZ、实际 binary16 块尺度。
- 第一轮只优化场景专属概率表，冻结全部编解码器权重和属性统计量；不加入联合微调。
- 初版只训练一个显式选择的场景；heldout 场景不能隐式进入分配训练。
- 单次硬采样学习、十次规则部署；训练时不物理删点；所有关闭点均保留反事实门控路径，无每块数量上限。
- `--budget-total-uses B` 与 `--budget-ratio-to-q3 r` 必须且只能选择一个，无默认预算。
- 成本包含 payload、XYZ、完整元数据/档位表和尺度流；保留输入实验明确记录的可靠侧流净 bit/use 假设；共享权重不计。
- 训练目标为 `D/D_ref + lambda*(C_proxy/B - 1)`；固定全 q3 校准 MSE 为 D_ref，实际部署成本用于控制和预算验收。
- 默认 2000 次真实分配更新、四个训练视角/窗口、25 步控制周期、10 步详细计时周期。
- 十次部署 seed 固定且保存；接收端读取发送的离散档位表，不重新采样。
- 只在初始及最终/选中端点保存代表视角图；中间验证仅保存指标。q0 的不可见区域不能标成删点空间分布。
- 定向测试通过后提交推送；不跑全仓历史训练回归，不在本机启动完整场景长训练；保留用户现有未提交文件。

## Review Focus

1. 概率期望满足预算、十次部署却超预算：只能判为不可行，禁止拿代理成本作最终验收（Task 3）。
2. 场景原始 PLY 行序与块/Morton 顺序不同：概率、候选游标、成本与最终渲染必须使用正确映射（Task 2、3）。
3. 候选改变块归一化及邻居恢复：需重算受影响整块，不能仅替换目标点输出；候选不能消耗真实路径 RNG（Task 2）。
4. 初始全 q0 或极小预算：仍有完整关闭点反馈；无可行结果时保存诊断，不能强行导出“达标”结果（Task 1、3）。
5. 控制/校准/导出中断：预算基准、最佳可行图、游标、视角与所有 RNG 必须精确恢复，不能重复控制更新或让最终测试参与选择（Task 3）。

## Builder choices (implementation defaults, not literature optima)

- 初始价格 0.01、价格步长 0.05、EMA 系数 0.9；分别提供显式 CLI 参数并写入配置。
- 镜像更新的 keep/tier 步长各默认 0.1；两分支的失真与成本项均按固定 `N_source` 全局增益处理，避免逐行归一化消除收益幅度。增益与两步长写入日志；不得随单点梯度大小变化。
- 从源记录的训练相机中均匀留出 4 个校准视角；相机不足时要求用户显式缩小校准/窗口数量，不能混入最终测试相机。
- 默认每 250 次更新评估校准质量，每 100 次更新保存恢复点；仅固定部署预算达标的候选可参与 best-feasible 排名。
- 默认每 250 次更新抽查至多 4 个候选干预并真实重渲染，保存一阶收益符号/排序误差；该上限仅针对额外核验，不限制 q0 训练反馈。

---

### Task 1: Budget price, mirror updates and recoverable coverage

**Files:**
- Create: `gaussian_jscc/allocation_budget.py`
- Create: `gaussian_jscc/allocation_mirror.py`
- Test: `tests/test_allocation_budget.py`
- Test: `tests/test_allocation_mirror.py`

**Interfaces:**
- `BudgetController(budget, initial_price, dual_lr, ema_decay)` exposes `price`, `observe(actual_uses) -> dict`, `state_dict()`, `load_state_dict(state)`.
- `MirrorTierTable(count, keep_probability=.99)` retains the sampling/probability interfaces of `GaussianTierMask`; adds `update(keep_costs[N,2], tier_ids[M], tier_costs[M,3], keep_lr, tier_lr, global_gain)` and explicit state serialization. Updates use detached action costs, not Adam or per-row normalized logit gradients.
- `BlockVisitScheduler(id_batches)` selects at most one actual retained source row per spatial block, returns original-row IDs and batch/block offsets, records evaluated rows and restores per-block cursors. `coverage()` reports evaluation counts, last visit and complete coverages over all source rows; also reports currently retained rows that remain unvisited.
- `ViewWindowScheduler(train_names, seed)` returns a shuffled coverage traversal of `window_views` distinct cameras and serializes its generator, permutation and cursor.

- [ ] Write failing tests: price increases after over-budget observations and decreases after under-budget observations; lambda is nonnegative; EMA round-trip produces identical subsequent updates; invalid/nonfinite costs fail without mutating state.
- [ ] Write failing tests: symmetric initialization; smaller persistent action costs produce smaller probability changes than larger costs; one global gain is shared by image/cost; probabilities remain finite and normalized after extreme finite updates.
- [ ] Write failing tests: unvisited tier rows remain bitwise unchanged; visited q1–q3 rows alone update; q0 keep rows can change; no tier quotas; padding and all-q0 blocks yield no positive candidate; cursor coverage and shuffled view order resume exactly.
- [ ] Run `python -m unittest discover -s tests -p test_allocation_budget.py -q` and corresponding mirror tests; confirm failures are due to missing behavior.
- [ ] Implement the four interfaces with log-domain updates, max subtraction and validation before mutation; do not call `deployment_probabilities` in the training loss.
- [ ] Run both test files green; commit Task 1 files only.

### Task 2: Point intervention kernel with full q0 and streaming multiview feedback

**Files:**
- Create: `gaussian_jscc/budgeted_allocation_kernel.py`
- Test: `tests/test_budgeted_allocation_kernel.py`
- Reuse, unchanged: `successive_codec.py`, `allocation_reuse.py`, `render_objective.py`, `optimization.py`, `data.py`.

**Interfaces:**
- `mixed_point_candidates(model, features, q, target_columns, snr, kind, geometry, *, semantic_y, actual, actual_rng, timings)` yields detached per-block raw-attribute differences for q1/q2/q3. Current tier has zero difference; other candidates change only each block's target tier, and re-decode every retained row in the affected block.
- `budgeted_mask_step(model, table, feature_batches, id_batches, geometry, task, scheduler, snr, kind, *, d_ref, price, budget, side_proxy, timings) -> dict` performs one scene/window's forward and streaming scene gradient, computes keep/tier action costs and updates only scheduler coverage (caller applies mirror update once).
- Returned dict contains `render_loss`, keep action costs, visited original-row IDs, positive-tier action costs, expected/sample proxy costs, q0 coverage, candidate calls/cache bytes and timing fields. Codec parameter gradients remain absent.
- `audit_interventions(...) -> list[dict]` compares selected one-point q1–q3 and q0 interventions with fresh decoded-scene rerenders; q0 retain-set changes always re-encode semantic features.

- [ ] Write failing tests with a tiny real successive codec: actual hard forward, delivered scale frames and post-call RNG equal the existing actual communication path for the same q.
- [ ] Write failing tests: neighboring q values remain bitwise unchanged; candidate block outputs equal independent fresh full-block decoding with paired noise; non-target retained neighbors' response is included in the VJP; original row IDs survive nontrivial block reordering.
- [ ] Write failing tests: every sampled q0 has a hard-zero splat and finite gate derivative; all-q0 blocks still produce feedback; hypothetical added points cannot change the live scene/context.
- [ ] Write failing tests: four streamed camera losses/gradients equal a direct mean over the same scene; codec/stats unchanged; candidate RNG/cache cannot escape the update; transient cache is released before next update.
- [ ] Run `python -m unittest discover -s tests -p test_budgeted_allocation_kernel.py -q`, confirming red before production implementation.
- [ ] Implement actual scene construction first, save the raw-scene gradient, then process actual/candidate batches with step-local semantic/actual/q0 reuse. Contract gradients against all affected retained rows. The q0 shadow path uses freshly encoded counterfactual retained sets; no source attributes/latent enter the real receiver.
- [ ] Implement the audit path independently of the first-order proxy and test its signed measured changes. Do not substitute two global random mask renders for local training feedback.
- [ ] Run kernel tests green; compare call counts/cache peaks against existing `local_mask_step(reuse_codec=True)` on the same CPU fixture; log measurements without claiming CUDA speedup; commit Task 2 files only.

### Task 3: Independent experiment, exact cost, source isolation and resume

**Files:**
- Create: `gaussian_jscc/budgeted_allocation.py`
- Create: `gaussian_jscc/budgeted_allocation_inputs.py`
- Create: `gaussian_jscc/budgeted_allocation_state.py`
- Create: `gaussian_jscc/budgeted_allocation_evaluation.py`
- Test: `tests/test_budgeted_allocation.py`
- Reuse: `successive_allocation.source_plan` for validated Joint A, `successive_transfer.load_source` for completed shared models, `successive_multiscene_inputs.make_scene`, `modular_allocation.measured_cost`, `allocation.deployment_tiers`, transport and tier visualization helpers.

**Interfaces:**
- `parser()` has required `--run --scene --out`, optional relocation `--manifest`, a required mutually-exclusive budget group, `--allocation-steps 2000`, controller/mirror/window/validation/profile/audit settings, `--resume`, `--dry-run`; no joint or beta training switch.
- `load_inputs(args)` validates a completed source and an explicitly selected train-role scene; returns CPU checkpoint/config/source identity/spec/old scene record/recorded net bit/use. Shared-model heldouts are rejected as allocation training inputs.
- `ExactDeploymentCost(model, scene, codec_id, snr, bits)` returns the same complete component accounting as `modular_allocation.measured_cost`; caches XYZ streams by exact retained-set identity. Metadata/tier bytes and scale framing remain map-dependent and measured, not treated as constants.
- `save_state(...)` / `restore_state(...)` atomically persist model identity, table, controller, baseline/q3 cost/B/D_ref, row/camera mapping, schedulers, all RNG, pending control/eval/audit state and best-feasible binding. No transient decode caches are serialized.
- `evaluate_map(model, scene, q, cameras, args, *, save_images=False)` reports source/photo PSNR/SSIM/LPIPS, exact side bytes/uses and tier distribution without sampling a new map; returns rows rather than selecting a checkpoint.

- [ ] Write failing dry-run tests: missing/conflicting budget, invalid source/scene/side-rate assumption, input/output nesting or changed scene identity fail without creating output or rendering.
- [ ] Write failing source tests for real small completed Joint A and shared-model fixtures; reject unfinished checkpoint and heldout selection. Verify calibration cameras are removed from allocation windows and final tests remain untouched until final selection.
- [ ] Write failing cost tests with nonidentity original-row/block mapping: exact payload + retained XYZ + complete metadata + scale frames equal existing packet accounting, including all-q0; XYZ cache hits when retained set is unchanged. No compression runs inside point candidates or every optimization step.
- [ ] Write failing budget tests: ratio is floored from one saved measured q3 reference; expected-under/ten-draw-over policy is infeasible; no feasible checkpoint means `budget_unmet`, with a lowest-violation diagnostic and no fake best-feasible export.
- [ ] Write failing end-to-end CPU tests: one actual update per four-view window, frozen model/stats, stable fixed deployment seed, selected q bound to codec ID. Resume mid-window boundary/control/evaluation with bitwise identical tables, controller, coverage, RNG and subsequent map.
- [ ] Run `python -m unittest discover -s tests -p test_budgeted_allocation.py -q`, confirm red.
- [ ] Implement preflight first; initialization measures q3 cost and D_ref once. Use a positive numerical floor for exact-zero D_ref and record it. Resume restores these values without remeasurement.
- [ ] Implement allocation loop using Tasks 1–2. Refresh actual ten-draw deployed bytes/cost only at control/validation/export nodes, update controller exactly once at its nodes, and use the last measured side-cost anchors between nodes. Sampled/expected per-step costs are explicitly labeled proxies.
- [ ] At calibration nodes, rank only actual-cost-feasible deployed maps by calibration MSE. Save a bound q/table/model identity snapshot; test cameras cannot choose it. At finish, test initial/latest/selected endpoints, saving selected representative panels only once if endpoints coincide.
- [ ] Record actual and candidate codec calls, q0 path, per-view forward/backward, mirror update, cost/control/audit time, coverage and GPU peak. Detailed CUDA events default every 10 updates; no per-block synchronization.
- [ ] Run entry/state/cost tests green and the targeted legacy `test_successive_allocation.py` and `test_allocation_reuse.py` tests to prove existing behavior unchanged; commit Task 3 files only.

### Task 4: Server launcher and compact experiment outputs

**Files:**
- Create: `scripts/test_budgeted_allocation.sh`
- Create: `gaussian_jscc/budgeted_allocation_report.py`
- Create: `docs/budgeted_allocation.md`
- Test: `tests/test_budgeted_allocation_report.py`
- Update: spec status after implementation/verification, without rewriting agreed requirements.

**Interfaces:**
- Launcher accepts `COMPLETED_RUN SCENE NEW_OUT` and requires an explicit budget flag; requires explicit `CUDA_VISIBLE_DEVICES`, supports `RESUME=1`, propagates arguments exactly and exits with child status.
- `summarize(out)` reads saved numeric records and writes `comparison/endpoints.csv`, a compact JSON summary, budget/cost/price curves, hard tier shares and candidate-coverage charts. Reports optimization step p50/p90 separately from evaluation/control wall time.

- [ ] Write failing tests: report cannot label budget violation feasible; rate figures use total complex channel uses, not equivalent MB without bit/use assumptions; selected map count matches bound checkpoint; missing optional LPIPS is explicit.
- [ ] Write launcher integration tests for required budget, defaults 2000/four views, forwarded source/scene, no output mutation on dry-run, resume flag and failed child exit status.
- [ ] Run report tests red, then implement launcher/report/docs with copyable nohup and exact-resume commands. Explain controller defaults, proxy limits, original-row alignment, incomplete positive-candidate coverage and q0 visualization meaning.
- [ ] Run new directed tests, targeted legacy allocation/reuse tests, `bash -n scripts/test_budgeted_allocation.sh`, `python -m py_compile` for new modules, and `git diff --check`. No full historical regression or full-scene CPU/CUDA training.
- [ ] Request a fresh read-only review of the entire change against the spec; fix Critical/Important issues with reproducing tests.
- [ ] Commit only this task's files; push main under the user's standing instruction; verify remote HEAD. Hand off the commit, verification scope, required-budget startup command and measured CPU call/cache comparison. Actual CUDA timing remains a server experiment.

## Self-review

- Spec sections 1–4 map to Tasks 1 and 3; section 5 maps to Task 2; section 6 maps to Tasks 1–3; sections 7–8 map to Tasks 3–4; all ten acceptance items have directed tests above.
- Original experiments and existing dirty files are outside implementation scope. Any shared helper change must preserve default behavior and receive a targeted regression test.
- The five Review Focus cases have explicit tests in their owning tasks. This plan adds no forced tier proportions, q0 cap, implicit budget, final-test-based selection or joint finetuning.
- No production code has been written for this plan. Implementation begins after plan confirmation, in the current working repository, preserving the user's prior in-place workflow preference.
