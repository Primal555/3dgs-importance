# Direct RD Allocation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this approved design in the current repository. Preserve existing uncommitted work.

**Goal:** 冻结收益网络，从全 q2 直接换挡并通过真实整体 RD 验收。
**Architecture:** 决策内核、独立实验入口和精简报告三个模块，复用已有批量空间特征及评估器。
**Tech Stack:** PyTorch、NumPy、pytest、Bash。
**Spec:** `docs/superpowers/specs/2026-11-02-direct-rd-allocation-design.md`

## Global Constraints

- 独立入口，冻结网络，无新增 replay，默认 2000 窗口、批次 128、10 窗口验收。
- 数据划分、归一化和完整成本口径保持可追溯，测试图像不参与提案验收。
- 所有已有用户改动保留；通过定向验证后提交本轮文件并推送。

## Review Focus

1. 新地图必须与窗口重建一致，q0 无投票门槛。
2. 所有提案来自同一已接受背景，同块冲突不同时应用。
3. 真实 RD 恶化时不可修改已接受地图，回退使用同一噪声条件。
4. 中断后恢复累积提案、已接受得分及调度状态，不重复最终指标。
5. 结果包不包含 replay、优化器、checkpoint；恢复独立且绑定原模型身份。

## Task 1：决策和整批验收

- [x] 添加 `tests/test_direct_rd_decisions.py`，先观察缺失模块失败。
- [x] 实现 `direct_rd_decisions.py`：HardTierMap、ProposalBank、audit_proposals。
- [x] 验证删除、同块排序、重复候选平均、缩步与拒绝。

## Task 2：独立入口和冻结恢复

- [x] 添加入口、CPU 实际窗口和断点恢复测试，先观察失败。
- [x] 实现 `direct_rd_allocation.py`、冻结推理数据载入、校准评估缓存、稀疏 checkpoint。
- [x] 验证匹配原场景与源模型、冻结哈希、只使用训练窗口和校准相机。

## Task 3：输出与交付

- [x] 实现 `direct_rd_artifacts.py`，测试自动精简包及 q2 对照摘要。
- [x] 添加启动脚本及运行说明，记录实际步数和验收开销。
- [x] 执行相关定向测试、Bash 语法及差异检查，独立审查后提交推送。

## 执行记录

按用户已经确认的路线直接执行；不新增交替训练。第 700 窗口权重存在性无法在本地确认，默认明确读取已完成源实验的 final checkpoint，报告实际来源窗口。

定向验证：direct allocation/decisions、batched spatial features/window/entry、candidate noise、calibrated direction，共 23 passed、2 skipped（CUDA）。Bash 语法检查通过。独立审查未发现阻断或重要问题。未运行本地全场景训练或全仓历史回归。

本轮恢复测试覆盖：未验收候选中断、完整运行一致性、重复完成调用、源收益 checkpoint 移走后恢复、配置变更拒绝。修复了复用 argparse 对象时将运行期 ImageDistortion 纳入计划的问题。
