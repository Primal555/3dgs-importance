# Calibrated Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 独立积累精确纠偏，并使用有效评价量校准概率更新。

**Architecture:** 保留 v1 类，新增稀疏动作纠偏库与 v2 调度器，升级现有 structured 入口。复用候选解码、精确渲染、固定 RD 目标与部署逻辑。

**Tech Stack:** Python、CPU PyTorch、pytest、现有 Bash 启动器。

**Spec:** `docs/superpowers/specs/2026-10-27-calibrated-evidence-design.md`

## Global Constraints

- 不修改损失、beta、符号预算、教师、采样规则及精确渲染预算。
- 直接在当前仓库实施，保留现有用户改动；只提交本任务文件。
- 定向验证；不启动全仓历史实验或全量训练。

## Review Focus

- 纠偏必须按当前动作隔离，精确动作不可重复计权。
- 重复视角不能虚增平均收益的置信度。
- q0 混合值及其不确定性必须使用同一组条件权重。
- 空候选、MSE-only 和全部 q0 均须可运行。
- 完整恢复必须包括稀疏纠偏及视角信息，并拒绝 v1 历史。

### Task 1: 独立纠偏与均值不确定性

**Files:** 新增 `gaussian_jscc/allocation_correction.py`、`gaussian_jscc/calibrated_evidence.py`、`tests/test_calibrated_evidence.py`。

- [x] 先写并运行纠偏持续性、动作隔离、衰减、有效样本量和重复视角失败测试。
- [x] 实现稀疏分量残差库、视角约束 ESS、v2 分量累计和置信度。
- [x] 运行新增测试及 v1 历史测试，验证通过。

### Task 2: 接入与恢复

**Files:** 修改 structured 入口、kernel、engine、报告、启动器及文档；扩展集成测试。

- [x] 添加实际入口采用 v2、精确值优先、全 q0、冻结/RNG、稀疏状态恢复一致性测试，观察失败。
- [x] 接入训练视角、原始 q、校准前后审计字段；升级 family，保存完整统计。
- [x] 定向测试，审查 diff，验证 Bash 预检及报告。
- [ ] 提交本任务代码和文档，推送 origin/main 并核对远端提交。

## 验证记录

71 个定向测试通过，10 个子测试通过；本机无 CUDA，1 个 CUDA 定向测试跳过。覆盖入口、Bash dry-run、冻结教师、RNG、逐项断点恢复、原有预算与固定 beta 路径。独立审查发现的设备转换和 memory=False 不确定性范围问题已修复，并加入对应测试。

71 万点、5562 候选、4 个 CPU 线程的统计调度微基准：3 步分别约0.298、0.278、0.281秒，不含编解码与渲染。保持每步32次精确检查；实际服务器整步性能由训练日志记录。
