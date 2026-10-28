# Factored Benefit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** 实现逐点保留价值／升档增量分解和决策导向监督，并接入原分配及固定回放入口。

**Architecture:** 新模型复用特征协议和归一化，q0共享价值与两个增量分开输出；新学习器使用独立损失。在线独立family，离线复用同一回放验证。

**Tech Stack:** Python、PyTorch、NumPy、Bash、pytest。

**Spec:** docs/superpowers/specs/2026-10-28-factored-benefit-design.md

## Global Constraints

- rates为0/8/24/48；10 dB；编解码器冻结；预算及部署规则不变。
- 原legacy入口、旧检查点和用户未提交文件保持原样。
- 只拟合训练标签；固定验证回放；0.25屏蔽率、0.5附加权重。
- 新数值回归系数0.25；方向与排序系数均为1。
- 当前仓库直接实现、定向验证并推送，沿用用户既定工作方式。

## Review Focus

- 目标档位变化不得改变同一q0上下文的共享保留价值。
- 训练用的conditional混合与在线代价一致。
- 新结构不得加载旧结构Adam状态，legacy计划身份不得变化。
- 离线初始化只能使用训练标签拟合尺度，源文件与回放不变。
- 固定档位和命中率指标须正确处理同代价及共享一阶基线。

### Task 1: 分解模型与决策损失

**Files:** gaussian_jscc/factored_benefit.py、gaussian_jscc/factored_benefit_loss.py、gaussian_jscc/allocation_benefit_learning.py、tests/test_factored_benefit.py。

**Interfaces:** FactoredBenefitNetwork保持forward(features,branch,masked)接口；DecisionBenefitLearner保持原回放及训练接口，覆盖losses。

- [x] 先写共享价值、增量重建、辅助屏蔽、混合方向及排序测试，运行确认失败。
- [x] 实现分解头、训练尺度、新损失与学习器钩子。
- [x] 运行新单测及原学习器测试，确认通过。

### Task 2: 在线及离线接入

**Files:** gaussian_jscc/learned_benefit_allocation.py、gaussian_jscc/budgeted_allocation_state.py、gaussian_jscc/budgeted_allocation_report.py、gaussian_jscc/benefit_diagnostics.py、tests/test_factored_benefit_integration.py。

**Interfaces:** --benefit-variant factored选择新family；--fresh-factored只在固定回放中随机初始化新评估器。

- [x] 写在线恢复、旧入口计划、离线源只读和固定回放测试，确认失败。
- [x] 实现选择与状态校验，补充离线源网络基线。
- [x] 运行在线小型codec及离线测试，确认通过。

### Task 3: 评价与服务器交付

**Files:** gaussian_jscc/allocation_benefit_report.py、scripts/test_factored_benefit_allocation.sh、docs/factored_benefit.md，以及相关测试。

**Interfaces:** 报告追加fixed_q1/q2/q3、最佳／预测分布、共同价值与增量误差，既有字段保留。

- [x] 写固定策略和同代价命中率测试，确认失败。
- [x] 实现报告、脚本及离线优先的启动说明。
- [x] 运行定向回归、编译、Bash语法及diff检查。最终77 passed、7 subtests passed。
- [x] 完成独立审阅并补强关键不变量测试；代码交付检查完成。
