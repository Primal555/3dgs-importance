# 候选渲染响应统一价值网络实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** 建立独立、不使用一阶输入的统一四档收益评估与分配实验。

**Architecture:** 候选块配对解码，投影覆盖patch与深度矩上下文，统一价值MLP，真实四状态监督，共用既有通信计量和部署框架。

**Tech Stack:** PyTorch、既有Gaussian rasterizer、pytest、Bash。

**Spec:** docs/superpowers/specs/2026-10-29-response-value-design.md

## Global Constraints

冻结编解码器；0/8/24/48；10 dB；可靠16bit XYZ；MSE＋SSIM；十次部署；无一阶输入；旧入口兼容；当前仓库实现；定向回归。

## Review Focus

- q0转换重新语义编码，所有候选复用同一噪声起点。
- 标签生成前冻结输入，真实干预图像不泄漏。
- 离屏、空块、不同椭球尺度与GPU内存峰值。
- 只更新实际评价点，两条分支覆盖一致。
- 断点恢复的网络、replay、覆盖、RNG和概率表精确一致。

### Task 1：投影与统一价值网络

**Files:** response_value_features.py、response_value.py、tests/test_response_value.py。

**Interfaces:** project_footprints(raw,camera)、response_features(...)、ResponseValueNetwork.forward/features values、ResponseValueLearner。

- [x] 先写投影、覆盖patch、零锚点／反对称、无aux依赖、数值回归测试并观察失败。
- [x] 实现几何patch特征与统一数值网络，运行定向测试至通过。

### Task 2：无梯度候选窗口与完整标签

**Files:** response_value_window.py、response_value_report.py、tests/test_response_value_window.py。

**Interfaces:** ResponseValueWindow.step(...)返回旧分配kernel兼容结果；records供既有replay；response_validate_network(learner,step)。

- [x] 先写同噪声单点干预、q0重编码、候选邻居变化、真实标签输入隔离、覆盖／预算测试并观察失败。
- [x] 实现窗口、完整组监督和报告，定向测试至通过。

### Task 3：独立入口、续训与交付

**Files:** learned_benefit_allocation.py、budgeted_allocation_state.py、budgeted_allocation_kernel.py、独立shell脚本及文档、tests/test_response_value_integration.py。

**Interfaces:** --benefit-variant response；独立family；新launcher固定新变体。

- [x] 先写入口dry-run、旧plan兼容、冻结模型、预训练与分配续训测试并观察失败。
- [x] 实现最小factory／dispatch改动和运行说明，定向回归。
- [x] 新上下文独立审查并修复4项重要问题；44项测试及7项子测试通过。

交付：提交并推送；服务器预训练和分配启动指令见`docs/response_value_allocation.md`。
