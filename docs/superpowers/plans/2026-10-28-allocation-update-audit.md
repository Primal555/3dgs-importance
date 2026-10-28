# Allocation Update Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在固定分配器上匹配审计实际更新代价。

**Architecture:** 复用真实 kernel 和历史更新逻辑，添加只读 inspector；单独实现标签渲染及短审计入口。

**Tech Stack:** PyTorch、pytest、Bash。

**Spec:** `docs/superpowers/specs/2026-10-28-allocation-update-audit-design.md`

## Global Constraints

- 不改分配训练算法、loss、beta、采样或部署规则。
- 每组20窗口、每窗口最多32次反事实；不训练新模型。
- 保留现有未提交文件，先交付服务器版本，再做真实结果本地检查。

## Review Focus

- 标签不进入被评估的历史预测。
- q0条件混合须与三档精确标签匹配。
- 真实模型、概率、历史和源文件只读。
- CPU/GPU设备和配对RNG一致。
- 短审计来源、训练相机和新输出目录须验证。

### Task 1: 匹配的动作审计

- [x] 写失败测试：假想更新与真实更新一致、q0条件混合、历史冻结、无标签泄漏。
- [x] 实现 inspector、历史原位恢复和 kernel 可选回调。
- [x] 定向单元与小型 kernel 测试通过。

### Task 2: 服务器入口与交付

- [x] 写失败测试：checkpoint加载、训练视角、dry-run、输出隔离和只读保证。
- [x] 实现入口、两组顺序启动器、数值汇总与文档。
- [x] 定向验证和独立审查。

## 验证记录

- 11个相关测试文件：77 passed、2 skipped、7 subtests passed，耗时38.38秒。
- Bash语法、目标Python文件编译与Git差异检查通过。
- 独立审查发现并修正CUDA条件概率设备、q0未就绪标签参照、固定10dB条件恢复与汇总文字。
- 启动器dry-run、小型真实入口、源检查点哈希不变、概率不变和预测历史恢复均已验证。
- 本机无CUDA，GPU专项用例留待服务器运行。尚未开始两组真实结果的本地统计检查。
