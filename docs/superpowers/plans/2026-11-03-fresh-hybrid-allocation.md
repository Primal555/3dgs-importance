# 从头分配实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 固定现有收益网络，完整重启分配学习，默认 2000 步。

**Architecture:** 新独立入口复用 engine 的 fresh 初始化路径，读取源配置与紧凑网络。沿用 TierStepTable 和 HybridPositiveProvider。旧 checkpoint、预训练循环均不进入新启动。

**Tech Stack:** PyTorch、Bash、pytest。

**Spec:** `docs/superpowers/specs/2026-11-03-fresh-hybrid-allocation.md`

## Global Constraints

- 不继承旧概率、历史、纠偏、覆盖、相机或 RNG 状态；冻结编解码器和收益网络。
- 原空间输入与 RD/候选/评估条件相同；默认正档位上限 1.0、分配 2000 步。
- 不加载旧分配大 checkpoint；源只读，output 独立。
- 仅定向验证，提交推送当前任务文件。

## Review Focus

- 首次 kernel 看到的状态全部初始化，网络参数来自原 700 窗口。
- 新入口/恢复需要原编解码器与场景，恢复不依赖源分配目录。
- 同配置 resume 精确一致；更改上限或预算必须新 OUT。
- 源路径形式及目录嵌套不能写入原运行。
- 紧凑包只含评估材料，网络/训练状态单独保留。

## Tasks

- [x] 失败测试：首步初始化、无旧 checkpoint、冻结和恢复。
- [x] 新入口、只读网络副本和阶段输出。
- [x] 脚本与使用文档，边界/兼容定向测试。
- [x] 独立审查，提交推送及启动代码。

## 验证记录

- 初次全流程验证：3 项通过，包含限幅/不限幅连续与恢复一致。
- 最终新入口、步幅表与原入口定向组合：12 项通过。
- 首次 kernel 的保留先验、三档概率、历史和计数均符合初态。
- 源分配 checkpoint 隐藏后启动成功；源分配目录移走后仍可恢复。
- 独立只读审查未发现实质问题；结果包排除 checkpoint 和网络。
