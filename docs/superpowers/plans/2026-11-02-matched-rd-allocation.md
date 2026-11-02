# Matched RD Allocation Implementation Plan

> **For agentic workers:** 使用 superpowers:executing-plans 逐项实现。当前仓库操作沿用用户授权，完成后提交推送。

**Goal:** 共同条件重评分、缓存真实复核和有限预算的独立子批验收。

**Architecture:** 提取 PixelWindow 的基线准备方法；新 MatchedWindow 缓存共同条件并缓存少量候选投影；MatchedAudit 在固定背景上累计替换进行精确联合评价。独立入口复用既有输入、冻结权重、完整成本与结果汇总。

**Tech Stack:** Python、PyTorch、现有 tile renderer、pytest、Bash。

**Spec:** `docs/superpowers/specs/2026-11-02-matched-rd-allocation-design.md`

## Global Constraints

冻结两类网络；4优化视角/2配对噪声；512重评分/64真实复核/16动作小批/12联合检查；不修改用户其他文件；定向验证完成后推送。

## Review Focus

- q0/恢复导致整块邻居变化，必须替换完整块。
- 后续子批必须包含此前通过的全部变化，避免重叠区域误算。
- 断点处相机、噪声、候选调度与地图必须一致。
- 缓存必须随地图变化失效，候选缓存有上限。
- 独立测试视角只能用于初始/最终评估，精确成本用于联合验收。

## Tasks

- [x] 先写并运行失败测试：提案覆盖、独立批次与预算、共同条件窗口和精确联合渲染。
- [x] 提取基线准备方法，添加候选观察 hook，实现 `matched_rd_window.py` 的基线和投影复用。
- [x] 实现 `matched_rd_decisions.py` 的覆盖抽样、同条件重排、真实复核和独立小批验收。
- [x] 实现 `matched_rd_allocation.py` 入口、恢复、开销日志和 Bash 启动器及使用文档。
- [x] 定向验证与完整改动审查；按项目默认流程提交推送。

## Ledger

初始状态：HEAD 9b9d696；保留 docs/progressive_mask_full.md 和已有未跟踪文件。设计预算是本轮实现取值，支持参数覆盖；CPU 验证不能代替服务器 CUDA preflight。

Ruling：64个真实标签是监督复核预算，不作为执行动作上限。原实现若只执行64点，会将2000窗口限制在约1.8%的点修改事件。改成真实确认小批优先、其他网络候选独立分组验收，在12次联合检查内保留更高吞吐；增加了可能被拒的网络子批，但不混入已确认子批。

验证：8个受影响测试文件，24 passed、1 skipped，15.08秒；跳过项为 CUDA 环境验证。Bash 语法与启动器 dry-run 通过，完整入口与中断恢复一致。独立审查及吞吐修正复审均未发现 Critical/Important 问题。服务器实际 CUDA 与耗时通过启动 preflight 和运行日志确认。
