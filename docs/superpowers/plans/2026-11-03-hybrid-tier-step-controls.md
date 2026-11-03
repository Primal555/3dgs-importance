# 正档位步幅对照实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 独立检验稀疏正档位的更新强度，保留原混合实验。

**Architecture:** 新建独立步幅表、只读快照/受控分叉入口，复用现有分配 kernel 和 provider。engine 抽取纯 plan 构造函数并增加无训练副作用的 fork_initial 评估。

**Tech Stack:** PyTorch、Bash、pytest。

**Spec:** `docs/superpowers/specs/2026-11-03-hybrid-tier-step-controls.md`

## Global Constraints

- q0 完整保留；空间输入、网络、RD、历史不变；源实验不写入。
- 默认正档位上限 1.0，显式不限幅；默认训练到原 2000 步。
- 快照与分叉身份必须匹配；resume 不重新读取可变化的源 latest。
- 只提交本轮文件，保留其他未提交工作。

## Review Focus

- 源保存时存在 pending 事件，分叉不能跳过相应控制更新。
- 源后续继续变化，新实验始终读取固定副本。
- 相同原上限对照的实际训练随机流和 q0 更新完全一致。
- 空候选、已经接近数值下限、极端成本、无穷或非法上限。
- resume 与 snapshot-only 不覆盖已有目录，紧凑包排除训练缓存。

## Tasks

- [x] 表的失败测试、实现和定量日志。
- [x] 快照/分叉失败测试、入口和 engine 最小钩子。
- [x] 完整小流程、连续/恢复一致性与旧入口定向兼容。
- [x] 脚本、说明、独立审查、提交推送与服务器命令。

## 验证记录

- 表与混合分叉/旧入口组合：17 项通过。
- 审查后补充显式 state.pt 嵌套目录拒绝、raw 分位数和 Bash 入口：8 项通过。
- 同上限分叉与原续训的 table、blocks、views、RNG、冻结 provider 状态逐位一致。
- 独立审查发现的目录保护边界已修正，定向复查通过。
- 结构化旧分配与神经保留入口兼容：9 项通过，1 项 CUDA 检查跳过。
