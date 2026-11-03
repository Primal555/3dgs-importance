# 混合正档位评估实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 从头预训练 700 窗口空间收益网络，自动进入密集一阶保留＋稀疏网络正档位分配。

**Architecture:** 复用空间预训练入口和旧校准分配器，为正档位估值增加可选 provider。独立 provider 复用本轮场景生成空间输入，冻结/在线两种模式显式记录。

**Tech Stack:** PyTorch、现有渲染器和 replay、Bash、pytest。

**Spec:** `docs/superpowers/specs/2026-11-03-hybrid-tier-benefit-design.md`

## Global Constraints

- 编解码器冻结；keep 分支禁止被网络预测覆盖。
- 默认 700＋2000，自动衔接；不加载历史收益权重。
- 只做定向检查，保留其他改动；完成后提交推送。

## Review Focus

- 四状态值差分的 D_ref 与 lambda 单位。
- 场景、采样与噪声复用。
- 冻结模式没有网络训练、online replay 独立并可恢复。
- 阶段切换与历史纠偏恢复状态完整。
- 紧凑导出、初末图和结果包。

## Tasks

- [x] 编写失败测试，验证 keep 路径一致和正档位替换。
- [x] 实现可选 kernel provider、空间预测 adapter 及分配状态扩展。
- [x] 实现自动两阶段入口、紧凑网络导出、脚本和文档。
- [x] 验证小规模冻结/在线、中断恢复、原校准路径；独立审查。
- [x] 提交推送并给服务器命令。

## 验证记录

新定向用例共 6 项，覆盖同场景 keep 原始反馈逐项相等、全部关闭点持续反馈、默认 700 自动流程、真实空间网络的冻结/在线运行、阶段间与阶段内精确恢复，以及结果包和源 replay 保护。旧校准、kernel 和早期收益路径的相关检查 21 项通过，1 项 CUDA 检查因本地无 CUDA 跳过。默认 Git Bash 启动入口 dry-run 独立复验通过。独立审查发现的默认 beta 必填、在线教师计时、候选解码计数问题均已修复并复核。
