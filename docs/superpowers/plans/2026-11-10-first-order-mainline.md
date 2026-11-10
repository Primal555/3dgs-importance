# 一阶分配主线恢复 Implementation Plan

> **For agentic workers:** 使用 superpowers:executing-plans 执行；本轮为入口与文档恢复。

**Goal:** 将已有校准一阶分配器设为当前主线，提供明确的启动命令。

**Architecture:** 统一脚本直接转发至历史 structured evidence 入口。保持原算法和训练默认值，保留当前编解码器与维护修复。

**Tech Stack:** Bash、Python、PyTorch。

**Spec:** [当前主线](../../mainline_first_order.md)。用户已确认恢复旧一阶路线。

## Global Constraints

- 在当前仓库操作，保留其他未提交内容及独立实验入口。
- 本轮仅恢复入口与说明，算法、超参数和停止规则沿用原实现。
- 验证完成后提交本轮文件并推送 main。

## Review Focus

- 历史基点须对应两组实际完成的 calibrated evidence 实验。
- 启动参数、RESUME 和退出码通过原脚本原样处理。
- beta 必须显式指定，训练目标与历史条件一致。
- 共享 kernel 在无神经 provider 时须保持原数学行为。
- README 的当前主线标识须与启动路径一致。

## Task 1：恢复主线入口

- [x] 核对 `c1162c2`、历史 plan.json 与当前默认参数。
- [x] 对照历史 kernel 与当前 kernel 的 CPU 小块反馈和 RNG。
- [x] 新增 `scripts/train_mainline_allocation.sh` 和主线说明，更新 README。
- [x] 检查 Bash 语法、真实 dry-run 参数与既有定向测试：21 passed，1 skipped（CUDA）。
- [x] 检查 diff，确认主线入口直接转发原脚本，其他工作区内容保持原状。

交付：提交以上入口和说明，推送 main，提供服务器拉取与启动命令。

本轮属于可逆的启动别名与文档修改，使用既有算法/恢复测试及实际命令验证，保持测试范围集中。
