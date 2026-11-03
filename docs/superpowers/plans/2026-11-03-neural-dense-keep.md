# 全点神经保留实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现全点密集训练的保留网络，配合稀疏空间正档位网络，独立并行对照。

**Architecture:** 复用混合版两阶段流程，新增保留网络及概率执行层。kernel 增加可选神经门控采样/密集 VJP 与有限监督钩子，原路径默认不变。

**Tech Stack:** PyTorch、现有渲染器、Bash、pytest。

**Spec:** `docs/superpowers/specs/2026-11-03-neural-dense-keep-design.md`

## Global Constraints

- 默认 700＋2000，空间正档位网络冻结，保留网络在分配阶段持续学习。
- 原混合入口与其他未提交文件保持不变。
- 样本噪声、概率平滑和恢复必须一致；只提交本轮改动并推送。

## Review Focus

- ST 门控与真实采样使用相同 Gumbel，概率平滑梯度不中断。
- packet/original 行映射；动态观测为下一步输入。
- q0–q3 标签的条件混合及有限渲染预算。
- 全点网络梯度、冻结正档位和精确恢复。
- checkpoint 缓存大小与 review 包。

## Tasks

- [x] 失败测试：直接/分块梯度、保留表执行、标签混合。
- [x] 保留策略、观测提取及真实监督。
- [x] kernel/状态/两阶段独立入口与文档。
- [x] 定向兼容、完整流程恢复和独立审查。
- [x] 提交推送并给启动代码。
