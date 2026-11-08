# 正档位独立学习与真实反馈检查 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 复用已有候选区域标签，检查隔离共同价值监督后，正档位选择能否同时改善普通样本与严重错误。

**Architecture:** 冻结原始共同价值网络，复制独立点编码器与正档输出头，加入已有区域 CNN。仅训练中心化价值、三个正档差值及其 RD 方向；物理单位与固定尺度保持一致。复用已有原结构／区域结构末步预测作为参照，新增末步训练与精确恢复支持。真实反馈检查区分历史预校正、当前精确替换及后续动作监督。

**Tech Stack:** PyTorch、固定 SpatialReplay、JSONL、Bash；默认 CPU。

**Spec:** 本轮用户已批准的独立分支、直接差值监督、现有数据离线验证方案。

## Global Constraints

- 当前仓库新增实验入口，保持现有训练行为，保留用户其他未提交文件。
- 不训练编解码器、不采新标签、不更新概率表、不重跑旧两组。
- 同源初始权重、4096／512 标签、相同批次与固定种子，默认 2000 次更新。
- 普通样本、严重样本、活跃样本、剔除最大收益记录均须显示；不以验证集选模。
- 下载包仅保留配置、指标、逐样本预测与反馈检查，权重／回放保留服务器。
- 只运行定向测试，验证后提交推送。

## Review Focus

- 部分候选与 q0 相同时，初始预测仍应与原网络一致，且共同价值在更新后不漂移。
- 训练／验证源隔离、旧参照数据身份与标签顺序必须逐条核对。
- 当前精确替换与历史残差预校正发生时间不同，旧日志不能冒充新增统计。
- 中断恢复必须恢复批次顺序、优化器和已完成次数；完成阶段可重复生成小包。
- 训练／数据／参照路径不得相互覆盖。

### Task 1：独立网络与正档监督

Files: `gaussian_jscc/benefit_isolated_positive_model.py`、`tests/test_benefit_isolated_positive.py`。

- [x] 编写并运行失败测试：初始等价、部分相同候选、共同价值固定、独立参数梯度、共同偏移不改变正档损失。
- [x] 实现冻结共同网络和独立正档编码器；相同候选的零先验与共同均值同时保持。
- [x] 训练只使用相同 RD 尺度的中心化两分量与三组差值，方向监督不包含 q0。
- [x] 运行定向测试。

### Task 2：固定数据对照入口与报告

Files: `gaussian_jscc/benefit_isolated_positive_controls.py`、`gaussian_jscc/benefit_isolated_positive_report.py`、`tests/test_benefit_isolated_positive_controls.py`。

- [x] 编写并运行失败测试：真实小回放只读训练、身份不一致拒绝、保存恢复等价、逐样本剔除报告。
- [x] 加载旧对照的数据与初始权重，逐条核对参照预测的 ID、真值和成本。
- [x] 复用旧洗牌顺序与逐更新种子，新增固定训练探针和训练／验证分层指标。
- [x] 保存 latest／final、末步预测、小型 review.tar.gz；不保存中间图片。
- [x] 运行定向测试。

### Task 3：真实反馈口径与交付

Files: `gaussian_jscc/benefit_feedback_audit.py`、`gaussian_jscc/budgeted_allocation_kernel.py`、`gaussian_jscc/structured_allocation_finite.py`、`scripts/test_benefit_isolated_positive.sh`、`docs/benefit_isolated_positive.md`。

- [x] 添加真实反馈流量检查，旧日志对未记录的当前替换明确留空。
- [x] 为后续在线运行新增当前精确替换的数量与动作差值一致性统计，不改变更新逻辑。
- [x] 验证 Bash 语法、恢复路径、输出包；执行相关定向检查及独立审查。
- [ ] 提交并推送本轮文件，给出服务器拉取与 CPU 启动命令。

## 验证记录

2026-11-08：7 个相关测试文件共 36 项通过；新增报告在真实 512 条下载预测上复算，复现 36 条改善、61 条恶化及剔除最大收益后恶化 26.84%。Bash 语法通过。独立只读审查无重要缺陷。审查过程中移除了审计路径自动发现，默认训练仅依赖第二组目录。
