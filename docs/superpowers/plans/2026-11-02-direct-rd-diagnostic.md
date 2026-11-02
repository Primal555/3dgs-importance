# Direct RD Diagnostic Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the approved diagnostic scope in the current repository.

**Goal:** 固定停滞地图，以同条件标签和分组实测定位换挡提案失败原因。
**Architecture:** 配对候选诊断窗口、分层报告和只读独立入口；原训练入口不变。
**Tech Stack:** PyTorch、现有精确 tile renderer、pytest、Bash。
**Spec:** `docs/superpowers/specs/2026-11-02-direct-rd-diagnostic-design.md`

## Global Constraints

- 固定原始 checkpoint、源模型和地图，全部测试输出写入新目录。
- 不训练网络，不覆盖原日志，不保存大型 replay 或图片。
- 使用现有固定噪声槽及完整成本口径，记录所抽动作与评价条件。

## Review Focus

- 候选窗口重放与原预测对应；同条件标签和完整渲染一致。
- 高分／普通样本划分、q0 锚与新旧档位索引正确。
- 更换视角和更换噪声分别比较，避免混淆原因。
- 单点真值和联合真值使用相同视角、噪声及成本口径。
- 源输出只读，暂停命令严格匹配用户、模块和路径。

## Tasks

- [x] 先写分层选择、配对窗口和报告归因测试，观察缺失模块失败。
- [x] 实现配对窗口，复用候选整块解码、空间输入与精确局部标签。
- [x] 实现固定 checkpoint 入口、完整渲染一致性检查及非重叠分组。
- [x] 实现精简报告、启动／暂停脚本和说明，执行定向测试与独立审查。
- [ ] 提交本轮文件、推送 GitHub，交付暂停和启动命令。

## 验证记录

定向测试覆盖诊断内核、报告、只读入口、相机选择、进程匹配及直接分配／批量投影／固定噪声回归，共 19 passed、1 skipped（CUDA）。Bash 语法检查通过，未跑全仓历史训练回归。

独立审查指出并修正：通用场景加载器会读取测试图像、原始逐种子收益被均值覆盖、样本不足缺少明确状态。新增选择性 COLMAP 训练图像加载和相应测试，保留 condition_samples，样本不足时停止机制主因排名。复审未发现剩余重要问题。
