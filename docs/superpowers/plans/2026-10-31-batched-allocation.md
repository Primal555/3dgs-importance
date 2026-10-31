# Batched Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 以批次 64 执行原分配算法，减少逐候选投影和特征开销，并兼容现有 checkpoint。

**Architecture:** 在 PixelWindow 提取候选批次和视图准备钩子，旧钩子保持逐项实现。BatchedSpatialWindow 统一投影并调用向量化空间特征函数；calibrated 入口以运行时参数选择，原 plan 身份不变。

**Tech Stack:** Python、PyTorch、现有 MaskGaussian CUDA 投影核。

**Spec:** docs/superpowers/specs/2026-10-31-batched-allocation-design.md

## Global Constraints

- 候选数、相机数、真实标签数、全块邻居、配对噪声、尺度帧和损失保持不变。
- 不新增依赖；保留旧实验入口默认路径。
- 批次是运行参数，不能导致旧 checkpoint plan mismatch。
- 当前仓库原地实现，按用户既有授权自动提交推送。

## Review Focus

- 重复原始块和尾块有效点数变化时的顺序与掩码。
- q0 切换对整块重建的影响及配对噪声槽位。
- 四档 projection 在离屏、零存在概率及跨 tile 条件下的等价性。
- 缓存仅在同一窗口、同一相机／目标下复用，不缓存变化的候选状态。
- 运行时批次变更、旧 checkpoint 恢复与继续在线学习的配置边界。

### Task 1: 批量投影、空间特征及候选窗口

**Files:** 新增 batched_spatial_features.py、batched_spatial_window.py；修改 pixel_benefit_window.py；新增对应测试。

**Interfaces:** candidate_batches 产生保持顺序的候选元数据和四档 raw／tiers；candidate_views 产生逐候选有序多相机视图；spatial_inputs_batch 返回有序特征列表。

- [ ] 先测试尾块、重复块、四档状态与旧空间输入逐项一致，真实渲染标签保持一致。
- [ ] 运行新增测试，确认缺少批量接口导致失败。
- [ ] 实现批次钩子、合并投影和缓存特征，保持旧窗口行为。
- [ ] 测试实际小块编解码器的原／快路径、概率更新和标签一致。

### Task 2: 恢复入口、CUDA 预检及交付

**Files:** calibrated_spatial_benefit_allocation.py、pixel_benefit_allocation.py、启动说明及集成测试。

**Interfaces:** --allocation-batch 为运行时可选正整数；同 family／原 plan 继续恢复；记录执行模式与实际批次。

- [ ] 先测试从原路径完成／中断的预训练和 allocation checkpoint 恢复，其他配置变更继续拒绝。
- [ ] 新增同后端的原／批投影与特征预检，CUDA 上正式运行前执行。
- [ ] 跑定向回归和 Bash／diff 检查；记录本地可验证范围。
- [ ] 独立审查整体修改，处理重要问题，提交推送并给服务器恢复命令。
