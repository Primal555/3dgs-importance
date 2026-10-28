# 逐点换挡收益网络实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现独立的可学习收益分配实验，先完成评估预训练，再显式接入2000步概率分配。

**Architecture:** 冻结现有四模块教师，复用采样、场景梯度、两轮候选、配对渲染和真实计价。收益网络分别预测正档位与保留的两个失真分量，真实标签进入CPU回放，解析通信项在网络外组合。

**Tech Stack:** Python、PyTorch、现有MaskGaussian光栅化与Bash启动脚本。

**Spec:** `docs/superpowers/specs/2026-10-28-learned-allocation-benefit-design.md`

## Global Constraints

- 当前仓库新增独立实验，保留用户已有未提交文件。
- 冻结教师、10 dB、0/8/24/48、XYZ16、MSE＋SSIM、beta=0.1。
- 200个预训练窗口、前50个窗口拟合固定尺度；每窗口32次真实反事实，20个正档位加4组完整q0评价。
- 25%组附加屏蔽一阶前向，权重0.5；每窗口4次Adam、学习率1e-4、128组、梯度上限1。
- 验证块不进入归一化、训练、精确覆盖和误差驱动调度；先预测、后标签、再概率更新、最后训练网络。
- 本轮只运行定向测试，不运行全仓历史回归；服务器画质和CUDA耗时单独确认。

## Review Focus

- 关闭点全部获得保留预测，完整q0组保持同一上下文、条件概率及配对噪声。
- 辅助屏蔽不通过梯度范数或编码偏置留下信息。
- 验证标签和旧动态上下文不会进入训练或当前概率更新。
- 预训练、归一化边界、待执行评估和阶段切换中断后恢复一致。
- 空正档位、纯MSE、无关闭点、异常预算和启动参数均有明确行为。

## Task 1：收益网络、损失与回放

**Files:** 新增 `gaussian_jscc/allocation_benefit.py`、`allocation_benefit_replay.py`；测试 `tests/test_allocation_benefit.py`。

**Interfaces:** `BenefitNetwork(protocol)`、`benefit_loss(prediction,target,rates,conditional,scale,branch)`；`BenefitReplay(capacity,seed)` 的 `add/sample/state_dict/load_state_dict`；`BenefitLearner`负责归一化、预测、小批训练与持久化。

- [x] 先写并运行失败测试：屏蔽输入不受辅助值改变影响，q0混合损失、MSE第二分量严格为0，回放深拷贝、分支均衡与确定性恢复。
- [x] 实现分支MLP、固定归一化与双分量输出、损失、CPU整组回放；每步运行上述定向测试。
- [x] 测量真实训练能拟合一个小型非线性收益样本，验证教师参数不参与优化。

## Task 2：特征与候选调度

**Files:** 新增 `allocation_benefit_features.py`、`allocation_benefit_schedule.py`；测试 `tests/test_allocation_benefit_features.py`。

**Interfaces:** `BenefitFeatures`从codec批次/场景梯度/相机基准图生成 `point/action/views/aux/available`；`BenefitScheduler`提供 `select_rounds/record/state_dict`、块划分、精确动作选择。

- [x] 先写并运行失败测试：相机投影与3×3残差、缺失标记、分量统一单位、块划分稳定且动作覆盖轮换。
- [x] 实现流式特征、解析q0尺度增量、候选/精确监督调度；生产代码不加入测试专用接口。
- [x] 运行任务1和任务2测试，确认空候选与全q0。

## Task 3：实际窗口与真实标签

**Files:** 新增 `allocation_benefit_window.py`；小幅扩展 `budgeted_allocation_kernel.py` 可选收益回调；测试 `tests/test_allocation_benefit_window.py`。

**Interfaces:** `BenefitWindow`在已有kernel中生成收益预测并收集精确组；返回现有 `keep_costs/tier_ids/tier_costs`，额外返回只读抽检记录。

- [x] 先写并运行失败测试：冻结真实小型codec、真实toy渲染，改变标签不改变先验网络预测；验证标签不覆盖概率代价。
- [x] 实现完整q0四状态、正档位干预、训练点精确覆盖；保留所有点的q0预测。
- [x] 测试配对噪声不扰动主随机流、上下文重建、32次上限与纯MSE；确认历史入口默认行为不变。

## Task 4：两阶段入口、报告和精确恢复

**Files:** 新增 `learned_benefit_allocation.py`、`allocation_benefit_report.py`；扩展 `budgeted_allocation_state.py` 新family；测试 `tests/test_learned_benefit_allocation.py`。

**Interfaces:** `main(args,max_updates=None)`；默认 `--phase pretrain`，`--phase allocation --resume`从同一输出继续，阶段不属于不可变配置；其他配置必须精确匹配。

- [x] 先写并运行失败测试：完整短跑与中断恢复等价，预训练不更新概率表，源教师只读，dry-run不建输出。
- [x] 实现固定参考、阶段进度、校准/测试隔离、网络验证选模、实际包成本、端点图像和数值对照。
- [x] 导出三种预测的分量/方向/错误代价、纠错与改错、窗口配对区间、分项耗时与最终部署对照。
- [x] 实现原子快照与日志修复，测试归一化拟合前后和阶段切换恢复。

## Task 5：启动脚本、交付与独立审阅

**Files:** 新增 `scripts/test_learned_benefit_allocation.sh`、`docs/learned_benefit_allocation.md`；补充入口测试。

- [x] 先写并运行脚本行为测试：GPU/源路径/阶段校验、dry-run、resume参数传递。
- [x] 实现启动脚本与服务器两阶段命令；保持只保存初始和最终图像。
- [x] 运行新增及相邻分配测试、Python编译、Bash语法、`git diff --check`。
- [x] 按执行技能要求进行一次独立整体验证审阅，修复实质问题并补充回归测试。
- [x] 仅提交本轮文件；服务器执行所需代码随正常仓库交付流程处理。

## 执行记录

- 2026-10-28：用户已确认设计并要求完整实现，沿用此前明确的当前仓库工作方式，本轮直接执行。
- 接口预检：任务3沿用原kernel返回结构；任务4新增family复用现有原子快照；任务1回放按源点引用静态属性，动态输入以采集时快照保存。
- 独立审阅发现低预算配置、屏蔽附加权重、q0诊断单位和恢复引用生命周期问题，已逐项补充RED→GREEN验证；另外验证了pending事件中断和归一化前断点。
- 报告以独立allocation_step绘图，分配耗时统计排除预训练窗口；相邻历史family保持原有口径。
- 最终定向验证：71 passed、2 skipped、7 subtests passed；Python编译、Bash语法及差异检查通过。跳过项需要CUDA；真实场景画质、显存和速度在服务器确认。
