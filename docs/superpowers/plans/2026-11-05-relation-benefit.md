# 关系收益网络实施计划

> 执行：superpowers:subagent-driven-development。用户已批准在当前仓库新增独立实验，保留未提交文件。

**Goal：**实现固定 codec 的关系价值学习、一次前馈四档分配及统一成本评价。
**Architecture：**固定 bank、共享关系模型、精确配对教师、轻量标签回放、末步直接部署。
**Tech Stack：**PyTorch、SciPy、现有 MaskGaussian 光栅器与局部 CUDA 教师。
**Spec：**`docs/superpowers/specs/2026-11-05-relation-benefit-design.md`。

## 约束及审查重点

原行号／Morton 行号严格区分；训练输入无测试照片；配对噪声；q0 重算整块；float64 标签；最终使用统一数字侧信息账本；分批邻域完整；不修改旧入口。不跑全仓回归。完成后提交推送。

## 任务

- [x] 任务 1：`relation_benefit_features.py` 与专属测试，构建 bank、空间与投影邻域。
- [x] 任务 2：`relation_benefit_model.py` 与专属测试，完整邻域闭包、注意力、价值分解、分批推理。
- [x] 任务 3：`relation_benefit_teacher.py`、`relation_benefit_learning.py` 与测试，教师、回放、损失和统一边际成本。
- [x] 任务 4：`relation_benefit_experiment.py`、服务器脚本、定向集成测试，恢复、评价、精简 review。
- [x] 审查与验证：模块边界、真实标签、续训一致、最终成本和输出，修复实际问题。
- [x] 提交推送，给出服务器拉取与启动命令。

## 执行裁定

采用固定候选预处理及四背景条件监督，画质参考为原场景渲染。四条件值等权聚合后一次选档，部署混合状态的真实 RD 单独验证。多背景实现及测试见 `2026-11-05-relation-context.md`。
