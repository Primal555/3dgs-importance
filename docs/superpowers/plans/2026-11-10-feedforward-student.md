# 一次前馈学生分配器实施计划

> **For agentic workers:** 使用 superpowers:executing-plans 顺序实现，末尾做一次独立审查。

**Goal:** 新场景预处理后一次前馈产生四档，支持教师动作蒸馏、冻结编解码器的 RD 训练、恢复与直接预测。

**Architecture:** 三支路输入，两级空间残差骨干，保留／正档位双头；硬动作执行，当前场景一阶反馈驱动共享网络；均匀稀疏正档位通过纳入概率加权。

**Tech Stack:** PyTorch、NumPy、SciPy cKDTree、现有3DGS光栅器及精确可见性扩展。

**Spec:** `docs/superpowers/specs/2026-11-10-feedforward-student-design.md`

## Global Constraints

- 独立分支 `experiment/feedforward-student`，保留已有未提交文件。
- 固定10dB与0/8/24/48，beta0.5，source PLY MSE＋SSIM，现有完整通信账本。
- 原分配器入口保留；新功能以新模块完成。

## Review Focus

- 原PLY、包顺序、空间层级索引，尾块padding和全q0场景。
- 当前确定性策略下均匀采样概率；梯度缩放及keep/tier的交叉路径。
- 教师身份、缓存哈希、测试相机隔离、可见性光栅器一致性。
- 大场景空间聚合显存、完整邻域、全局分支梯度、分块重算。
- 阶段切换Adam、断点RNG／采样器／选模恢复以及最终硬计费。

### Task 1: 空间网络与索引

创建 student_policy_model.py、student_policy_graph.py。接口：build_graph(xyz)->CPU tensor字典；StudentConfig；StudentPolicy.forward(attributes,latent,visibility,graph,controls)->(keep_logits[N,2],tier_logits[N,3])。测试先覆盖形状、梯度、孤立／重合点、输入置换以及分块一致性。

### Task 2: 固定预处理与教师输入

创建 student_policy_inputs.py、student_policy_features.py。读取现有load_inputs与完成教师，准备属性、全保留latent、8相机统计、空间图与固定参考。测试教师不匹配拒绝、相机隔离、缓存身份和原行映射。

### Task 3: RD反馈和目标

创建 student_policy_feedback.py。只读的当前策略适配现有kernel；使用均匀无放回候选和无历史当前反馈，输出keep全点和tier稀疏梯度。测试精确分支范围、纳入概率权重、代理目标与有限差分、冻结codec及配对噪声。

### Task 4: 完整训练、恢复与部署

创建 student_policy_experiment.py、student_policy_state.py、student_policy_report.py 和脚本。先做合成相机场景的双阶段／断点／predict 测试，再完成缓存、阶段切换、评价、图像与精简包。保留实际阶段步数和各项耗时。

### Task 5: 验证、审查与推送

只运行本轮和相关定向测试、Bash语法与dry-run验证。一次独立只读审查后修复具体问题，提交并推送当前独立分支，提供服务器fetch/switch/launch命令。

## 实施记录

- Task 1–4 已完成：空间网络、原行预处理、均匀采样VJP、两阶段训练、精确恢复和零更新新场景预测。默认SH3学生参数量为1,470,629。
- 原始CPU单元测试先失败再实现；完成两阶段中断恢复、准备阶段中断复用及独立新场景预测的合成渲染集成测试。
- 使用本地真实共享codec与历史beta0.5教师读取核验，711,691点及D_ref=0.001671516249358023／C_ref=46193136一致。
- 一次独立只读审查确认主路径，指出同PLY相机变动应重算参照和新参照耗时应计入准备。两项均添加RED→GREEN回归测试并修复。
- GPU数学一致性和大场景峰值显存留待服务器验证；本地CUDA用例明确跳过。准备阶段首视角执行主／缓存光栅器一致性检查，RD前执行现有配对候选小检查。
- 最终定向验证47项通过、2项CUDA跳过；两个Bash入口语法与默认参数检查通过。此次未运行全仓历史回归。
