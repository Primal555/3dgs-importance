# 物理特征收益评估实施计划

Goal：新增独立实验，比较紧凑收益预测和决策监督。

Architecture：复用 PixelWindow 的配对干预、PixelReplay 和部署循环，通过显式工厂及视图构建接口接入新特征与学习器。

Tech Stack：PyTorch、SciPy cKDTree、现有 C++／CUDA tile 渲染扩展。

Spec：`docs/superpowers/specs/2026-10-29-physical-benefit-design.md`。

约束：保留现有入口，冻结编解码器；仅定向测试；保留用户未提交内容；完成后提交推送。

- [x] 任务 1：RED／GREEN：紧凑特征、准确积分标签、可见性缓存与 CPU／CUDA 统计。
- [x] 任务 2：RED／GREEN：共享评分网络、RD 排序监督、训练尺度和状态恢复。
- [x] 任务 3：RED／GREEN：独立入口、启动脚本、窗口与断点联调。
- [x] 任务 4：定向回归、独立审查和服务器命令；按默认流程提交推送。

审查重点：标签与特征隔离、q0 锚定、完整块变化、损失单位、验证泄漏、缓存生命周期、CUDA 同步和精确恢复。

验证结果：164 项定向测试通过，9 项 CUDA 测试跳过；启动脚本 bash 语法、Python CLI、依赖检查通过。独立审查提出的 q0 探针、遮挡前 alpha 和 SciPy 声明均已修复。GPU 数值一致性与实际耗时由服务器启动检查和训练日志验证。
