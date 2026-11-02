# 连续掩码 RD 分配恢复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 恢复完整 A 路线，在统一 RD 目标下持续学习并依据实测部署结果收尾。

**Architecture:** 新增独立入口，复用历史掩码梯度与当前冻结 successive 编解码器。独立控制器管理降学习率及收尾，独立训练适配器统一图像与资源单位。

**Tech Stack:** PyTorch、现有 Gaussian rasterizer、Bash、unittest。

**Spec:** `docs/superpowers/specs/2026-11-02-dense-rd-allocation-design.md`

## Global Constraints

- 保留收益网络实验；不改变源编解码器与场景属性。
- 单次采样学习、十次规则部署；每轮全点学习路径。
- 只执行本轮定向验证，默认提交推送并给服务器命令。

## Review Focus

- packet 顺序与原 PLY 顺序成本对应。
- 图像梯度、训练资源期望、部署报告采用同一 D_ref/C_ref。
- q0 反事实反馈、全部参数冻结以及当前模型兼容性。
- 调度器恢复、校准和测试隔离及输出体积。
- 中断重启日志、RNG、视角与 Adam 连续性。

## Tasks

- [x] 1．先写控制器、成本顺序及归一化梯度测试；确认缺失实现时失败。
- [x] 2．实现 `dense_rd_control.py`、`dense_rd_training.py` 并验证。
- [x] 3．实现 `dense_rd_allocation.py`、启动脚本、恢复与紧凑报告；增加端到端小场景定向测试。
- [x] 4．更新使用文档，运行必要相关测试、Bash 语法检查和差异检查，独立审查。
- [x] 5．仅提交本轮文件并推送，给出服务器拉取、启动及查看进度命令。

## 验证记录

定向新测试最初因缺少实现失败，随后通过。与既有掩码梯度、解码复用、successive 梯度测试合计 27 项通过；审查补充打包中断恢复测试后，新入口 9 项再次通过。Git Bash 语法检查通过。本地真实 `successive_transfer_v1` 主干的 4,882,520 个参数加载通过，档位为 0／8／24／48。独立审查提出的压缩包中断恢复问题已改为原子替换并复现验证。实际 CUDA 渲染效果及速度由服务器实验确认。
