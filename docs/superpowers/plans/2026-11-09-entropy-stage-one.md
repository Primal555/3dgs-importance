# 条件熵选档第一阶段实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. 按任务验证并记录结果。

**Goal:** 实现固定48维编解码器、固定保留集合的条件熵训练和真实RD选档独立入口。

**Architecture:** 训练集统计与分组条件logistic概率模型产生逐点信息分数；离散映射在校准视角通过现有编解码器、渲染及实际计费选择。已有分配入口保持原样。

**Tech Stack:** PyTorch、现有3DGS评估路径、JSONL、Bash。

**Spec:** `docs/entropy_stage_one.md`。

## Global Constraints

- 在当前仓库新增本轮文件，沿用用户已指定的工作方式，保留其他未提交内容。
- 模型／场景／块顺序／输入集合／图像目标／成本单位固定。
- 仅熵模型有训练梯度，真实硬档位路径作为最终评估依据。
- 完成定向验证后提交并推送；输出提供小型review包。

## Task 1：概率模型与数据

- [x] 写并运行区间概率、梯度、因果性、统计隔离、块划分与行号测试。
- [x] 实现 `entropy_model.py` 和 `entropy_training.py`，验证完整覆盖和恢复。

## Task 2：校准与实验入口

- [x] 写并运行档位映射、真实成本和小场景入口测试。
- [x] 实现 `entropy_stage_one.py`，复用既有源加载、场景、评估和成本函数。
- [x] 增加启动脚本和精简检查包；验证校准／测试分离和冻结权重。

## Task 3：交付

- [x] 定向测试、代码审查、命令行与脚本语法检查。
- [x] 更新验证记录并提供服务器启动命令；按项目规则完成提交与推送。

## 实施记录

- `python -B -X utf8 -m unittest discover -s tests -p 'test_entropy_*.py' -v`：11项通过。测试覆盖真实successive编解码路径；CPU入口测试的栅格器使用小型替身。
- 概率模型、训练／恢复、校准入口先运行失败测试，再实现对应功能。
- 独立审查发现末批不完整块不能直接二维拼接、首轮中断缺少初始checkpoint两项问题；分别增加失败用例并修复，最终11项全部通过。
- 启动脚本 `bash -n`、Python入口 `--help` 通过。
- 统计分数与实际通信成本分开记录；真实CUDA栅格器及GPU耗时由服务器实验检验。
