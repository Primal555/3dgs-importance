# Pixel Benefit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** 交付缓存局部真实干预、像素收益网络和可恢复的训练/分配独立实验。

**Architecture:** 窗口级场景缓存与受影响块精确重渲染生成密集标签。共享场景上下文、逐候选预测失真图，完整组积分监督，固定验证与真实末步保存。

**Tech Stack:** PyTorch、独立 C++/CUDA 扩展、现有 MaskGaussian 前处理及合成定义。

**Spec:** docs/superpowers/specs/2026-10-29-pixel-benefit-design.md

## Global Constraints

- 保持旧入口、冻结模型、10 dB 和0/8/24/48可用；不修改用户其他文件。
- 原像素 tile，所有遮挡 token，不用局部信息压缩替代真实合成。
- 保留原噪声槽位；q0 改变存在集合重新编码；teacher图像不能进入特征。
- 默认64真实组；缓存复用、精确续训、固定验证；latest/final 不回退。
- 本地CPU验证与服务器CUDA验证分开报告；沿用定向测试范围。

## Review Focus

- 遮挡解除后的背景必须在候选合成中出现，不能使用基线 n_contrib 截断。
- 候选形状扩大须进入新的 tile；同深度顺序与原行号一致。
- SSIM差的输出区域和读取halo必须完整，图像边缘不能按crop边界重新填充。
- 筛选候选块不改变噪声槽位；共享缓存不能在改mask或更新网络后过期使用。
- 训练采样、验证冻结、RNG、概率表及模型/优化器需要可恢复且相互对应。

### Task 1: 独立局部渲染器

**Files:** gaussian_jscc/pixel_local_renderer.py；gaussian_jscc/csrc/pixel_local/*；tests/test_pixel_local_renderer.py。
**Interfaces:** 使用 spec 的 TileScene/capture/project/affected_tiles/render_tiles。
- [x] 先写删除显露背景、候选扩展与跨tile的CPU参照测试并观察失败。
- [x] 实现参照和独立CUDA backend，复用原前处理，按tile稳定合并候选。
- [x] CPU数学测试通过；提供CUDA对原renderer的实际一致性用例。

### Task 2: 密集收益网络与监督

**Files:** gaussian_jscc/pixel_benefit.py；tests/test_pixel_benefit.py。
**Interfaces:** spec定义输入、PixelBenefitNetwork、pixel_density_labels；补充 integrate_density(density,valid,height,width,d_ref)，完整组损失。
- [x] 写像素符号、SSIM边界、q0锚点、无token截断与共同误差梯度测试并观察失败。
- [x] 实现深度顺序编码、候选读取、像素预测和固定训练尺度。
- [x] 验证dense积分等于whole-image差、tail梯度和小样本可拟合。

### Task 3: 采样与磁盘回放

**Files:** gaussian_jscc/pixel_benefit_replay.py；tests/test_pixel_benefit_replay.py。
**Interfaces:** PixelReplay(root,seed,max_groups,validation_groups)；add(record,validation)；freeze_validation()；sample(count)->records；validation_records()；state_dict/load_state_dict；sample_sources(candidates,impact,difficulty,count,generator)->source IDs及逐次抽样概率。
- [x] 写固定留出、抽样校正、磁盘分片和精确恢复测试并观察失败。
- [x] 实现按完整组存储与抽样，元数据索引、固定验证、训练困难度更新；uniform下限保证非零概率。
- [x] 验证来源隔离、重复抽样无重复渲染、缺失分片明确报错。

### Task 4: 冻结候选解码与窗口

**Files:** gaussian_jscc/pixel_benefit_window.py；gaussian_jscc/pixel_benefit_features.py；gaussian_jscc/successive_codec.py；tests/test_pixel_benefit_window.py。
- [x] 写筛选batch仍配对、邻居变化、halo拼接、无标签特征泄露测试。
- [x] 实现选择优先的候选解码；显式保留完整噪声；原始场景、投影、图像、局部索引缓存。
- [x] 连接Task1-3，训练和推断分支独立，逐视角记录与完整组监督。

### Task 5: 实验入口与验证交付

**Files:** gaussian_jscc/pixel_benefit_allocation.py；scripts/test_pixel_benefit_allocation.sh；tests/test_pixel_benefit_integration.py。
- [x] 写dry-run、resume身份、fixed validation、末步保存与阶段切换测试。
- [x] 复用已有场景加载、精确通信核算、部署和最终评估，接通200窗口预训练与显式2000步分配。
- [x] 写服务器编译/一致性验证/训练命令，提供分阶段耗时。
- [x] 定向回归与独立审查，修复后交付，记录GPU尚待实测的项目。

## 本轮验证结果

- 定向测试：137 passed、7 skipped，另有1项 subtest 通过；跳过项均需 CUDA。
- Bash 启动脚本语法检查通过；独立代码审查未发现阻止交付的问题。
- 本地环境为 PyTorch 2.9.0 CPU。服务器启动入口会先执行 CUDA 局部与全图渲染一致性检查；真实 Truck 单步耗时、显存和回放磁盘占用仍需实测。
