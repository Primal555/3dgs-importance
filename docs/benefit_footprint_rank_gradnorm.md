# 共享覆盖区域双头：RankNet＋GradNorm

入口：`scripts/test_benefit_footprint_rank_gradnorm.sh FOOTPRINT_RUN NEW_OUT [options]`。

本实验在 `FootprintSplitNetwork` 上训练。共享点编码器、覆盖区域 CNN、共同价值预测头、档间差异预测头均参与更新。结构与 `truck_footprint_split_gpu3_20261107_000336` 的 upgraded 实验相同。

## 初始化与比较

读取原标签采集目录的 `data/initial.pt`，使用原双臂实验的构造顺序，确保覆盖区域 CNN 的初始参数也相同。训练完成的 upgraded 预测仅作为结果比较参照，其检查点不参与初始化。

沿用原 4096 条训练记录、512 条验证记录及其源点隔离、固定归一化和通信成本。原标签、初始文件、参照结果均只读。入口校验记录身份、教师值、通信成本与参照一致。

原始初始文件可能含有早前训练的空间价值网络；“正常初始化”在这里指与原双头实验使用同一起点，新增 CNN 按同一随机序列构造。

## 两类监督

原数值监督完整保留。设归一化误差为 e，三档平均误差为 m，权重为 cw、pw：

`L_numeric = absolute + cw*common + pw*six_pair`

代数合并后为：

`L_numeric = (1+2*pw)*mean((e-m)^2) + (1+cw+pw/2)*mean(m^2)`。

其中 six_pair 包括 q0 锚点与 q1–q3 的全部六对比较；测试检查合并前后的数值和参数梯度等价。默认 cw=pw=1 时，中心化误差系数为 3，共同误差系数为 2.5。

排序监督使用完整 RD 成本，包括已知通信成本。四档成本为 `[0, D1+R1, D2+R2, D3+R3]`，六对成本差经过 `sigmoid(-gap/tau)` 得到真实软概率与预测概率。RankNet 交叉熵减去教师熵，得到在真实预测处为零的 Bernoulli KL。接近相同的成本使用接近 0.5 的软标签。温度沿用同一训练集拟合的值。

数值恢复按逐视角监督，排序按同一相机窗口的平均成本监督。GradNorm 平衡 numeric、ranking 两项任务；不再另乘旧 decision_weight。监测参数为共享点编码器 `points[2]` 的权重和偏置，两项监督均可通过两个预测头到达共享编码器。

GradNorm 沿用 alpha=1.5、权重 Adam 学习率 0.025、训练进度 EMA=0.95、正 softmax 权重且权重和为 2。初始损失基准遍历完整训练集计算。网络学习率、AdamW、dropout、整体梯度上限 10 沿用原配置。

## 训练与指标

默认 CPU 2000 次更新，每 250 次评价。每批沿用原版均匀洗牌循环及 batch_groups，importance 权重仍用于加权批次均值。总标签访问预算保持为 `2000*batch_groups`。

评价同时报告全体、普通、大差距样本的错选代价、档间 RMSE、命中率、非零变化样本命中率，以及相对 original／upgraded 的改善和恶化数量、移除单项最大收益后的结果。训练任务调权不改变评价的物理 RD 成本。

`metrics/optimization.jsonl` 每 10 次记录两项损失、任务权重、加权梯度范数、目标范数及整体裁剪前梯度。`summary.json` 记录各分支参数是否改变及归一化是否保持不变。末步权重用于最终结果。

只需下载 `OUT/review.tar.gz`。恢复训练需要保留原 SOURCE 的 data/replay、initial.pt、参照记录以及本实验的 latest.pt。恢复使用相同参数、SOURCE、OUT 并设置 `RESUME=1`。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

SOURCE="$PWD/output/truck_footprint_split_gpu3_20261107_000336"
OUT="$PWD/output/truck_footprint_rank_gradnorm_$(date +%Y%m%d_%H%M%S)"

nohup bash scripts/test_benefit_footprint_rank_gradnorm.sh \
  "$SOURCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

## 方法来源

- RankNet：[Learning to Rank Using Gradient Descent](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)。
- GradNorm：[Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://proceedings.mlr.press/v80/chen18a.html)。

本轮复用已实现的损失与调权方法，适配到完整共享双头结构。其在本任务上的效果由本轮固定标签实验验证。
