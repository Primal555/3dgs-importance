# 数值恢复、RD 概率排序与 GradNorm

入口：`scripts/test_benefit_rank_gradnorm.sh FOOTPRINT_RUN NEW_OUT [options]`。

复用 `truck_footprint_split_gpu3_20261107_000336/data` 的候选区域标签和初始化，以及已完成的原结构、区域分头结构末步预测。
独立正档位网络与冻结共同价值沿用上一版。默认 CPU 训练 2000 次，不重新渲染、不训练编解码器、不更新概率表。

## 两种不同的监督

令三个档位的两分量误差为 e，原来的中心化项与全部三对差值项满足 `L_pair=3*L_centered`。
本轮合并成一项：`L_magnitude=(1+3*pair_weight)*mean(((e-mean_tier(e))/s)^2)`。
源配置 pair_weight=1 时系数为 4，数值部分保留旧有强度与统一 RD 尺度 s。

第二项直接学习三个档位的完整 RD 代价顺序：

`true_cost=sum(teacher_components)+known_rate_cost`

`predicted_cost=sum(predicted_components)+known_rate_cost`

对每对档位，真实和预测概率为 `sigmoid(-cost_difference/tau)`。
RankNet 式概率交叉熵减去固定教师熵，得到在真实预测处为零的 Bernoulli KL；数值回归与该排序目标具有不同含义。
接近相同的档位使用接近 0.5 的软目标。排序项无硬平局阈值、无差值绝对值的外层权重。
排序温度 tau 沿用上一轮从同一训练集非零 RD 差值中位数拟合的值，最低为 1e-8；不从验证标签拟合。
不乘旧版 `(tau/s)^2` 因子，实际更新强度由 GradNorm 调整。

## 普通／高差距任务

分层继续使用同一训练集 RD 跨度 Q90，普通与高差距样本各有 magnitude、ranking 两项任务。
只有一个非空分层时只启用对应两项。采用各层独立洗牌循环，每批按接近原数量比例分配，保证非空层至少一条。
实际配额写入 settings。典型 8 条批次为 7 条普通、1 条高差距；以实际标签分布计算为准。

源重要性权重保留，除以完整训练集内该层的平均权重。使用固定平均值能避免单条高差距批次抵消自身重要性。
分层条件目标及 GradNorm 会改变训练代理；最终评价继续使用未改动的物理 RD、未加权 regret 和误差。

## GradNorm 的实际更新

训练前 eval 遍历完整训练集，按同一权重口径计算各任务初始损失 L(0)，不更新参数。
每步计算各任务对 `relative.state_head[2]` 隐藏层权重和偏置的实际梯度范数。
以平均加权梯度和相对训练速度的 alpha 次方为目标，优化任务权重；网络优化使用 detach 的任务权重。
梯度范数计算使用一阶 autograd，不计算 Hessian，完整网络每步仅反向更新一次。

采用原论文 NYUv2 实验的 alpha=1.5、任务权重 Adam 学习率 0.025。以下是本项目适配：

- 初始损失基准来自完整训练集，避免前几个批次决定尺度。
- 相对训练速度使用 0.95 EMA，降低单条重尾样本造成的批次波动。
- 任务权重采用 float64 softmax 正参数化，归一化总和为任务数，不设固定下限，允许补偿回归与排序的较大尺度差异。
- 网络仍使用源 AdamW 学习率 1e-4、原 dropout 与整体梯度上限 10。

RankNet 的温度以及 GradNorm 的分层任务、EMA 和正参数化属于本实验设计，不代表两篇论文已验证的 3DGS 方案。

## 结果与恢复

每 250 次输出训练探针与验证集的全体、普通、高差距指标；记录活跃样本命中率、相对旧参照的改善／恶化数量、剔除最大收益记录后的表现。
`metrics/optimization.jsonl` 每 10 次记录任务损失、实际使用权重、原始／加权／目标参数梯度及相对训练速度。
`metrics/history.jsonl` 的 task_weights 为评价时、权重优化器更新后的数值；optimization 的 weights 为本步网络更新所用数值。
末步保存，不以验证集选模。检查点包含网络、网络优化器、GradNorm 权重与优化器、损失 EMA、分层采样与访问计数。

下载 `OUT/review.tar.gz`。权重保存在 `latest.pt`、`final.pt`，不进入下载包。
恢复使用相同 SOURCE、OUT 与参数并加 `RESUME=1`，初始损失基准直接读取本实验保存值。

## 服务器命令

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
SOURCE="$PWD/output/truck_footprint_split_gpu3_20261107_000336"
OUT="$PWD/output/truck_rank_gradnorm_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/test_benefit_rank_gradnorm.sh "$SOURCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

## 参考

- RankNet，ICML 2005，概率排序与软目标：https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf
- GradNorm，ICML 2018，梯度目标、相对训练速度及 NYUv2 设置：https://proceedings.mlr.press/v80/chen18a/chen18a.pdf
