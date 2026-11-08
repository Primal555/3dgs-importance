# 独立正档位学习对照

入口：`scripts/test_benefit_isolated_positive.sh FOOTPRINT_RUN NEW_OUT [options]`。

本轮复用 `truck_footprint_split_gpu3_20261107_000336/data` 中的 4096 条训练标签与 512 条验证标签，以及
`training/metrics/final_predictions.jsonl` 中原结构、区域结构各 2000 次更新后的预测。只训练一个新增结构，不重跑旧两组。

## 训练内容

- 从同一份 `data/initial.pt` 开始，保留原始权重、输入／输出归一化和 RD 尺度。
- 原始共同价值网络冻结并始终处于 eval。正档位分支使用独立复制的点编码器、输出头和候选区域 CNN。
- 区域投影零初始化，检查初始有效预测与原网络一致。与 q0 完全相同的候选保持零输出，未屏蔽候选共同承担冻结的共同价值总量。
- 正档位监督包含每视角中心化两分量、q1/q2/q3 三组差值回归与窗口平均 RD 方向。两个分量保持原物理尺度，不分别重新归一化。
- 原 `pair_weight`、`decision_weight`、方向近似平局阈值、学习率 `1e-4`、dropout 和重要性权重沿用源配置。q0 绝对价值损失不更新新分支。
- 训练记录、洗牌种子与逐步 dropout 种子沿用上一轮；默认 2000 次更新、每 250 次评价。末步保存，不以验证集选模。

这组同时改变了任务隔离和监督范围，检验二者的组合效果。结果不能单独归因于某一种梯度冲突。
共同价值属于固定辅助输出；本入口不更新实际 q0 保留分支、概率表或编解码器。

## 检查结果

`metrics/history.jsonl` 包含独立留出集及固定训练探针的全部／普通／高差距样本报告。
分层阈值沿用先前仅由训练标签拟合的 Q90，训练探针是已参与训练的样本。
同时对两个旧参照给出改变、改善、恶化数量，最大收益记录、最大代价记录，以及排除最大一次／前三次收益后的 regret。
另列非零变化样本结果，避免零失真变化记录掩盖实际判断难度。

`summary.json` 汇总末步结果，`final.pt` 与 `latest.pt` 保存权重和恢复进度。
下载 `review.tar.gz` 即可检查，包内不含权重与回放。

## 真实反馈口径

可选传入 `--allocation-run PATH`，读取已有分配日志并生成 `feedback_audit.json`。默认训练只依赖第二组实验目录。
它分别统计历史残差复用、真实测量次数、已加入训练的动作标签。
旧日志没有当前精确替换的直接统计，该项显示 null，不能解释为零。

本轮同时为后续在线运行增加 `positive_current_exact_replacements` 和
`positive_current_exact_gap_max_abs_error`，核对真实测量后更新所用的动作差值。测量前审计记录注明阶段。
这些统计不修改原有更新逻辑，也不放宽历史残差匹配条件。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

SOURCE="$PWD/output/truck_footprint_split_gpu3_20261107_000336"
OUT="$PWD/output/truck_isolated_positive_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/test_benefit_isolated_positive.sh "$SOURCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

使用相同 SOURCE、OUT 与参数，增加 `RESUME=1` 续训。恢复点为最近一次评价，默认最多重做 249 次更新。
默认 CPU，无需指定 GPU；使用 `TRAIN_DEVICE=cuda CUDA_VISIBLE_DEVICES=卡号` 可改为 GPU 训练。
