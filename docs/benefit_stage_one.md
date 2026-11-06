# 第一阶段：离线正档损失对照

入口：`scripts/test_benefit_stage_one.sh WIDE_HISTORY_RUN NEW_OUT`。

源实验为 `neural_capacity_history` 的宽网络完整历史版本。可传实验根目录或 `state.pt`，根目录优先读取 `allocation/checkpoints/latest/state.pt`，其次读取预训练检查点。检查点读取为一次固定快照，源训练允许继续。训练回放分片保持只读，全部数据留在原目录。

分配检查点的 plan 只包含分配设置。评估器配置缺失时，从同一实验的 `benefit_network.pt` 恢复，兼容搬移后的 `pretrain/plan.json`、根目录 `plan.json`；不使用通用默认值替代原设置。启动日志与 `settings.json` 会记录配置来源，并检查网络宽度与导出初始网络身份。

本阶段只训练收益网络。默认每组 2000 次优化更新、批次沿用源设置，固定训练子集 4096 条；数据不足时使用全部可用记录。训练与独立固定验证源点必须分离。磁盘分片经有界 LRU 读取，默认 256 MiB，不将全部历史加载到内存。

## 两组的对齐

- 原组：`CorrectionLearner` 原损失。
- 修复组：同一网络、同一输入、同一初始权重及归一化。仅修改损失的任务拆分与正档尺度。
- 两组 AdamW 均重新初始化，学习率、权重衰减、方向项权重沿用源设置。
- 采用同一均匀抽样训练子集和同一批次序列，保留源点采集的重要性权重；完整教师与动作标签按同样顺序调用网络，使 dropout 随机数配对。
- 不渲染、不生成标签、不更新高斯属性或概率表，不启动 2000 步分配。当前在线入口保持原行为。

## 修复损失

对 q1/q2/q3 的三对差值独立监督。两个分量分别为 MSE/d_ref 和 lambda_s×结构误差/d_ref。

正档数值尺度为训练子集中逐视角、逐档对的分量 RMS；方向尺度为同一相机窗口平均后的 RD 差值 RMS，包含记录中的档间通信代价差。尺度仅由训练标签拟合一次，最小值 1e-8，不改变网络输入归一化或输出参数化。

修复目标为：

`正档差值分量回归＋源方向权重×连续KL方向损失＋0.25×相对q0共同价值回归`。

取消正档位方向项的 2% 硬忽略区；连续方向监督仍由真实差值大小加权。只有真实换挡差值的动作标签，不施加缺失的 q0 绝对监督。`--q0-aux-weight` 可改变辅助项权重，默认 0.25 为本次实验设置。

本阶段沿用回放记录的代价口径；尚未修改在线边际计费、纠偏迁移或候选足迹输入。

## 打印和结果

- 正档位：方向错误数/非平局比较数、误判代价、RD 差值 MAE 和 RMSE。
- 三档选优：网络选择相对真实最优的平均 RD 差距，并列出固定 q1/q2/q3 的同口径差距、最优档位命中率。
- q0 辅助：共同偏差/RMSE、最佳正档与删除的方向比较、辅助四档选择代价。这些预测不执行删点。
- 价值误差：逐视角与窗口平均 RMSE 分开，防止跨视角误差互相抵消后被忽略。
- 独立固定留出集与近期训练回放分别打印。近期训练回放是固定训练子集靠后的记录，已参与训练；独立固定集仍描述早期留出上下文。当前策略的新独立标签在后续阶段采集。
- 所有 RD 数值使用源 `d_ref` 的归一化单位，不能直接解释为 dB 或整幅场景部署 RD 的增减。
- `settings.json` 记录快照 SHA256、尺度、原忽略区覆盖数和配对规则；`metrics/history.jsonl` 保存所有评估窗口；`summary.json` 对比第 0 次与末次更新；最终网络保存到 `original/final.pt` 和 `repaired/final.pt`。
- 无最佳窗口挑选，保留末步网络。下载 `review.tar.gz` 即可检查指标，权重和回放不进入压缩包。

## 服务器运行

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

# SOURCE 填正在运行或已经完成的宽网络实验目录。
SOURCE="$PWD/output/truck_neural_capacity_history_gpu1_时间戳"
CHECK_OUT="$PWD/output/truck_benefit_stage_one_$(date +%Y%m%d_%H%M%S)"
DEVICE=cpu nohup bash scripts/test_benefit_stage_one.sh \
  "$SOURCE" "$CHECK_OUT" >"${CHECK_OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${CHECK_OUT}.pid"
tail -n 50 -F "${CHECK_OUT}.log"
```

空闲 GPU 上可设置 `CUDA_VISIBLE_DEVICES=1 DEVICE=cuda`。先测速可加 `--updates 100 --evaluate-every 100`；只读取并评价可用 `--updates 0`。这两种模式各使用新的 `CHECK_OUT`，不覆盖之前结果。已下载且移动回放目录时，可用 `--replay-root` 指向原分片结构的新位置。
