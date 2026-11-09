# 独立 A／B 预测与分组梯度预算

本实验直接读取 `truck_response_terms_gpu3_20261109_181126/data` 的真实 A、B、S 标签，使用 CPU 重新初始化收益网络并训练。默认4096组训练标签、512组留出标签，64000次样本访问，即1000次更新。

## 预测与监督

共享主干和覆盖区域输入沿用上一轮。两个有符号输出通道预测 A、S，每个通道保留共同值与档间差异。独立 Softplus 头预测非负 B。最终失真为 `[A+B, S]`，通信成本单独加入。

主干使用同一原始初始化文件；三个预测头按固定随机种子重新初始化。B 的初始输出取训练集正能量均值，按训练 RMS 缩放。候选与 q0 输入完全相同时，三项均强制为零。

价值损失系数为1；A、B、S辅助监督经固定训练尺度归一化，GradNorm只平衡三项辅助任务，总权重为1。排序表达及温度沿用当前版本。

## 分组数值梯度

保留原采样修正与重要性权重，分别累积普通组与高代价组的全部数值任务梯度，得到 `g_o`、`g_h`。普通梯度范数参考采用EMA，系数0.95；首个非零普通梯度初始化参考。

`alpha=min(1, ordinary_ema / ||g_h||)`，合成 `g_num=g_o+alpha*g_h`。无有效普通参考时，`alpha=1`并记录原因。有限性检查在改变EMA之前完成。预算计算覆盖全部可训练参数，排序梯度随后限制为不超过实际合成数值梯度范数。最后沿用全局裁剪阈值10。

这会改变平均平方误差梯度的样本权重。每组标签仍参与监督；报告同时展示组梯度、实际缩放、裁剪、分层regret和档间RMSE，用于判断高代价样本是否获得有效更新。

## 运行与留存

新入口只进行固定标签CPU训练，不采集新渲染、不更新概率表。来源目录保持只读。预算EMA随 `training/latest.pt` 保存，支持精确断点恢复。下载 `review.tar.gz` 检查结果；继续训练需要保留来源 `data/replay` 与新实验的 `training/latest.pt`。

```bash
SOURCE="$PWD/output/truck_response_terms_gpu3_20261109_181126"
OUT="$PWD/output/truck_independent_response_cpu_$(date +%Y%m%d_%H%M%S)"
RESUME=0 nohup bash scripts/test_benefit_independent_response.sh \
  "$SOURCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

恢复时使用同一 `SOURCE`、`OUT` 和参数，将 `RESUME=0` 改为 `RESUME=1`。当前默认预算为1000次更新；本轮从初始状态训练，不读取上一轮末步权重。
