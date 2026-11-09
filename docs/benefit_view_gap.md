# 逐视角档间差距与排序梯度尺度实验

本实验沿用覆盖区域输入、分离预测头和固定标签集。默认仅运行新目标的 batch64 组，CPU、微批次 8、16000 次样本访问，共 250 次 AdamW 更新。与上一轮 batch64 组使用相同初始化、采样顺序、学习率及评测视角。

## 两项修改

### 排序概率与梯度增益分离

排序目标仍是 `sigmoid(-真实档间RD差/tau)`，tau 由训练标签拟合。RankNet KL 乘以 `tau/s`，s 为原数值回归尺度。单对损失相对于 `预测RD差/s` 的梯度绝对值不超过 1。概率温度保持原值，数值尺度和标签不变。

每个有效批次先积累全部参数的三任务梯度，得到 GradNorm 权重后，计算数值合成梯度 `g_num=w_common*g_common+w_relative*g_relative` 和排序梯度 `g_rank=w_rank*g_ranking`。排序实际应用系数为 `min(1, ||g_num||/||g_rank||)`。无排序梯度时系数为 1；数值梯度为零而排序梯度非零时系数为 0。之后沿用整体范数裁剪 10 和 AdamW。

该系数用于同一次参数更新，保证排序梯度范数不超过数值合成梯度。它约束更新前的梯度；Adam 的参数位移还受历史状态影响。GradNorm 的申请权重、实际权重、缩放前后梯度范数分别记录。

### 逐视角总 RD 差获得明确权重

令每个视角的两分量预测误差除以 s，并分别减去三个正档位的均值，得到 a、b。a 对应 MSE，b 对应加权结构项。旧档间回归等于：

`c/4 * mean((a+b)^2 + (a-b)^2)`，其中 `c=1+2*pair_weight`。

新档间回归为：

`c/4 * mean(2*(a+b)^2 + (a-b)^2)`。

总 RD 误差方向的权重提高到 2，分量差方向权重保留为 1。两方向是原误差空间的正交分解；使用已有真实标签，提高对最终 RD 档间差的重视程度。数值损失逐视角计算后再平均，公共值回归保持原定义。排序继续对窗口平均 RD 监督。

## 初始化与结果

来源必须是含 `data/initial.pt`、回放文件和 `training` 结果的完整 footprint_split 目录。`data/initial.pt` 是原评估网络快照，已有点特征网络和头部参数被加载；覆盖区域新分支按原随机种子构造。每次实验创建新优化器，使用相同起点重新运行，不加载此前最佳覆盖区域网络权重。

固定验证集同时记录：

- 普通／高代价样本的 regret、命中率与最大错选代价。
- 原 RD 单位的逐视角档间 RMSE、窗口平均档间 RMSE。
- 完整排序正确率，以及选中最优档位后的档间 RMSE。比较使用 `1e-12` 的真实差距平局容差。
- 每一步累计的排序梯度限幅比例、平均系数、整体裁剪频率；抽样记录分层共享参数梯度。

样本分层阈值与归一化仅从原训练数据取得。最终保存末步网络，`review.tar.gz` 仅打包设置、指标和逐条预测。检查点保留在实验目录，恢复时使用同一 OUT 与 `RESUME=1`。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
SOURCE="$PWD/output/truck_footprint_split_gpu3_20261107_000336"
OUT="$PWD/output/truck_benefit_view_gap_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/test_benefit_view_gap.sh "$SOURCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

关注普通样本和高代价样本能否同时降低档间误差与错选代价，并结合整个后段的波动判断；完整训练效果由服务器实验验证。
