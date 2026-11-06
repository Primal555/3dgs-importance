# 统一 RD 与三档联合选择离线实验

本轮同时评价普通样本的数值精度和高代价错选。训练记录沿用已完成第一阶段离线实验中的固定清单，
从同一个 `initial.pt` 开始，更新次数、批次大小和种子沿用原对照。每条训练记录通过均匀洗牌循环获得覆盖，
并保留原采集重要性权重。仅训练一个新组，已有原损失组的末步网络只用于复核和评价。

## 损失定义

对记录中的各视角先平均，再将两项失真与已包含 beta 的档间通信成本相加，得到三个正档位的 RD 代价。
主回归监督去除三档均值后的 RD；辅助回归监督每个视角的两项原始数值，两项共用原 `scales.rd` 尺度 s。
因此，视角误差互相抵消时仍有数值梯度，普通小误差也持续参与训练。

联合选择教师为 `softmax(-true_RD/tau)`，预测为 `softmax(-predicted_RD/tau)`。
温度 tau 只由固定训练记录的非零正档 RD 差值中位数确定，下限为 1e-8。
联合 KL 乘 `(tau/s)^2`，将局部曲率还原到 RD/s 的共同单位，避免小温度压倒数值回归。

`L = centered_RD_MSE + 0.25 * per_view_component_MSE + decision_weight * (tau/s)^2 * KL`

分量辅助权重 0.25 为本轮工程设置；方向权重沿用源配置，当前为 0.25。
真实完整预测为所有监督项共同的最优点。相对 q0 的共同偏差由辅助数值项约束。
只有 action_gap 标签的记录使用两个已观察动作的 RD 差回归和两状态软选择，不生成其他状态的标签。

## 指标与对照

完整训练记录真实正档 RD 极差的 Q90 用作固定分层阈值。验证记录按同一阈值分为普通和高差距子集；
高差距子集描述较大的档间差异，与模型是否预测正确无关。每次评价同时报告：

- 全部、普通、高差距记录的数量、三档 regret、命中率、档间 MAE/RMSE。
- 逐视角 RD RMSE、窗口平均两分量 RMSE，持续检查普通样本的拟合。
- 最大选档 regret、最严重 10 例占比，检查高代价错选。
- q0 辅助统计和已参与训练的近期回放指标，分别保存在报告中。

默认使用原对照相同的 2000 次更新，末步评价，不挑选最佳窗口。
`reference.json` 保存重算过的已有原组末步分层指标；`summary.json` 与它使用完全相同的验证记录和阈值。
启动时逐字段复核原初始及原末步验证报告，发现不一致即停止。
原始第一阶段运行缺失训练配置时，读取同源 `benefit_network.pt` 的设置；不会加载它的网络权重。
历史文件未保存实际批大小时，必须显式填写当时的 `--batch-groups`；本次对照为 8。
之后的第一阶段实验会保存实际批大小与完整配置，新入口直接使用这些不可变元数据。

当前是固定回放离线实验，不加载 PLY 或相机，不重新渲染标签，不更新概率表。
训练与验证源点检查隔离。输入、输出归一化均保持原值。现有在线分配入口保持原行为。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
SOURCE="$PWD/output/truck_benefit_stage_one_20261106_180906"
OUT="$PWD/output/truck_benefit_rd_choice_$(date +%Y%m%d_%H%M%S)"
DEVICE=cpu nohup bash scripts/test_benefit_rd_choice.sh \
  "$SOURCE" "$OUT" --batch-groups 8 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

数据迁移后用 `--replay-root` 指向原分片目录，`--pipeline-root` 指向保存同源 `benefit_network.pt` 的原实验根目录。
下载新 `OUT/review.tar.gz` 即可，包含设置、对照、指标历史和末步逐样本预测。回放和权重不进入下载包。
