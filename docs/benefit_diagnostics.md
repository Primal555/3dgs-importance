# 收益评估网络的两项小检查

输入为已完成200窗口预训练的收益分配实验目录。脚本固定已有训练／验证回放，冻结概率表和编解码器，保持原损失、Adam状态、学习率和归一化。整个过程不调用渲染器、不采集新标签。

## 检查内容

1. 零点偏差：分别检查升档、降档和保留／删除的有符号误差、误选有害动作、漏掉有利动作及高代价错误。代价分层的两个阈值来自训练标签的绝对代价25%／75%分位数，验证时固定。
2. 固定数据续训：默认额外执行1000次收益网络Adam更新，每100次用同一批训练记录和验证记录评估。记录采样沿用原回放规则，最近窗口的参考时间固定为源检查点窗口。

零点检查额外使用训练记录拟合每个分支的平均偏差，再对验证预测作等量平移。这仅用于判断偏差的影响，不改变网络、概率表或部署规则。q0另外分解三个保留方案的共同误差和相对排序误差。

本检查不预设训练不足。可能看到的结果包括：

- 数值误差、训练和验证的决策regret一起改善：增加更新在当前标签分布上有效。
- 数值误差下降，训练与验证的决策regret仍高：应重点检查监督目标与决策目标的关系。
- 训练决策改善、验证决策退化：训练／验证泛化差距扩大。
- 训练集拟合的偏移修正明显改善验证判断：支持整体零点偏差的解释。

这些观察可同时出现。固定回放检查不衡量概率变化后的实际部署画质，也不能单独确定唯一原因。

## 服务器运行

`BENEFIT_RUN`指向刚完成预训练的输出目录。需要保留`checkpoints/latest/state.pt`及原PLY，照片和相机文件均不需要。迁移后可用`--ply /新路径/point_cloud.ply`定位同一PLY；程序校验场景指纹。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

# 如果仍在原预训练终端，可直接使用上一轮的 OUT。
BENEFIT_RUN="$OUT"
CHECK_OUT="$PWD/output/truck_benefit_checks_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/test_benefit_diagnostics.sh \
  "$BENEFIT_RUN" "$CHECK_OUT" --updates 1000 \
  >"${CHECK_OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${CHECK_OUT}.pid"
printf '结果：%s\n日志：%s\n' "$CHECK_OUT" "${CHECK_OUT}.log"
tail -n 40 -F "${CHECK_OUT}.log"
```

默认CPU、4线程，只训练小型收益网络。需要GPU时增加`--device cuda`并设置`CUDA_VISIBLE_DEVICES`。只做初始偏差检查时，使用`--updates 0`及独立的新输出目录。

## 查看结果

- `initial_bias.json`：源网络的有符号偏差、动作分层、q0共同偏差，以及训练集拟合偏移后的验证表现。
- `history.jsonl`：固定样本上的训练／验证完整指标，分别记录一阶预测、收益网络、屏蔽一阶输入的收益网络；`additional_updates=0`为起点。
- `final_bias.json`：最后一次评估的同口径偏差检查。
- `summary.json`：固定数据指纹、样本数、原检查点未改变的校验，以及续训前后结果。
- `offline_final.pt`：仅供诊断的收益网络权重，不接入分配入口。

`signed_bias>0`表示把动作代价估高；`false_beneficial`表示把真实有害动作判为有利；`false_harmful`表示漏掉真实有利动作。`common_mode_error_energy_fraction`高表示q0三个保留方案的误差主要来自共同偏移。

`full_input_loss`为原监督损失在完整输入下的评估值，不含随机屏蔽的附加项。训练更新仍保留原屏蔽率与权重。训练和验证的固定记录数不随离线更新增加。
