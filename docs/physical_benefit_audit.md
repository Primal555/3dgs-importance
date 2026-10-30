# 物理特征收益评估器收口审计

入口：`scripts/test_physical_benefit_audit.sh PRETRAIN_RUN NEW_AUDIT_OUT`。默认 CPU，现有 checkpoint、replay、源码即可运行。无需 PLY、原始图像或 GPU。

本轮一次回答以下问题：

1. 按源采样概率校正后，训练／验证误差与固定档位基线有何差异？遍历 checkpoint 索引中的全部训练记录和全部验证记录。
2. 原始特征与标签相对于固定尺度是否存在极端值？高代价分层中的错误是否集中？输出代价分层、分量误差和每视角真实成本。
3. 共同误差、档间误差及 MSE／结构分量分别影响多少决策？用真实标签构造 oracle 校正，仅用于定位，不能用于部署。
4. 固定训练池是否可拟合？从训练池选择 64 个随机记录和 64 个高代价记录，保持原归一化、损失、AdamW 状态、学习率和 dropout，进行 1000 次私有网络更新。每次 4 组，轮换确保每组实测接受至少 31 次更新。验证集不参与选池和训练。
5. 输入是否存在不可辨识性？合成三个不重叠高斯，交换两个邻居的颜色修正位置。两组 158 维特征完全相同，但真实渲染收益方向相反。整图 MSE 独立检查局部教师结果。
6. 配对噪声、候选批处理及边际符号代价是否一致？使用 checkpoint 的真实 codec 权重，在确定性合成块上比较原始前向、显式噪声及完整块重算；用实际 binary16 scale 字节差检查成本。

输出：`summary.json`、`before.json`、`fixed_fit_history.json`、`fixed_fit_visits.json`、`mechanism_checks.json`、`settings.json`。源 checkpoint 不变，审计网络不写回训练目录。

原实验未记录精确的历史逐组更新次数，审计不能事后还原。`historical_group_draws` 仅给出总抽样次数；固定池的新访问次数逐组记录。

原始优先级从 1 开始，发生变化的记录可确认至少更新过一次；优先级仍为 1 的记录不据此判定为从未训练。固定池同时累计梯度裁剪比例与回归／排序损失，检查固定尺度是否使训练持续受裁剪限制。

固定池包含刻意选出的高代价样本，需在同一固定池的前后进行比较。训练数据的重要性加权修正来源采样；完整扫描 replay 后无需额外加 replay 抽样权重。代价分层阈值只从训练标签拟合。

```bash
RUN="$PWD/output/truck_physical_benefit_gpu2_20261029_223815"
AUDIT_OUT="${RUN}_audit_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/test_physical_benefit_audit.sh "$RUN" "$AUDIT_OUT" >"${AUDIT_OUT}.log" 2>&1 &
tail -n 30 -F "${AUDIT_OUT}.log"
```

审计以当前设计疑点为边界。得到真实数据结果后，依据输入不可辨识性、固定池拟合表现和加权决策误差确定下一轮修改；只有输入文件或一致性检查失败时才补齐对应证据。
