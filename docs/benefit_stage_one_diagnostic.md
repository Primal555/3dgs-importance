# 第一阶段修复组的逐样本检查

读取已完成离线对照的 `original/final.pt`、`repaired/final.pt` 和
`data_manifest.json` 中的固定验证记录。默认 CPU，只重新前向，不更新权重或调用渲染器。
无需读取原在线分配器的当前状态。回放文件迁移后可用 `--replay-root` 指定原有标签的新位置。

```bash
SOURCE="$PWD/output/truck_benefit_stage_one_20261106_180906"
CHECK_OUT="$PWD/output/truck_stage_one_diagnostic_$(date +%Y%m%d_%H%M%S)"
DEVICE=cpu nohup bash scripts/test_benefit_stage_one_diagnostic.sh \
  "$SOURCE" "$CHECK_OUT" >"${CHECK_OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${CHECK_OUT}.pid"
tail -n 40 -F "${CHECK_OUT}.log"
```

每条记录先平均各视角的两项失真，再加入已包含 beta 的通信成本，分别选择 q1–q3 中
预测 RD 最低的档位。选档 regret 为选中档位的真实 RD 减去三个正档位的最小真实 RD。
真实最优存在并列时，选择其中任意一个均计为正确。所有统计使用记录均匀权重，与原离线报告一致。

`summary.json` 包含：

- 原报告的逐字段重算核对，覆盖 regret、误判数、档位数量和分量误差；不一致时保存报告并报错。
- 两组逐样本配对的新增错选、纠正错选、有害改选和有益改选数量。
- 有害改选的代价总和、有益改选的收益总和，以及二者相抵后的平均变化。
- 最严重 10 次有害改选占新增损失的比例；默认导出最严重 20 例和改善最大 20 例。
- 实际改选 RD 变化的 MSE、加权结构项和通信成本分解。

`records.jsonl` 保存全部验证记录的源点索引、采集窗口、两组选择、真实和预测的逐视角分量、
共同误差及档间误差、改选收益与预测档间差值。q0 统计标注为辅助输出，与实际运行的密集删点分支区分。
两组 checkpoint 的 SHA256 和验证清单摘要随报告保存。

下载 `CHECK_OUT/review.tar.gz` 即可检查全部输出，无需下载回放张量或网络权重。
这些数据用于确定新增损失落在哪些样本和分量；各项训练损失调整的独立因果效果仍需单变量对照。
