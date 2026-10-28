# 分配器实际更新代价审计

入口：`gaussian_jscc.allocation_update_audit`。加载已完成的 `calibrated_structured_allocation_v2` 实验，默认读取 `checkpoints/best_rd/state.pt`。复用检查点中的编解码器、逐点概率、收益历史、β、失真归一化系数和 SSIM 权重。

## 检查内容

- 正档位：比较当前一阶预测、加入旧纠偏的当前预测、最终历史更新代价与精确换挡结果。
- 保留或删除：精确测量 q0/q1/q2/q3。历史就绪时，按照正档位条件概率计算三档混合标签；历史未就绪时，匹配训练中的采样档位回退。同时单独检查当前采样档位的一阶及纠偏结果。
- 概率变化：在单点概率表副本上执行真实更新，记录升降档或保留删除的对数优势变化，并核验更新方向。

画质使用源 PLY 渲染作为参照，沿用训练中的 MSE 和 SSIM 混合失真。通信项使用训练中的边际成本代理。每次预测先固定，随后生成精确标签；本轮精确标签不进入被评估的历史。

默认每组20个训练视角窗口，每窗口使用源实验的视角数量，最多32次反事实渲染。默认预算分为26个正档位动作和2个保留删除点，各检查3个替代状态。候选按照决策边界、历史不确定性与覆盖情况选择，因此误判率用于这些重点候选的诊断。

模型、概率和预测历史保持固定。每窗口暂时加入一份近似观察，以复用真实历史更新路径，完成审计后原位恢复；覆盖调度计数允许推进。源实验文件只读。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs

GPU=2
RUN1="$PWD/output/truck_calibrated_evidence_gpu1_20261028_010635"
RUN2="$PWD/output/truck_calibrated_evidence_gpu2_20261028_010738"
OUT="$PWD/output/truck_update_audit_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" WINDOWS=20 AUDIT_SAMPLES=32 \
  nohup bash scripts/test_allocation_update_audit.sh "$OUT" "$RUN1" "$RUN2" \
  >"${OUT}.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"${OUT}.pid"
printf 'PID：%s\n输出：%s\n日志：%s\n' "$PID" "$OUT" "${OUT}.log"
tail -n 40 -F "${OUT}.log"
```

两组按顺序使用同一张卡。修改 `GPU` 即可选卡。检查启动参数时在环境变量中添加 `DRY_RUN=1`，仅加载并验证检查点，不渲染、不创建输出。可选 `CHECKPOINT=latest` 或 `CHECKPOINT=final`，默认 `best_rd`。

## 输出解释

每个源实验分别生成一个子目录，包含：

| 文件 | 内容 |
| --- | --- |
| `plan.json` | 检查点路径及哈希、源步数、β、失真系数、视角和预算 |
| `actions.jsonl` | 每个动作的预测、精确标签、方向判断、误判代价和假想对数优势变化 |
| `windows.jsonl` | 每窗口视角、采样占比、候选数、反事实次数和耗时 |
| `summary.json` | 按正档位、三档混合保留、回退保留及采样参照分组汇总 |
| `complete.json` | 完成状态与模型、概率、历史保持不变的校验 |

`sign_error_fraction` 衡量收益方向误判率；`mean_absolute_error` 衡量预测数值误差；`mean_regret` 统计误判动作对应的真实率失真代价。`hypothetical_update.wrong_directions` 直接检查实际概率更新是否朝向更优动作。不同 β 的误判代价包含各自通信项，应结合 β 和动作类型解读。

不保存图片、PLY 或新检查点。服务器审计运行期间，再开展已有真实结果的本地统计检查。
