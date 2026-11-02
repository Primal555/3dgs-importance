# 共同条件下的直接档位分配

本轮使用现有冻结编解码器和冻结收益网络，从全 q2 的整数档位图开始。保持 10 dB、0/8/24/48、beta=0.1、MSE+结构损失、原有 D_ref/C_ref 和真实通信成本统计。

## 运行流程

1. 每窗口继续初筛约5562点。在固定分配优化视角下交替使用两组噪声，原背景不变时复用解码、投影、可见性和空间统计。
2. 每10窗口从每块胜出的候选中选择最多512点：四分之三来自排序前部，其余名额分布在剩余排序范围，防止只重复检查最前面的动作。
3. 对这些候选，在共同的4个分配优化视角及相同的2组噪声下重新预测全部四档。只有共同条件下的结果进入下一步。
4. 最多64点使用缓存投影计算真实四档局部标签，重新选择有利档位。排序仍使用原边际通信成本近似；整批验收使用完整实际数据包成本。
5. 真实复核通过的动作每16个形成独立子批，优先验收。其余网络候选按剩余预算分成独立根子批，避免混入已确认子批。先访问根子批，再拆分失败子批，每轮最多12次真实联合检查。通过后立即更新累计地图，后续检查同时包含此前通过的修改。
6. 地图变化后丢弃旧背景和提案缓存。下一轮在新背景上评价；每100窗口及最终进行完整解码/渲染核对。

4个分配优化视角从训练相机中选取，参与反复优化。独立测试相机只用于初始/最终报告，不用于候选筛选或接受决策。

## 计算开销与取舍

- 512/64/16/12 是重评分、逐点复核、已确认子批大小和联合检查预算，不会对2781个提案全部逐点真实渲染。64限制逐点标签量，执行候选最多仍可有512个。
- 真实复核和子批拆分复用候选四档投影，不再次运行编解码器或收益网络。联合评价使用完整影响区域、深度排序与 SSIM halo，直接计算实际非线性效果。
- 两个固定噪声条件各保存一份背景；仅缓存本轮最多512个候选的投影。地图变化后缓存失效，避免积累全场景候选四档。
- 固定两组噪声用于保证优化目标一致；最终独立测试默认4次采样。最终是否达到全 q2 质量且开销更低，以 `review/summary.json` 为准。
- 重评分及真实复核仍依赖初筛候选池；64点/12次检查可能导致本轮部分动作未执行，日志会给出 `deferred_batches`。这些是明确的计算预算。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
BENEFIT="$PWD/output/truck_soft_direction_gpu2_20261030_152022"
OUT="$PWD/output/truck_matched_rd_gpu2_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_matched_rd_allocation.sh \
  "$CODEC" "$BENEFIT" truck "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

默认2000个候选窗口。使用全 q2 起点，便于与上一版直接比较。已有收益网络默认取 BENEFIT 中的 final checkpoint；实际来源步数和权重散列记录在 plan/training.json 中。

新入口中断后，用同一个 OUT 和相同参数，添加 `RESUME=1` 再启动。新入口不会读取旧 direct 实验的恢复状态。恢复所需的是 `checkpoints/latest/state.pt`、原编解码器和场景数据；新 checkpoint 内含收益网络推理权重，恢复时不依赖旧回放池。

## 结果与耗时

- `metrics/loss.jsonl`：初筛时间、总窗口时间、背景缓存命中和显存。
- `metrics/audits.jsonl`：`rescore_seconds`、`exact_seconds`、`local_audit_seconds`，以及重评分数、真实标签数、实际检查数、成功提交和剩余子批。
- `predicted_actions_rejected_by_teacher`：共同条件下网络仍认为有利，但真实局部复核发现有害的动作数。
- `teacher_changed_best_tier`：真实标签与网络选择的最佳档位不同的数量。
- `execution_candidates`、`network_only_actions`：进入联合验收的总动作数及其中未逐点复核的数量；每次验收的 `verified` 标记区分两类子批。
- `metrics/full_checks.jsonl`：局部累计 RD 与完整渲染 RD 的误差。
- `review/summary.json`、`review/endpoints.csv`：最终画质、总信道使用及相对全 q2 的比较。
- `review.tar.gz`：供下载分析的小型结果包，仅含指标、初始/最终图和最终档位图；checkpoint 单独保存。

运行时间以服务器日志为准；本地验证只验证数值和调用复用。启动会执行 CUDA 一致性 preflight。
