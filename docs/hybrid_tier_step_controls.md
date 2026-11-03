# 正档位步幅对照

入口 `gaussian_jscc.hybrid_tier_step_controls` 从正在运行的混合分配实验读取一次固定 checkpoint。源进程继续运行，新实验写入独立目录。

## 本轮变量

| 项目 | 行为 |
| --- | --- |
| q0 保留学习 | 沿用源算法、上限、学习率、历史和真实纠偏 |
| q1–q3 上限 | 独立参数，默认 1.0；实际单次上限仍为该值乘候选点 trust |
| 不限幅选项 | `--unbounded-tier-updates`，直接执行原始镜像更新；保留溢出检查与 -700 数值支持下限 |
| 对照选项 | `--tier-max-log-odds 0.25`，与源原上限一致 |
| 收益网络与编解码器 | 全程冻结，复用原权重、尺度及空间输入 |
| 训练预算 | 保留全局步号，默认继续到源配置总目标 2000，只运行剩余步数 |

没有重复的 700 窗口预训练。源的概率表、全部估值历史、纠偏、候选覆盖、相机调度和随机状态都进入分叉。新实验的初始图对应分叉时分配；本轮校准最优结果从这一初始状态开始选择。待处理的源 control/calibration 事件保留，新初始评估不更新通信代理。原空间输入行为在本轮维持一致，隔离步幅改变的影响。

实际 q0 轨迹会随新正档位分配和渲染反馈变化；q0 更新算法和参数保持相同。

## 先固定一次公共快照

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

SOURCE="$PWD/output/truck_hybrid_tiers_gpu0_20261103_124400"
SNAPSHOT="$PWD/output/truck_tier_step_snapshot_$(date +%Y%m%d_%H%M%S)"
python -m gaussian_jscc.hybrid_tier_step_controls \
  --source "$SOURCE" --out "$SNAPSHOT" --snapshot-only
echo "$SNAPSHOT"
```

该命令只使用 CPU 和文件复制。输出 `source_step` 和源文件 SHA256。原 latest 由训练进程原子替换，快照复制持有同一打开文件，得到一份完整固定状态。快照目录包含 `source_state.pt`、`benefit_network.pt`、`source.json`。

## 启动上限 1.0

```bash
GPU=1
OUT="$PWD/output/truck_tier_step1_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_hybrid_tier_step_controls.sh \
  "$SNAPSHOT" "$OUT" --tier-max-log-odds 1.0 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
echo "$OUT"
tail -n 40 -f "${OUT}.log"
```

## 启动正档位不限幅

另选空闲 GPU，使用同一个 SNAPSHOT：

```bash
GPU=2
OUT_FREE="$PWD/output/truck_tier_step_free_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_hybrid_tier_step_controls.sh \
  "$SNAPSHOT" "$OUT_FREE" --unbounded-tier-updates >"${OUT_FREE}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT_FREE}.pid"
echo "$OUT_FREE"
```

这两条启动命令是独立选项，根据实际空闲卡运行。若要从公共快照补跑原强度对照，使用新 OUT 和 `--tier-max-log-odds 0.25`。不提供上限参数时默认 1.0。

默认终点为源总目标。例如快照为第 600 步，则继续 1400 步到 2000。可显式 `--target-step 1000` 运行到第 1000 步，终点必须大于快照步号。各组对照使用相同终点。

## 恢复

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/test_hybrid_tier_step_controls.sh \
  "$SNAPSHOT" "$OUT" --tier-max-log-odds 1.0 >>"${OUT}.log" 2>&1 &
```

恢复时保持上限、target-step 和 OUT 一致。恢复使用 OUT 中已固定的源副本，原实验 latest 的后续变化不影响结果。

## 新增日志

每 10 步打印 `tier-step` 行，完整数据位于 `allocation/metrics/loss.jsonl`：

- `tier_raw_log_odds_step_mean`：限幅前，本次候选两两对数优势的最大变化量，按候选平均。
- `tier_applied_log_odds_step_mean`：实际落到表上的相同变化量，包含数值下限的影响。
- `tier_applied_log_odds_step_quantiles`：实际步幅的 p50、p90、p99。
- `tier_raw_log_odds_step_quantiles`：限幅前步幅的 p50、p90、p99，用于检查原始提议的长尾。
- `tier_step_limited_fraction`：被正档位上限缩小的候选比例。
- `tier_update_scale_mean`：限幅比例的候选均值。
- `tier_actual_to_raw_step_ratio`：实际步幅总量／原始步幅总量，按原始步幅加权。
- `tier_candidate_trust_mean`：当步候选点的 trust，区别于全部保留点平均值。
- `tier_probability_tv_mean`：当步候选条件三档概率的平均总变差。
- `tier_probability_peak_before_mean`、`tier_probability_peak_after_mean`：候选更新前后最大概率均值。
- `tier_numeric_floor_fraction`：本次提议触及 -700 数值支持下限的候选比例。

不限幅时 `tier_step_limited_fraction` 为 0；如果数值下限生效，实际步幅仍可能小于原始步幅。概率集中程度和真实部署 RD 同时比较，不能仅依据概率集中判定收益。

## 输出

初始与选中结果图、部署档位图、末步与全 q2/q3 指标沿用混合版。优先下载 `review.tar.gz` 或读取 `review_summary.json`。包中排除源快照、checkpoint、网络权重与 replay；继续训练所需状态保留在服务器 OUT 下。

本地定向验证包括：q0 同输入逐位一致、原上限分叉与原实验续训状态相同、独立上限与不限幅计算、固定源与断点恢复。最终画质和速度由服务器确认。
