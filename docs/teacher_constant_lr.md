# 教师组恒定学习率对照

对照实验为 `truck_teacher_views32_cal32_gpu2_20261113_123503`。本轮从相同的初始概率表开始，随机种子为42，保留分支学习率始终为0.1，正档位学习率始终为0.4，训练2000步。使用现有 `--no-early-stop` 关闭自动降学习率和早停。每100步校准选模，末步也独立保留和评估。

沿用每步32个训练视角、32个固定校准视角、32个测试视角和4次噪声试验；183个训练视角的名单与对照一致。β=0.5，10 dB，档位为0/8/24/48，保留限幅0.25，正档位限幅2.0，候选批量128。原有历史统计、置信度、真实纠偏及其采样方式保持当前配置。编解码器与源高斯属性冻结。

目标尺度继续读取 `truck_teacher_views64_shuffle_gpu2_20261112_145947`，与对照的引用来源一致：D_ref=0.001671516249358023、lambda_s=0.00629856382701326、C_ref=43258896。成本使用 `digital_side_file_v1`，属性按实际发送复符号计数，XYZ、完整元数据、块尺度分别按序列化文件计费。侧信息实际传输验证沿用独立的 `digital_side_delivery` 入口。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git fetch origin
git switch experiment/feedforward-student
git pull --ff-only origin experiment/feedforward-student
conda activate maskgs

GPU=2
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
REFERENCE="$PWD/output/truck_teacher_views64_shuffle_gpu2_20261112_145947"
OUT="$PWD/output/truck_teacher_constant_lr_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 nohup bash scripts/test_teacher_constant_lr.sh \
  "$RUN" "$REFERENCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

启动日志应包含 `constant learning rates, full step limit`。`metrics/loss.jsonl` 每步记录 `lr_scale=1.0`、`effective_keep_lr=0.1`、`effective_tier_lr=0.4`。完成标记应为 `step=2000`、`stop_reason=step_limit`、`final_lr_scale=1.0`。

检查重点是1400步之后的校准RD、概率变化量，以及末步测试结果。分别比较选中版本与末步的PSNR、SSIM、LPIPS、实际通信成本和RD；末步是否保持收益是本轮核心。下载 `OUT/review.tar.gz` 即可分析。中断恢复使用同一 RUN、REFERENCE、OUT，设置 `RESUME=1`。
