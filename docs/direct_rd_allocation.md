# 冻结收益网络，直接换挡与真实 RD 验收

本实验从全 q2 出发，冻结编解码器和收益网络，单独检查分配层。默认读取已有收益实验的 `checkpoints/final/state.pt`；预训练尚无 final 时读取 latest。`training.json` 与完成标记记录真实来源窗口和网络更新数。若有第 700 窗口独立权重，可显式使用 `--benefit-checkpoint /path/to/state.pt`。

候选评估沿用批量四档整块重建和空间特征。输入背景就是已接受的实际档位图。每 10 个窗口汇总预测，按平均预测 RD 收益排序，同一编码块最多改变一个点。q0 直接参与选择。

每批提案使用固定校准视角、2 次配对 AWGN，以及包含 XYZ、元数据、尺度帧的完整通信成本计算真实 RD。改善至少 1e-6 才接受；失败时尝试排名前半、前四分之一的动作，否则撤回。每次验收后清空累积提案。接受旧图的校准评估缓存复用，未验收候选在 checkpoint 内保存。

默认 2000 个候选窗口，对应至多 200 次整批验收；收益网络更新数保持不变。最后不足 10 窗口也执行验收。验收 RD 在这组有限校准条件下单调改善，最终测试集仍独立评估。固定 β 的 RD 优化与“开销低于 q2、质量达到 q2”分别报告。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs
GPU=2
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
BENEFIT="$PWD/output/truck_soft_direction_gpu2_20261030_152022"
OUT="$PWD/output/truck_direct_rd_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_direct_rd_allocation.sh \
  "$CODEC" "$BENEFIT" truck "$OUT" --beta 0.1 \
  >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

先确认空闲 GPU。复用原实验不会覆盖原输出。中断恢复时固定原 OUT，并在相同命令前添加 `RESUME=1`；批次可通过 `--allocation-batch 64` 调整。其他配置保持一致。checkpoint 保存默认间隔为 10 窗口，未保存的窗口会重算。

## 保存与查看

- `review.tar.gz`：默认下载包，包含初始／最终图和档位图、曲线、完整指标、小型配置及最终 uint8 档位数组。
- `review/summary.json`：测试集 initial、latest、uniform_q2、uniform_q3 的 PSNR／SSIM／LPIPS、真实成本及 RD，另列相对 q2 的双目标结果。
- `metrics/audits.jsonl`：每次提案数量、接受数量、各次回退、预测改善与真实 RD 变化。
- `metrics/loss.jsonl`：候选推理耗时、验收耗时、显存、评估点数和实际档位分布。
- `checkpoints/latest/state.pt`：唯一恢复存档，含冻结收益网络、稀疏提案、调度器、随机状态及实际地图，不含旧 replay、Adam 或重复编解码权重。恢复仍需原 CODEC 和场景数据。

仅初始和最终保存图片。全 q2 结果复用初始评估；全 q3 只保存指标。查看包不包含 checkpoint。可重新生成：

```bash
python -m gaussian_jscc.direct_rd_artifacts "$OUT"
python -m json.tool "$OUT/review/summary.json"
```

本地验证采用 CPU 小场景和配对执行测试；CUDA 一致性在服务器启动预检中检查。整场景收益以服务器测试结果为准。
