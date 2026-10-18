# β=0.005 固定部署图精修

复用当前模块化训练流程，默认从 β=0.005 最终模型和已保存的十次采样部署图开始。固定 q0/q1/q2/q3，XYZ 仍为可靠 16bit；四个模块都训练 1000 步，学习率 `1e-5`，10 dB AWGN，每四步保留原有独立轮换的全 q1/q2/q3 前缀保护。使用原 PLY 渲染 RGB MSE，不添加属性/latent 损失，不再计算速率梯度，不再采样、更新分配器。保持原点序、Morton 块及 q0 空洞，不重排保留点。

这是已有模型的精修分支，**首次启动保留模型权重，但 Adam 重新开始**；不是原训练的精确续跑。可用 `MODE=joint` 从同一模型和已学习概率表另开继续联合训练对照，两组均新建 Adam、同一步数及学习率。对照仍以单次采样学习、十次规则部署；与固定图的差异同时涉及分配变化及训练/部署配置，不应解释成单一机制因果。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/truck_beta_parallel_gpu2_20261017_213438/beta0p005"
OUT="$PWD/output/truck_fixed_finetune_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_fixed_deployment_finetune.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

另起新 OUT，添加 `MODE=joint` 环境变量即可运行对照，不会自动启动第二组。GPU 号按空闲情况修改。脚本参数后可追加 `--manifest PATH` 迁移数据；`--resume` 在同一 OUT 恢复模型、Adam、RNG、步数及固定档位，预算/配置必须不变。

每 250 步记录部署及固定 q1/q2/q3 对原 PLY/照片的 PSNR、SSIM、LPIPS 与实际通信成本。只在初始和最终保存代表图像；最终评估全部测试视角。沿用 `metrics/loss.jsonl`、`metrics/validation.jsonl`、`metrics/test.jsonl`、`deployment/final/tiers.npy`、`checkpoints/latest` / `final` 与 `complete.json`。固定图模式不保存概率表，也不导出额外 PLY。

判断重点：固定成本下部署画质是否改善、前缀能力是否保持、是否优于继续联合训练的率失真折衷。固定图时 β 仅表示源分配的来历，训练记录中的 `rate_loss=0` 是有意设计。
