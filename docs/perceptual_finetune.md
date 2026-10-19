# 固定部署：MSE＋SSIM / MSE＋LPIPS

仅新增两组，不重跑纯 MSE。与 `truck_fixed_finetune_gpu2_20261019_000527` 的纯 MSE 精修保持相同起点：
`output/truck_beta_parallel_gpu2_20261017_213438/beta0p005` 的最终模型及最终部署档位。
**不要把已经精修 1000 步的纯 MSE 模型作为这两组的初始化**，否则不再是相同步数的损失对照。

## 默认权重的文献依据

1. **MSE＋SSIM**：参考 [Semantic Communication for Task Execution and Data Reconstruction，III-C/IV](https://arxiv.org/html/2510.20067)。该文使用 `(1−β) MSE + β (1−SSIM)`，并报告 `β=0.25` 的折衷设置。我们只借用重建部分的相对权重，保持 MSE 系数为 1，写成 `MSE + (1/3)(1−SSIM)`。这与 `0.75 MSE + 0.25(1−SSIM)` 成正比，不是说损失数值或实际优化轨迹完全相同。本文的 β 不是本项目分配器的通信惩罚 β。
2. **MSE＋LPIPS**：参考 [Generative Joint Source-Channel Coding for Semantic Image Transmission，V-B](https://arxiv.org/html/2211.13772)。该节明确写出其 DeepJSCC 前向算子基线训练目标为 `MSE + LPIPS`，因此采用相对权重 **1**；不宣称该数值是其所有 GenerativeJSCC 配置统一设置。
3. [NTSCC，式（26）](https://arxiv.org/html/2112.10961) 支持混合感知目标，但该式包含 GAN，不能将其中未明确给出的数值当作我们的两项损失权重来源。本实验不引入 GAN。

以上是**有文献出处的试验起点，而不是针对当前 3DGS 精修验证过的最优权重**。尤其现有 MSE 约为 0.002，而 `1−SSIM`、LPIPS 可能约为 0.1，因此感知项可能主导数值。数值占比不等于梯度占比，须观察 PSNR/SSIM/LPIPS 的共同变化，不能凭总 loss 给两组排名。文献任务、网络和训练范围不同，本实验不是其完整复现。

权重可通过 `SSIM_WEIGHT` / `LPIPS_WEIGHT` 覆盖；直接 Python 入口可用 `--image-aux-weight`。缺省 Python 权重也会根据目标解析为上述值，记录在 `training.json`、checkpoint settings 中。

## 不变的配置

- Truck，可靠 16bit XYZ，q1/q2/q3 为 8/24/48 个复符号。
- 固定最终部署 q0–q3，保留原始 Morton 块和 q0 空洞；不重新训练、采样分配器。
- 四个模块均可训练；首次启动保留权重、重建 Adam。每组 1000 步，LR=`1e-5`，10 dB AWGN，seed=42。
- replay 反传；多视角逐张累积场景梯度。每四步执行独立循环的 q1/q2/q3 前缀保护，**部署主任务与前缀保护均使用本组相同的混合损失**，仍平均两者的编解码器梯度。
- 参考图为同相机原 PLY 渲染，不是照片。MSE 对未截断 RGB 求均值；SSIM/LPIPS 对显示范围 `[0,1]` 计算。SSIM 沿用仓库 11×11 高斯窗口；LPIPS 沿用 VGG v0.1，转为 `[-1,1]`，不缩放/裁剪图像，不更新 VGG 权重，但保留对恢复 RGB 的梯度。
- 不添加 latent、属性或通信惩罚梯度。训练日志 `aux_loss=0` 仍指原属性辅助损失关闭，**不代表本实验的感知损失为零**。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/truck_beta_parallel_gpu2_20261017_213438/beta0p005"
OUT="$PWD/output/truck_perceptual_finetune_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_perceptual_finetune.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

默认在同一张卡**顺序**运行两组，分别保存在 `$OUT/mse_ssim` 和 `$OUT/mse_lpips`。不会重复跑纯 MSE。LPIPS 训练反传比评估更占显存，不建议未经检查直接并跑。

只启动一组时，在 `nohup` 前添加 `LOSS=mse_ssim` 或 `LOSS=mse_lpips`；可以用不同 GPU 和 OUT 单独启动。中断后用同一 OUT、同样配置及 `LOSS=中断的组`，在脚本最后追加 `--resume`；恢复该组 Adam、RNG、步数和固定档位，不要改权重或总步数。迁移路径可追加 `--manifest PATH`。

可先执行 `python -m gaussian_jscc.perceptual_metrics --device cpu` 检查/缓存 VGG 权重；缺失时明确报错，不以假 LPIPS 或静默退回 MSE 代替。

## 保存与对比

每 250 步验证固定 q1/q2/q3 和实际部署，记录对原 PLY、真实照片的 PSNR/SSIM/LPIPS 及通信成本；最终全部测试视角 × 两次噪声。只在初始、最终保存代表恢复图和部署档位颜色图，不保存每次验证图片。

- `metrics/loss.jsonl`：`image_mse`、`image_ssim_loss` 或 `image_lpips_loss`、`image_aux_weighted`、`image_aux_to_mse_ratio`、`view_mse`、`view_objective`。
- `render_loss` / `loss`：部署主任务的混合损失，不可当成 MSE 计算 PSNR。
- `prefix_anchor_*`：执行前缀保护时记录对应分项；`optimization_image_loss` 是该次实际使用的主任务/前缀平均目标值。
- `metrics/validation.jsonl` / `metrics/test.jsonl`：**评估口径不变**，PSNR 始终来自图像 MSE，不来自训练混合 loss。
- `training.json`：实际权重、文献链接、输入范围、训练模块、源 checkpoint。
- `checkpoints/latest` / `final`，`deployment/final/tiers.npy`，`complete.json` 沿用原结构。旧纯 MSE 默认行为和历史精确续训配置不变。

对比已有纯 MSE 最终结果以及两组最终 `metrics/test.jsonl`，重点检查实际部署与各固定前缀的三指标，而不是以总损失跨组排名。固定档位和符号预算不变，metadata 压缩长度仍按最终模型标识实测，可能有极小字节差异。

## 本机验证范围

21 项针对性 CPU 检查通过，覆盖真实缓存 VGG 的输入梯度及 replay/direct 梯度一致性、混合损失分项、SSIM 组精确续训、旧 MSE 流程、原分配梯度、评估 LPIPS 边界和两组脚本参数。另通过 Bash 语法检查。未执行全量回归或真实场景 CUDA 训练，不以单元测试宣称画质改善或显存足够。
