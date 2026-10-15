# 信道恢复后联合微调（Truck，2000＋1000）

本实验不改变网络结构。从已经训练好的完整 modular checkpoint 接续，
保留语义两端、信道两端、空间上下文、去噪/邻居修正分支以及实际符号预算。
不是随机初始化，也不是仅加载语义教师、重建随机信道网络。

## 更新范围与目标

| 本轮步骤 | 更新参数 | 冻结参数 | 目标 |
| --- | --- | --- | --- |
| 1–2000，`channel_bootstrap` | 完整信道编码器、完整信道解码器（含已有修正分支） | 语义编码器和语义解码器 | `MSE(y_hat, stop_gradient(y))` |
| 2001–3000，`channel_render` | 完整信道编码器、完整信道解码器、语义解码器 | 语义编码器 | 原 PLY 场景渲染 RGB MSE ＋ `1 × latent MSE` |

固定真实 AWGN SNR=10 dB；两阶段均按 q1→q2→q3→mixed 循环。
mixed 是固定评估种子下的随机正档位混合，不是学习得到的部署分配器；本实验不训练分配器或 q0。
每 4 步更新循环一次，不增加额外优化步。当前 error-gates 来源的预算为 8/24/48 个复符号/点，
以实际 checkpoint 配置为准，不能被 CLI 默认 4/12/24 覆盖。XYZ 仍经可靠 16bit 路径交付。

信道两端 LR=1e-4，联合阶段语义解码器 LR=1e-5。联合目标中的系数 1 是沿用的可调工程超参数，
不是最优性结论。新实验使用新 Adam，两阶段切换也重建 Adam；本实验的断点续训则保留本阶段 Adam 和 RNG。
不继承来源实验的 neighbor-only、denoiser-only 或 freeze-semantic 优化限制。
渲染沿用 replay，避免整场景编解码计算图常驻显存。

这里不是只修复颜色：联合 RGB 渲染目标会经透明度、形状、方向与颜色等属性路径更新语义解码器。
保留 latent MSE 是为了继续约束信道恢复 y；它不直接约束语义解码器，因此不保证 clean 画质不下降。
本轮先测量这种折衷，不额外加入尚未验证的 clean 蒸馏损失。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
RUN="$PWD/output/truck_error_gates_gpu3_20261015_120326/neighbor_plain10"
OUT="$PWD/output/truck_channel_joint_gpu3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/output"
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_channel_joint_finetune_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

GPU=3 是启动示例，启动前确认该卡可用。脚本接受任一已训练 arm 目录；传入 error-gates suite 根目录时，
会明确选择 `neighbor_plain10`，并打印完整初始化路径。若要测试门控版，显式传入 `neighbor_gated10`。
不会覆盖源实验。`--init-codec` 使用 checkpoint 内的完整结构配置，不支持在该入口修改结构或符号预算。

断点续训时使用原 OUT（不要重新生成时间戳），保持原先全部实验选项：

```bash
CUDA_VISIBLE_DEVICES=3 RESUME=1 nohup bash scripts/test_channel_joint_finetune_truck.sh \
  "$RUN" "$OUT" >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

## 输出与判读

- `training.json`：完整来源路径、来源步数、加载范围、新阶段预算与学习率。
- `metrics/loss.jsonl`：逐步 latent/render MSE、各模块梯度与实际参数更新量、耗时与显存。
- `metrics/validation.jsonl`：每 500 步及阶段边界，clean、无噪声有限信道、10 dB q1/q2/q3/mixed；包括 latent MSE/NMSE、各属性误差、相对原 PLY/照片的 PSNR/SSIM 和颜色偏差。
- `metrics/test.jsonl`：最终测试视角，另含 0/5/10/15/20 dB 的评估（不是多 SNR 训练）。
- `images/initial`、`images/final`：同一验证视角的 clean 与 10 dB 各档对照图；不保存中间轮次图。
- `checkpoints/initial`、`end_channel_bootstrap`、`end_channel_render`、`latest`、`final`：训练状态。
- `charts`：目标分项、latent 恢复、各路径画质、模块梯度/更新量曲线。

本轮步数从 0 重新计数，3000 为本轮完成；来源的历史步数保存在 `source_step`。
先比较 0→2000 的特征恢复收益，再比较 2000→3000 的真实噪声画质收益与 clean 变化。
q1/q2/q3 分开观察，避免只看 mixed 掩盖高档退化。图中模糊椭球是否改善仍需实际渲染结果确认。

## 之前“每 4 步颜色训练”的范围

`semantic_color_every=4` 只作用于 `semantic_bootstrap`：每第 4 次普通属性更新替换为方向 RGB MSE。
会更新语义编码器、语义解码器共享特征与 DC/SH 输出头；尺度、旋转、透明度输出头的梯度被清空，
该步不受 Adam 动量更新。共享特征变化仍可能间接影响其他属性输出。
信道两端不参与该步。本次跳过语义预训练，不重新执行该颜色步骤；联合阶段使用原有场景渲染目标。
