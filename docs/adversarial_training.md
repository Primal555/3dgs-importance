# 独立 MSE + LPIPS + GAN 实验

此实验不替换正在运行的 MSE+SSIM、MSE+LPIPS 两组，不训练档位分配器。
仍使用可靠的 16bit XYZ 和 8/24/48 个复符号预算。

## 初始化与阶段

复用同一份已完成的 4000 步属性训练 checkpoint，只复制语义网络权重，信道网络重新初始化：

`output/truck_clean_color_gpu1_20261014_103802/checkpoints/end_semantic_bootstrap/codec.pt`

当前完整网络配置来源（只读配置及场景信息，不加载其后期权重）：

`output/truck_beta_parallel_gpu2_20261017_213438/beta0p005`

后续训练共 5000 步：

| 阶段 | 步数 | 更新模块 | 目标 |
|---|---:|---|---|
| semantic_render | 1000 | 语义编码器、语义解码器；另更新判别器 | 无信道 MSE+LPIPS+GAN |
| channel_bootstrap | 2000 | 信道编码器、信道解码器 | 10 dB latent MSE；不更新判别器 |
| channel_render | 2000 | 信道编码器、信道解码器、语义解码器；另更新判别器 | 10 dB MSE+LPIPS+GAN + latent MSE |

不隐含额外 warmup。生成端各模块学习率默认 `1e-4`，阶段边界新建生成端 Adam。
信道阶段 q1/q2/q3/mixed 轮换不变。判别器参数与 Adam 跨阶段保留，中间暂停。

## 目标与判别器

`L_G = RGB_MSE + LPIPS_WEIGHT * LPIPS + GAN_WEIGHT * mean(softplus(-D(fake)))`

`L_D = 0.5 * [mean(softplus(-D(real))) + mean(softplus(D(fake_detached)))]`

channel_render 另加 `1 * latent_MSE`。默认 `LPIPS_WEIGHT=1`、`GAN_WEIGHT=0.001`。
后者是可调工程起点，不是已验证最优值，也不是照搬论文的公开系数。

MSE 使用未裁剪 renderer RGB；VGG-LPIPS 和判别器使用裁剪至 [0,1] 后映射到 [-1,1] 的 RGB。
LPIPS 权重冻结，但保留输入梯度。

判别器为无条件局部全卷积网络，70×70 感受野，5 个 4×4 卷积，通道 64/128/256/512/1，
步幅 2/2/2/1/1，谱归一化、LeakyReLU，无 BatchNorm、Dropout。
输入为原分辨率渲染图，最小高宽 32，不额外裁块或缩放。
判别器 Adam 默认学习率 `1e-4`、betas `(0.5,0.999)`。

各视角共用判别器，但判别器不接收原图配对条件、相机或 latent。
它判断局部外观是否像真实渲染图，不直接验证内容对应；同视角忠实恢复仍由 MSE、LPIPS 约束。
思路借鉴 [NTSCC IV-A3 式(26)](https://arxiv.org/html/2112.10961)，但不声称复现其 latent 条件 GAN。

每步先冻结 D 参数与谱归一化 buffers，逐视角计算 G 的场景梯度，再 replay 编解码器并更新 G。
随后用该步缓存到 CPU 的、更新 G 之前的脱离梯度真/假图更新 D 一次。
不再解码或重渲染一套假图，不保留 renderer/VGG 图给 D。
为避免初始冻结 D 使用未经估计的随机谱向量，每层先执行 20 次 CPU power iteration；不读取训练数据、不更新 Adam、不改变采样 RNG。

## 启动（服务器 Bash）

选择有余量的卡；不要把第三组默认塞进正在运行两组的卡。以下以 GPU 3 为例：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs

RUN="$PWD/output/truck_beta_parallel_gpu2_20261017_213438/beta0p005"
OUT="$PWD/output/truck_adversarial_gpu3_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_adversarial_training.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

脚本使用 `exec`，后台 PID 对应实际 Python 训练进程。关闭日志查看或 SSH 不会主动结束 nohup 任务。
服务器重启、显存/内存不足或其他进程主动终止仍会中断训练。

覆盖参数（例如 `GAN_WEIGHT=0.0003`）需放在 `nohup` 前，或使用脚本尾部 CLI 参数。
`BOOTSTRAP` 可覆盖公共初始化路径；`SEMANTIC_RENDER_STEPS`、`CHANNEL_STEPS`、`JOINT_STEPS` 可改步数。
CPU 预检、不训练：

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/test_adversarial_training.sh "$RUN" "$OUT" --dry-run
```

精确续训必须使用原 OUT 和原始配置参数：

```bash
CUDA_VISIBLE_DEVICES=3 RESUME=1 nohup bash scripts/test_adversarial_training.sh \
  "$RUN" "$OUT" >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
```

不要在同一个 OUT 同时启动两个任务；不确定是否还在运行时先检查 PID。
`--resume` 读取 `checkpoints/latest/codec.pt`；缺少判别器/Adam 状态或目标配置不匹配时拒绝续训。

## 输出与判读

所有实验产物位于同一个 OUT：`training.json`、`metrics/`、`charts/`、`checkpoints/`、`images/`。
外部 `${OUT}.log` 与 `${OUT}.pid` 是启动日志和 PID，不另外建立训练子目录。

- `metrics/loss.jsonl`：MSE、LPIPS、原始/加权 GAN、D loss、真/假 logits、D 梯度和更新范数、G/D 耗时、显存峰值。
- `charts/objectives_by_phase.png`、`charts/adversarial_diagnostics.png`：分别观察生成损失分量及判别器状态。
- `metrics/validation.jsonl`、`metrics/test.jsonl`：对原 PLY 和照片的 PSNR、SSIM、LPIPS、颜色及 latent 指标。
- 图片只保存初始和最终验证结果，包含 clean / finite_noiseless / 10 dB q1/q2/q3/mixed；中间保留数值，不保存图片。
- 原子 checkpoint 同时包含 codec、生成端 Adam、判别器、D Adam、D 更新次数、进度和 RNG；部署 codec 不调用判别器。

不能用 D loss 或总 loss 下降判定优于其他组。应按同一相机、同一路径和同一档位比较 PSNR/SSIM/LPIPS，
并检查颜色、文字和纹理是否被伪造、不同视角是否出现不一致。GAN 不保证 PSNR 提升。

本机检查为 CPU 小样本接线、真实 VGG 梯度、续训、CLI 和回放一致性；真实 CUDA 场景性能以服务器结果为准。

### 本次验证记录（2026-10-19）

- 新增 8 项对抗实验测试通过，覆盖真实预训练 VGG + 合成可微 renderer、独立入口、阶段切换和精确续训。
- 全量 unittest discovery 运行 413 项：4 项跳过；一次运行中的旧测试
  `test_clean_color.CleanColorTests.test_clean_pipeline_never_calls_channel_and_resume_is_exact`
  在 Windows 替换临时 `progress.json` 时出现 `WinError 5`。该测试单独原样重跑通过，未复现，未针对它修改生产代码；不能据此将该次全量运行标记为全绿。
- 实际 Truck 元数据和公共属性 checkpoint 的 CPU dry-run、Bash 脚本语法及入口 dry-run 通过；dry-run 不创建输出目录。
- 独立代码审查未提出阻断修改；本机未运行完整 CUDA 场景训练。
