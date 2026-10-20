# Truck：只续训信道编解码器

本实验固定当前 SSIM 语义表示，从 **联合训练前** 的信道 checkpoint 接着训练。
不重新初始化网络，不加载已联合更新的最终权重，不修改网络结构、符号预算或学习率。

## 本轮设置

- 来源：`output/truck_perceptual_training_gpu2_20261019_101631/mse_ssim`。
- 权重：`checkpoints/end_channel_bootstrap/codec.pt`，不是 `final/codec.pt`。
- 纯信道更新由 2000 步延长至 6000 步，新增 4000 次实际 optimizer 更新。
- 原实验还包含 1000 步语义渲染，所以全局 step 从 3000 延续至 7000。
- 信道编码器、信道解码器学习率均继承为 `1e-4`，保留它们的 Adam 一、二阶动量和随机数状态。
- 语义编码器、语义解码器均冻结；损失只有 `MSE(y_hat, stop_gradient(y))`。
- 8/24/48 复符号、16bit 可靠坐标、10 dB AWGN、q1/q2/q3/mixed 轮换不变。
- 原本后续的联合训练预算被明确设为 0，不会训练完 6000 步后自动进入联合阶段。
- SSIM 图像目标的历史配置仍保留在元数据中，但没有任何渲染训练步骤，不参与本轮反向传播。

新文件写到独立 `OUT`；源实验只读。`training.json.continuation` 记录源 checkpoint 路径、SHA256、
继承的阶段步数、新增预算以及动量继承信息。`RESUME=1` 是继续完成这 4000 步，不会再追加 4000 步。

## 记录内容

- 每步 `metrics/loss.jsonl`：latent MSE/NMSE、档位、梯度、各模块更新量、耗时及显存峰值。
  续训文件只记录新增步骤：全局 3001–7000，`phase_step` 为信道 2001–6000。
- 开始时重新验证第 2000 步基线；每 500 步验证并保存 latest checkpoint。
- `metrics/validation.jsonl`：clean、有限符号无噪声、10 dB q1/q2/q3/mixed，包含 latent MSE/NMSE、
  对 Source PLY 和照片的 PSNR/SSIM/LPIPS，以及有符号 RGB 偏差和颜色差异误差。
  两次噪声试验及固定验证视角沿用原流程；验证不更新网络。
- 只在最终端点保存代表性图像，位于 `images/final`；不为中间每次验证导出图片。
- 最终评估所有原测试视角，保存 `metrics/test.jsonl`、图表及最终 checkpoint。
- 所有结果位于同一个 `OUT` 中。纯信道每步很快，但整场景渲染、LPIPS 和通信成本统计仍需耗时，
  验证期间可能暂时不打印训练行。

判读时优先看固定验证的各档位 latent NMSE；同时检查误差下降是否伴随颜色和画质改善。
不能仅凭训练 loss 或非零梯度宣称颜色问题已经解决。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

GPU=2  # 改为当前空闲显卡
RUN="$PWD/output/truck_perceptual_training_gpu2_20261019_101631/mse_ssim"
OUT="$PWD/output/truck_channel_continue6000_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/continue_ssim_channel_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 50 -f "${OUT}.log"
```

`nohup` 后可断开本地连接；`Ctrl+C` 退出 `tail` 不会停止训练。
不需要加载当前 `truck_ssim_joint_lr_split_20261020_092212` 的联合训练终点。

只读预检（不加载场景、不创建 OUT、不启动训练）：

```bash
python -m gaussian_jscc.modular_channel_continue --run "$RUN" --out "$OUT" --dry-run
```

中断后先确认原训练进程已退出，设置回相同的 `RUN` 和 `OUT`，再执行：

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/continue_ssim_channel_truck.sh \
  "$RUN" "$OUT" >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 50 -f "${OUT}.log"
```

不要让两个进程同时写同一个 OUT。准确恢复要求可见 CUDA 设备数量不变；单卡任务可以更换物理卡。

## 验证范围

CPU 测试使用真实渐进式编解码器、Adam 和 AWGN，仅相机和渲染后端使用合成替代。
检查动量保留、断点恢复、权重与连续训练逐项一致、语义模块冻结、联合阶段不执行和源文件不变。
这不是在本机重新进行完整 CUDA 收敛实验。
