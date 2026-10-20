# Truck 联合阶段续训

本入口用于把已完成的单场景模块化 MSE／SSIM 联合阶段再延长一段，不支持 GAN、LPIPS 训练控制或多档扫描实验。
与 `--init-codec` 不同，它保留四个模块、共享统计、Adam 一／二阶动量、随机状态、全局步数和阶段档位轮换游标。
不重新进行属性恢复或信道初始化，不更换学习率、损失或结构。新结果写入新目录，原实验只读。

本轮从 `truck_ssim_joint_lr_split_20261020_092212` 的最终 checkpoint 接续：

- 联合步数 2000 → 3000，新增 1000 次真实更新。
- 信道编码器／解码器维持 `1e-4`，语义解码器 `1e-5`；语义编码器冻结。
- 8／24／48 复符号、可靠 16bit XYZ、AWGN 10dB、MSE+(1/3)(1-SSIM)+latent MSE 均不变。
- 重新测量第 2000 步基线，之后在 2250／2500／2750／3000 步验证。
- 保留原来的 PSNR、SSIM、LPIPS、颜色、latent 和梯度／参数更新记录，最终测试仍用 32 个测试视角。
- 新 `metrics/loss.jsonl` 只记录新增更新，step 由 2001 开始；`training.json.continuation` 明确记录父实验和累计／新增预算。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
GPU=2  # 改成当前空闲显卡
RUN="$PWD/output/truck_ssim_joint_lr_split_20261020_092212"
OUT="$PWD/output/truck_ssim_joint_continue_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/continue_ssim_joint_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 50 -f "${OUT}.log"
```

只读检查可用 `python -m gaussian_jscc.modular_joint_continue --run "$RUN" --out "$OUT" --dry-run`。
它检查 CPU checkpoint／元数据并显示扩轮计划，不创建输出、不加载场景、不启动训练。

如果本次续训中断，先确认旧进程已退出，再保留同一个 `RUN`、`OUT`，使用：

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/continue_ssim_joint_truck.sh \
  "$RUN" "$OUT" >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 50 -f "${OUT}.log"
```

`RESUME=1` 只接着完成本次已经设定的 1000 步，不会再次增加 1000 步。
准确恢复要求相同的可见 CUDA 设备数量；单卡可以换到另一张物理卡。首次启动和中断恢复均不要同时启动两个进程写入同一个 OUT。

验证：CPU 小型真实网络、Adam、AWGN 配合合成渲染，检查分段续训与连续训练的权重逐项一致、动量不重置、阶段不重跑、档位轮换不断档。
这不是本机真实 CUDA 收敛测试。
