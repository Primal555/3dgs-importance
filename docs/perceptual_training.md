# 从共同属性初始化开始的两组混合损失实验

本入口替代“只在最后精修时换损失”的主实验。旧精修入口仍保留，但不由本脚本启动。
**不重跑纯 MSE，不接分配器，不读取旧的最终语义/信道权重。**

## 共同起点与训练阶段

默认属性初始化：
`output/truck_clean_color_gpu1_20261014_103802/checkpoints/end_semantic_bootstrap/codec.pt`

该节点已完成 **4000 步属性恢复**（含每四步独立的方向颜色监督），尚未进行任何语义渲染或信道训练。本机只读检查确认它与当前 β=0.005 Truck 记录的场景指纹、坐标配置、相机划分及场景设置一致，语义结构兼容。不能用同一历史目录的 `final` 或 `end_semantic_render` 替代；新入口会拒绝已有后续训练的初始化。

当前 β=0.005 目录只提供数据路径、场景记录和**完整网络配置**，包括 PointNet 双端上下文、8/24/48 符号、完整前缀、逐点去噪及 plain 邻居恢复。旧信道权重、部署档位及概率表均不读取。

| 阶段 | 新训练步数 | 更新模块 | 目标 |
|---|---:|---|---|
| 共同属性恢复 | 复用已完成的 4000 步 | 不再训练 | 原局部响应/方向颜色 MSE |
| 无信道语义渲染 | 1000 | 语义编码器、语义解码器 | 本组 MSE＋SSIM 或 MSE＋LPIPS |
| 10 dB 信道恢复 | 2000 | 信道编码器、信道解码器 | 恢复该组学出的 (y)，latent MSE |
| 10 dB 通信联合渲染 | 2000 | 信道编码器、信道解码器、语义解码器 | 本组混合图像损失＋latent MSE |

联合渲染阶段沿用现有模块划分：**语义编码器冻结，语义解码器允许适应受损特征**，不是解冻全部四个模块。两组各新增 5000 次真实优化器更新，日志 step 从 0 开始，不把复用的 4000 步混入新 loss 曲线。

两组均重新初始化信道模块，seed=42；复制同一份早期语义权重和属性标准化参数，各阶段 Adam 按原流程重新开始。默认模块学习率均为 `1e-4`（沿用编解码器训练入口，而不是旧末端精修的 `1e-5`）。`SEMANTIC_LR` 控制语义模块 LR，包含最后阶段的语义解码器；`CHANNEL_LR` 控制信道模块 LR，不进行隐含自适应下降。

## 两组目标

- `mse_ssim`：`MSE + (1/3)(1−SSIM)`。参考 [Semantic Communication for Task Execution and Data Reconstruction，III-C/IV](https://arxiv.org/html/2510.20067) 中 `0.75 MSE + 0.25(1−SSIM)` 的相对权重。
- `mse_lpips`：`MSE + LPIPS`。参考 [Generative Joint Source-Channel Coding，V-B](https://arxiv.org/html/2211.13772) 明确给出的 DeepJSCC 前向算子基线目标。

系数是文献参考起点，不是本任务的最优结论；可用 `SSIM_WEIGHT`、`LPIPS_WEIGHT` 调整。MSE 对原始未截断 RGB 求均值，SSIM/LPIPS 使用显示 RGB `[0,1]`；VGG LPIPS v0.1 转为 `[-1,1]`，冻结 VGG 参数但保留图像梯度。不缩放或裁剪渲染图像。所有参考图仍为同相机原 PLY 渲染，不是照片。

新增目标仅用于完整图像的两个 render 阶段，不替换前期逐点属性、方向颜色或信道 latent 恢复目标。通信联合阶段保留 `channel_latent_weight=1`。

## 同卡并行启动

用户已确认两组峰值总显存可容纳于同一张卡，**默认同卡并行**。两个子进程继承相同 `CUDA_VISIBLE_DEVICES`，各有独立日志和 PID。父进程等待两组结束；某一组失败时不会杀掉另一组。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_beta_parallel_gpu2_20261017_213438/beta0p005"
OUT="$PWD/output/truck_perceptual_training_gpu2_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_perceptual_training.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
```

父日志 `${OUT}.log` 会给出两组 PID 与日志路径。开始后持续查看两组：

```bash
tail -n 30 -F "$OUT/mse_ssim.log" "$OUT/mse_lpips.log"
```

`tail` 关闭或终端断开不会中断 `nohup` 训练。若脚本在预检查就退出，应先查看 `${OUT}.log`。两组真实 Python PID 另存于 `$OUT/mse_ssim.pid`、`$OUT/mse_lpips.pid`，**外层 `${OUT}.pid` 是调度父进程 PID，不是两个训练 PID**。停止实验应先核对并停止两个训练 PID；只停止父进程不能视为训练已停止。

可用 `LOSS=mse_ssim` / `LOSS=mse_lpips` 只启动一组，或在脚本末尾加 `--sequential` 临时改为顺序执行。属性 checkpoint 换路径时使用 `BOOTSTRAP=/绝对路径/codec.pt`；缺失时明确报错，不静默改用最终模型或重新训练。

正式启动前，可在任意机器执行下列只读预检查（不渲染，不分配 CUDA，不写结果）：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/test_perceptual_training.sh "$RUN" "$OUT" --dry-run
```

### 中断恢复

```bash
CUDA_VISIBLE_DEVICES=2 RESUME=1 nohup bash scripts/test_perceptual_training.sh \
  "$RUN" "$OUT" >"${OUT}.resume.log" 2>&1 &
```

恢复已完成组会跳过；未完成组恢复其 `checkpoints/latest/codec.pt` 中 Adam、RNG 与阶段步数，日志追加。启动前确认旧训练进程已经退出；Linux 上若记录的对应 PID 仍在运行会拒绝重复启动。若某组还没有首次 checkpoint 就失败，需使用新 OUT 或先处理该未完成目录；不会悄悄丢弃进展并从头开始。

续训保持相同源目录、属性 checkpoint、目标权重、预算、学习率、seed 和评估设置。`suite.json` 记录完整参数、共同 checkpoint 的 SHA256 和来源；改变初始化文件或配置会被拒绝。

## 输出与比较方式

两组分别在 `$OUT/mse_ssim`、`$OUT/mse_lpips`，集中在一个实验根目录下。

- 每 500 步及阶段边界记录 clean、无噪声有限前缀、10 dB q1/q2/q3/mixed 的 PSNR、SSIM、LPIPS、颜色误差与 latent 误差。
- 最终全部测试视角 × 两次信道噪声，专注 **10 dB**；不额外扫描 0/5/15/20 dB。原始入口未传 `--test-snrs` 时仍保留旧扫描行为。
- 只保存初始和最终代表图；不导出额外 PLY，也不在每 500 步保存图片。
- `metrics/loss.jsonl` 分开记录真实 `image_mse`、`image_ssim_loss` / `image_lpips_loss`、`image_aux_weighted`、`render_loss`、`weighted_latent_loss`。`loss` 不再等于 MSE，不能据此换算 PSNR。
- 原 PLY 与真实照片分别评估；`metrics/test.jsonl` 指标定义不变。q1/q2/q3/mixed 是固定或随机混合前缀诊断，**不是学习分配后的部署结果**。
- `checkpoints/initial/codec.pt` 可核对两组共同的语义初始化及相同的随机信道初值。阶段末和 latest/final 权重沿用原保存方式。
- 图表分别展示阶段目标、图像 MSE、加权感知项及通信 latent 项，不能跨组直接按总 loss 排名。

这两组之间是相同初始化和预算的目标对照。历史纯 MSE 的后续训练轮次及优化历史不完全相同，可作性能参考，**不包装成严格匹配的全流程三组消融**；按要求不重复启动第三组。

## SSIM 低学习率通信重跑（2026-10-19）

独立入口 `scripts/test_ssim_low_lr_truck.sh SSIM_RUN NEW_OUT` 复用 SSIM 组的
`checkpoints/end_semantic_render/codec.pt`，不使用已退化的联合训练最终权重。
新实验只复制语义权重和标准化参数，信道模块重新随机初始化，Adam 从头开始。
保留来源的完整结构、8/24/48 档位、16bit 坐标和空间上下文配置，不重新训练语义表示。

两阶段分别为 2000 步信道 latent MSE 恢复、2000 步通信联合渲染。
**信道编码器、信道解码器均固定 LR=1e-5，联合阶段语义解码器也固定 LR=1e-5**；
语义编码器全程冻结。联合目标保持 `MSE + (1/3)(1-SSIM) + latent MSE`，
10 dB、seed=42、每 500 步验证/保存，记录 PSNR、SSIM、LPIPS 与颜色误差。
不新增梯度裁剪，旧实验入口及其默认学习率保持不变。

```bash
RUN="$PWD/output/truck_perceptual_training_gpu2_20261019_101631/mse_ssim"
OUT="$PWD/output/truck_ssim_lr1e5_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_ssim_low_lr_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -F "${OUT}.log"
```

GPU 2 仅为示例，启动前自行确认空闲。中断后保留相同 RUN/OUT 并加 `RESUME=1`，
读取该新实验 latest 中的 Adam/RNG 精确恢复，不重新初始化信道。
这不是修改旧运行的学习率后直接续训；同时降低两个信道模块的学习率，
可能减慢有限步数内的收敛，需要结合曲线判断稳定性和最终质量。

## SSIM 分模块学习率联合重跑（2026-10-20）

独立入口 `scripts/test_ssim_joint_lr_truck.sh SSIM_RUN NEW_OUT`，来源使用原 SSIM 组：
`output/truck_perceptual_training_gpu2_20261019_101631/mse_ssim`。
从其 `checkpoints/end_channel_bootstrap/codec.pt` 复制**全部四个模块**和完整配置，
不使用最终退化的联合权重，也不使用信道 LR=1e-5 那组的初始化。
信道已经完成 2000 步 LR=1e-4 的恢复；新实验不再重复训练它。

只进行 **2000 步联合训练**：信道编码器/解码器 LR=1e-4，语义解码器 LR=1e-5，
语义编码器冻结。保留 `MSE + (1/3)(1-SSIM) + latent MSE`、10 dB、seed=42、
q1/q2/q3/mixed 轮换、每 500 步验证/保存及初始/最终代表图。
记录原 PLY/照片的 PSNR、SSIM、LPIPS、颜色误差和 latent 恢复误差。
结构、档位、坐标交付和无梯度裁剪设置均不改变。

新 OUT 的 Adam 和本轮步数从头开始，旧 checkpoint 的训练权重不重置；
它是重做联合阶段，不是从上轮最终模型续训。中断时使用相同 RUN/OUT 加 `RESUME=1`，
恢复该新实验 latest 内的 Adam/RNG/进度。旧低学习率入口保留用于对照。

```bash
RUN="$PWD/output/truck_perceptual_training_gpu2_20261019_101631/mse_ssim"
OUT="$PWD/output/truck_ssim_joint_lr_split_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_ssim_joint_lr_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -F "${OUT}.log"
```

GPU 2 为示例，启动前确认空闲。与历史最终模型比较时应注明初始化和优化历史差异；
本入口不会自动声称学习率改动改善了画质，也不启动分配器或对抗训练。

## 本机验证范围

36 项针对性 CPU 检查通过，覆盖共同属性初始化、两个 render 阶段的真实 VGG-LPIPS/SSIM 梯度、信道 latent 目标、阶段冻结、精确续训、旧 MSE 入口、图表，以及真实子进程并行和失败后继续监督。另通过 Bash 语法检查及实际历史 checkpoint 的只读预检查；未运行全量回归或真实场景 CUDA 训练，不据此宣称画质改善、显存或并跑速度。
