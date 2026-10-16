# 双编解码器版本的图像质量评估

本次只扩展评估，不改变语义/信道网络、训练目标、档位预算、学习率或阶段步数。
PSNR、SSIM 越高越好，LPIPS 越低越好。现有 SSIM 继续保留；新增 LPIPS 的逐视角、逐噪声记录、控制台摘要与图表。

## 启用范围

以下脚本默认附带 `--lpips`，额外参数末尾的 `--no-lpips` 可以显式关闭：

- `scripts/test_modular_codec_truck.sh`
- `scripts/test_channel_joint_finetune_truck.sh`
- `scripts/test_clean_color_truck.sh`
- `scripts/test_modular_snr_controls.sh`

直接调用 `python -m gaussian_jscc.modular_experiment` 或 `modular_snr_controls` 时，手工添加 `--lpips`。
底层 `evaluate_paths` 默认关闭 LPIPS，保持未修改的历史消融入口兼容，不给所有旧实验强加网络下载。
独立 integrated 单编解码器路线及原始 3DGS `metrics.py` 本次不修改。

LPIPS 是评估而非训练损失，不加入优化器、模型参数、checkpoint，也不改变后续训练样本/噪声的随机数状态。
历史 checkpoint 可以在原训练参数不变时启用 LPIPS 续训，但不会自动补齐旧轮次的 LPIPS。
主训练入口的 `training.json` 保留首次启动配置；每次调用的当前协议见 `metrics/image_metric_protocol.json`。
独立 SNR 控制入口把协议写在 `experiment.json`、`summary.json`；两种入口的每条评估记录也都单独携带协议。
已经启动的 Python 进程不会因为拉取代码而自动获得新指标。

## 实现与计算口径

使用仓库 `lpipsPyTorch` 的预训练 VGG 实现、校准权重版本 `0.1`。不使用随机网络或降级替代值。
每次整场景评估构建一个冻结、eval 模式的评估网络，复用到本次全部路径、视角及噪声试验；评估结束后不保留为训练模块。
不调用每对图片都会重新创建网络的函数式快捷入口。

- 输入是同视角的完整 RGB 图，不是四联拼图，也不是编码特征 y。
- SSIM 与 LPIPS 使用裁剪到 `[0,1]` 的显示 RGB；LPIPS 额外显式变换为 `2*x-1`，即 `[-1,1]`。
- 不做缩放、图像平移对齐、自动白平衡或颜色校正。
- 继续保留原有未裁剪渲染 RGB 的 MSE/PSNR，避免改变历史指标定义。协议中明确区分这两种输入口径。
- 分别对 Source PLY 渲染和真实照片计算；clean、有限无噪声、AWGN 下的 q1/q2/q3/mixed 均覆盖。
- 每次噪声试验先求全部评估视角均值，再对试验求均值。
  `*_std_trials` 是噪声试验的视角均值标准差，不是相机之间的标准差，也不是多个训练种子的置信区间。
- 继续使用原有固定视角、固定评估种子与符号槽位配对。LPIPS 构建和推理不消耗后续训练的 RNG 状态。

注意：原仓库 `metrics.py` 虽也使用 VGG，但不同库/输入归一化可能导致数值差异；论文比较时必须固定实现、预处理和分辨率。
LPIPS 不是百分比，也没有可通用解释成画质等级的固定阈值。它不能替代颜色偏差记录或多视角一致性检查。

## 首次启动前检查权重

服务器环境需具备与 torch 匹配的 torchvision。首次运行需要下载 torchvision VGG16 权重和 LPIPS 校准权重，缓存到 torch hub；不要把它们加入 Git。
先在服务器执行，避免长训练启动后才发现权重下载失败：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
python -m gaussian_jscc.perceptual_metrics --device cpu
```

成功时输出 `identical_image_lpips: 0.0` 以及网络与版本信息。CPU 预热后的权重也可用于 CUDA 评估。
缺少依赖、下载失败或无效图片会明确报错，不会写入零值冒充成功。

原本的 2000 步信道恢复 + 2000 步联合微调命令保持不变：

```bash
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_channel_joint_finetune_truck.sh \
  "$RUN" "$OUT" --channel-bootstrap-steps 2000 --channel-render-steps 2000 \
  >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
```

`RUN`、`OUT` 仍需先设置，OUT 使用新目录；示例 GPU 不代表当前空闲。

## 文件、字段和图表

- `metrics/validation.jsonl`、`metrics/test.jsonl`：`source_ssim`、`photo_ssim`、新增 `source_lpips`、`photo_lpips`，各自的 `*_std_trials` 与 `per_view_trial` 细项。
- 主训练入口的 `metrics/image_metric_protocol.json`：本次调用的指标协议；SNR 控制入口见 `experiment.json` 和 `summary.json`。
  关闭 LPIPS 时明确记录 `enabled: false`，指标字段不伪造为零。
- 控制台验证摘要同时打印原 PLY 参考的 PSNR、SSIM、LPIPS。
- `charts/validation_ssim.png`、`validation_lpips.png`：分别展示对原 PLY/照片的 clean、无噪声、10 dB 各档指标。
- `charts/ssim_snr.png`、`lpips_snr.png`：最终测试 SNR 扫描，两类参考分开显示，误差棒含义同上。
- 保留原有 PSNR、loss、latent MSE/NMSE 与颜色图；不增加中间轮次渲染图片。

绘图契约：训练进展与信道 SNR 分开；clean/noiseless/AWGN 分面避免九条线挤在一起。
档位使用固定颜色和不同点形。稀疏验证节点只画观测点，不制造平滑曲线。
旧日志缺少 LPIPS 时跳过相应值，不插值、不补零；SSIM 旧日志可直接重画：

```bash
python -m gaussian_jscc.modular_plots "$OUT"
```

## 给已完成的 checkpoint 补算（不重新训练）

可使用已有 SNR 控制入口，在新目录重新渲染最终 checkpoint：

```bash
RUN="$PWD/output/truck_channel_joint_gpu3_20261016_001659"
EVAL_OUT="$PWD/output/truck_joint_quality_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_modular_snr_controls.sh \
  "$RUN" "$EVAL_OUT" --snrs 10 --policies fixed_both --reference-snr 10 \
  >"${EVAL_OUT}.log" 2>&1 &
```

这只运行评估，更新次数为零，原 run 和 checkpoint 只读。
对当前只在 10 dB 训练的模型，`fixed_both --reference-snr 10` 与部署 `trained_range` 在此处使用相同的网络条件。
指标见新目录 `metrics/test.jsonl` 与 `summary.json`，终端也会显示三项指标。
这个控制入口原有图仍是 SNR 条件对照；主训练入口才自动生成上面四张新增质量图。
