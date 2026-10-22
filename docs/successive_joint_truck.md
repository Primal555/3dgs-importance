# 条件增强信道完成后的联合微调 A

独立入口 `gaussian_jscc.successive_joint`，脚本
`scripts/test_successive_joint_truck.sh`。只接受已完成的
`successive_channel_v1` Truck 实验及其 `checkpoints/final/codec.pt`。
原实验只读，输出必须是另一个独立目录。

需要同时更新语义编码器的B对照版本，使用独立脚本
`scripts/test_successive_joint_encoder_truck.sh`，见
[联合微调B说明](successive_joint_encoder_truck.md)。本页A的默认范围和参数不变。

## 参数范围和目标

保留完整训练权重、8/24/48 累计复符号预算、48 维实数 y、可靠 16bit XYZ、
已交付 binary16 块尺度和已训练的空间修复分支。没有随机重建信道、重做纯信道阶段，
也不接入档位分配器。语义编码器及属性均值/标准差冻结；语义编码器保持 eval 模式，
防止训练态缓冲量漂移。其余三个模块实际更新：

| 模块 | 更新 | 默认学习率 |
| --- | --- | --- |
| 语义编码器 | 否 | — |
| 信道编码器 | 是 | 1e-5 |
| 信道解码器，含已有邻居修复 | 是 | 1e-5 |
| 语义解码器 | 是 | 1e-5 |

默认 2000 次真实 Adam 更新，固定 10dB AWGN，q1/q2/q3/mixed 独立循环。
mixed 为随机正档位，并非学习得到的部署分配。q0 仅用于填充，不学习删点。

`L = MSE(render(received), render(source PLY)) + latent_weight * MSE(y_hat, stop_gradient(y))`

`latent_weight` 默认 1，沿用既有模块化联合阶段的工程起点，可公开调整；不宣称最优。
原 PLY 完整场景渲染是图像目标，真实照片只用于评估。此阶段不再添加原纯信道训练的
局部响应、方向颜色辅助和无噪声约束：属性路径由实际场景渲染梯度共同调整，特征恢复
MSE 继续约束信道。SSIM、LPIPS 和颜色偏差均作为指标，不加到训练损失中。

新阶段建立全新 Adam，不继承原有主干/空间训练动量。断点续跑则保留当前阶段 Adam、
RNG、游标和待完成的评估。没有新增自动裁剪或强拒绝更新机制。

使用原始分组和完整场景 replay：先重建场景并计算场景梯度，再用同一噪声重算编解码器，
反传渲染及特征目标。根模型保持 train 模式，保留尺度量化的训练梯度；评估通过真实
尺度字节序列化/解析。语义编码器及属性统计哈希每次保存均核对不变。

## 服务器启动

GPU0 只是示例，请选择实际空闲显卡。RUN 必须指向这次**已完成条件增强实验**，
不是之前的 response-scratch 教师目录。如果实际完成目录不同，修改 RUN 那一行。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

GPU=0
RUN="$PWD/output/truck_successive_gpu0_20261022_091448"
OUT="$PWD/output/truck_successive_joint_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_successive_joint_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -F "${OUT}.log"
```

附加 `--dry-run --device cpu` 只核对来源、完整性及配置，不创建输出、不加载场景、不训练。
路径迁移使用 `--manifest /absolute/path/manifest.json`，仍需满足同一 PLY 指纹与相机拆分。
默认直接执行 2000 次联合更新，可用 `--steps`、`--lr`、`--latent-weight` 显式覆盖。

精确续跑保留原 RUN 和 OUT，不重新生成时间戳：

```bash
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_successive_joint_truck.sh \
  "$RUN" "$OUT" --resume >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -F "${OUT}.log"
```

## 输出和判读

- `initialization.json`：完整权重来源、来源步数和校验值，新 Adam 的空状态。
- `metrics/loss.jsonl`：渲染/特征分项、各模块梯度与更新、真实学习率、相机、耗时及 GPU 峰值。
- `evaluation/joint_XXXXXX`：每 250 步数值验证，使用保留验证视角，不保存中间图。
- `evaluation/initial`、`evaluation/final`：同一组真实测试相机，默认32视角×4次配对噪声；
  clean、无噪声、10dB q1/q2/q3/mixed，原 PLY/照片 PSNR、SSIM、VGG-LPIPS、颜色与真实通信成本。
  图像仅保存初始/最终两个代表视角。
- `comparison/summary.json`、`comparison/endpoints.csv`：初始到最终的分档变化。
- `checkpoints/initial`、`latest`、`end_joint`、`final`：完整状态与精确续跑数据。
- `complete.json`：最终评估与保存成功后才生成。微调步数从0开始，原训练步数单独保存在 `source_step`。

重点看实际10dB的 q1/q2/q3 是否共同改善、latent MSE 是否明显恶化、RGB 偏差和模糊结构是否改善。
语义解码器发生改变，因此 clean 可以变化，不是必须保持不变的上界。
本阶段仍会保留纯信道试验的尺度成本和实际帧交付边界。
渲染反向比纯特征训练耗时明显增加，不能按上一轮每个特征批次的速度估算。

本机定向 CPU 测试使用真实编解码、尺度、反向、Adam 和续跑；仅渲染/相机使用合成替代。
它验证工程正确性，不代表真实 CUDA 画质或显存峰值；不运行全仓回归或本机全量场景训练。
