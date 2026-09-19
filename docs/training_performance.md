# 当前 codec 的训练性能测试

`benchmark_training.py` 测量 learned_joint 全场景前向、渲染和反向传播，不进行优化器更新，不覆盖权重。

当前 block 默认包含 256 个 Gaussian；`blocks-per-batch` 表示一次处理多少个这样的块。它不改变每个点的档位，也不把整块强制设成同一档。

最新 `train-learned` / 两阶段入口默认 `replay`：第二阶段逐批重算编解码器以降低显存。
第一阶段局部预训练不重算。没有混合机制或自动回退；`direct` / `checkpoint` 仍可显式选择。
计时工具也接受 `--backward direct replay`；但下面的历史benchmark目标含辅助项，
不能直接当作当前两阶段纯渲染训练的精确耗时。完整训练日志会记录CUDA峰值显存。

发送端使用局部空间网格，接收端使用窗口注意力。replay 先恢复完整场景并计算渲染梯度，然后复用相同信道随机数逐批重算 codec。完整场景和渲染器本身仍占显存。

```bash
CUDA_VISIBLE_DEVICES=0 python -u benchmark_training.py \
  --ply /path/to/point_cloud.ply \
  --checkpoint /path/to/learned_codec.pt \
  --source /path/to/scene --out /path/to/new_timing_output \
  --blocks-per-batch 16 32 --backward replay \
  --warmup 1 --iterations 3 --snr 10 --channel awgn \
  --resolution 2 --training-data-device cpu --device cuda
```

该历史计时入口默认重建辅助权重 1、投影权重 .05，且不裁剪梯度；
这并非最新 `render_mse_v1` 的纯图像MSE目标。
可用 `--clip-mode` / `--clip-norm` 测量明确启用裁剪后的开销。此计时不包含验证、优化器更新、checkpoint 写入或 teacher 缓存首次生成，不能直接视为完整训练墙钟时间。

输出 `results.json`、`timings.jsonl` 和 `config.json`，包含排除预热后的单步耗时及峰值显存。可增加 `--backward checkpoint replay` 比较两种反向路径。

不要用旧 4096 点块、旧几何网络的计时推断当前架构速度。必须使用当前 learned_joint 检查点重新测量。

训练脚本、停止条件及结果说明见 [主说明](learned_joint_jscc.md)。
