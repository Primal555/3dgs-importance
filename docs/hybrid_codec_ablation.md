# 位置与非位置属性的渲染消融

当前 codec 的 XYZ 与属性来自同一个学习式 payload。以下评估用于区分两类恢复误差，不改变训练结构，也不额外传输信息。

```bash
python benchmark_codec.py \
  --ply /path/to/point_cloud.ply --checkpoint /path/to/learned_codec.pt \
  --source /path/to/scene --out /path/to/new_evaluation \
  --device cuda --channels none awgn --snrs 10 --tiers 1 2 3 \
  --resolution 2 --save-images --hybrid-ablation
```

所有 Gaussian 保留，评估四种参数组合：

| 组合 | XYZ | 其余属性 |
| --- | --- | --- |
| 输入 PLY 参照 | 原始 | 原始 |
| 仅位置恢复误差 | 解码 | 原始 |
| 仅属性恢复误差 | 原始 | 解码 |
| 完全解码 | 解码 | 解码 |

对照图从左到右为原始照片、输入 PLY 渲染、仅位置误差、仅属性误差、完全解码。
混合参数使用同一次传输结果，且按同一个 Morton 排序对齐行，不能直接混合排序不同的 PLY。

`none` 表示不加信道噪声，不代表无损；仍有神经编码瓶颈。
照片参照的 PSNR 与输入 PLY 参照的 codec PSNR 是两种指标，不能混为一谈。

新架构没有中间位置 seed，旧 `--position-seed-ablation` 入口已移除。
已有历史结果仍可用绘图工具查看。

更多说明见 [learned_joint_jscc.md](learned_joint_jscc.md)。
