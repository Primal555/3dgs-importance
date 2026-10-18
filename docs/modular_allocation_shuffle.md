# β=0.005：固定删点集合的档位打乱对照

目的：在相同删点集合和相同 q1/q2/q3 数量下，检验“把高档分给哪些点”是否带来实际画质收益。**不重新训练**编解码器或分配器，不修改原实验结果，也不使用 XYZ-in-payload 实验分支。

默认读取 `output/truck_beta_parallel_gpu2_20261017_213438/beta0p005` 的最终 `checkpoints/final/codec.pt` 与 `deployment/final/tiers.npy`。直接复用已经保存的十次采样部署结果，不重新采样 q0。检查最终模型 ID、步数、场景指纹、数据划分、档位计数和 β，防止混用模型/场景。

## 对照条件

- learned：原部署档位。
- shuffle_101 至 shuffle_105：只在保留点之间全局随机置换 q1/q2/q3。每个 q0 的位置不变，各档位数量、属性符号总数、坐标字节数不变；点顺序、Morton 分块、归一化与原编解码器不变。
- 同一批独立测试视角（默认全部）、10 dB AWGN、每组两次信道试验。各组用相同的逐点最大前缀噪声槽和随机种子，避免档位改变导致噪声错位。LPIPS 默认启用，与原实验口径一致。
- 分别对原 PLY 渲染和真实照片记录 MSE、PSNR、SSIM、LPIPS。实际测量坐标与完整档位元数据的压缩开销；置换可能改变档位表压缩大小，**不能声称真实总成本严格相同**。数字开销折算沿用原实验的净 bit/信道使用假设，不把模拟符号的 float 存储大小当通信流量。

这是对**正档位空间分配**的检验，不是对删点效果的检验。当前接收端包含邻域协作，因此结果也包含档位位置改变后对邻居恢复的影响，不能解释成完全独立的逐点重要性真值。多个置换种子的离散程度不是多次重新训练的置信区间。

## 服务器启动

先激活 `maskgs`，选择一张空闲 GPU；下例使用 2 号卡：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_beta_parallel_gpu2_20261017_213438/beta0p005"
OUT="$PWD/output/truck_beta005_shuffle_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_modular_allocation_shuffle.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

可在脚本两项路径参数之后追加 `--manifest PATH` 迁移数据路径，或 `--shuffle-seeds 101 102 103` 改变置换种子；场景内容仍必须一致。`TRIALS`、`TEST_VIEWS`、`IMAGE_VIEWS` 环境变量分别控制信道次数、测试视角数（0=全部）、保存图像视角数（0=不保存）。每次使用新输出目录。

## 输出与判断

全部结果在一个新目录：

- `summary.json`：各组画质与通信成本，以及逐视角/信道次数配对的 learned−shuffled 差值；给出五次置换差值的均值、标准差、最小值、最大值。
- `metrics.csv` / `metrics.jsonl`：逐视角、逐信道试验的原始指标。
- `layouts/*.json`：各档位占比、实际开销、被置换标签数、档位图哈希和聚合画质。
- `manifest.json`：模型、场景、噪声种子、测试视角和评估口径；`complete.json` 表示评估完成。
- `images/`：只保存 learned 和第一个 shuffle 的两个代表视角，共至多 4 张对比图；不导出 PLY 或额外模型。仅保存这两组的原始点序档位数组，其余由种子复现。

差值中，PSNR/SSIM 为正、MSE/LPIPS 为负，表示 learned 优于打乱。若多数置换都稳定变差，支持当前分配确实选对了升档对象；若差异很小，则当前正档位空间选择的独立收益仍有限。结合元数据成本差值一起判断，不按训练总 loss 排名，也不预设实验必须证明有效。
