# 路线二：四档概率与 Gaussian JSCC 联合训练

本实现保留重要性建模作为方法的一部分。每个输入 Gaussian 拥有可学习四维 logits，经过 softmax 得到“不传、低、中、高”四档概率。它们由最终恢复质量和通信资源成本共同更新，不由注意力权重替代。

## 训练链路

```text
已训练 PLY（完整 xyz / opacity / scale / rotation / SH）
       │                      每个 Gaussian 的四档 logits
       │                                    │
       │                          Gumbel-Softmax 硬采样
       │                                    │
       └───── 局部上下文编码 ← 档位嵌入 / 前缀 mask
                        │
                归一化 → 带噪信道
                        │
              接收 latent → 初始 xyz
                        │
              预测位置上的网格上下文
                        │
               恢复 xyz 和其他属性
                        │
         存在门 = 低 + 中 + 高档的硬选择值
                        │
             MaskGaussian mask 光栅器
                        │
          渲染失真 + 辅助恢复损失 + β 期望资源
                        │
            更新四档 logits 与编解码器
```

输入 PLY 的候选 Gaussian 集合和目标属性固定，不再做 densification，也不移动原始目标。若输入为已经被 MaskGaussian 剪枝的 PLY，只能对剩下的候选继续学习；要从原始全部 Gaussian 开始，应输入 vanilla 3DGS 的最终 PLY。

`GaussianTierMask.logits` 的形状是 `[N,4]`，始终以原始 PLY 行顺序保存。编码时只重排索引，不改变参数与原始 Gaussian 的对应。默认每个点是静态四档概率，训练中随机采样 SNR，所以它学习的是指定 SNR 分布下的资源选择。启用 `--condition-snr` 后，增加同样形状的可学习 SNR 斜率：

```text
scores_i(SNR) = logits_i + slopes_i × (SNR − 10) / 10
```

这仍然是每个 Gaussian 自己的四档概率表。部署时可随 SNR 改变决策；不具有跨场景泛化能力，也不以 Transformer 预测重要性。预算当前通过 β 控制，没有实现显式预算输入或保证总符号数不超限的硬约束。

## 为什么梯度能穿过离散选择

前向使用 one-hot 硬采样 `s_i`，每个点只取一档。反向使用 Gumbel-Softmax 的连续近似梯度。这是有偏直通估计，不是离散决策的精确导数。

```text
existence_i = s_i[1] + s_i[2] + s_i[3]
prefix_mask_i = s_i @ four_prefix_templates
tier_embedding_i = s_i @ embedding_table
```

因此低、中、高档的存在门全部为 1，透明度维持正常；只有零档的存在门为 0。码率降低通过保留更少复符号实现，不把低、中档当成部分透明。

训练 tensor 暂时保留最大 latent 长度以求梯度，但未选符号在解码前被 mask 掉，不算实际传输。零档点对网格累积和功率计算的贡献为 0；网格归一化范围仅由硬选择保留点决定。固定硬选择且无噪声时，保留点输出已与真实 pack/channel/unpack 路径做一致性测试。随机信道两种路径使用相同噪声分布，但随机数流布局不同，不保证同一个 seed 下逐样本输出相同。

渲染联合训练使用本仓库 `mask_diff_gaussian_rasterization`，其 `masks` 输入能为当前未参与图像合成的点提供存在门梯度。这里没有在 vanilla rasterizer 前简单将 opacity 乘零，因为其 alpha 阈值可能截断需要的梯度。部署后零档点真正删除，使用普通 3DGS 光栅器评估保留点。

网格插值的坐标权重现在可以向初始和中间预测位置回传梯度；离散格点索引、硬活跃集合和包围盒选择仍不求导。

## 损失与开销口径

默认复符号档位为 `K=[0,8,16,32]`，可通过 `--rates` 修改四个长度。概率 `pi=softmax(scores)`，期望每点负载：

```text
expected_symbols = mean_i sum_k pi_i[k] × K[k]
L_rate = expected_symbols / K_max
L = 0.8 × L1_render + 0.2 × (1 − SSIM)
    + attr_weight × L_aux + beta × L_rate
```

`L_aux` 是分组均衡的 Gaussian 参数损失加初始位置损失；它按硬保留点计算，但其存在门在辅助项中停止梯度，避免仅为了删除辅助误差而鼓励丢点。模型参数仍获得该辅助梯度。渲染门、符号 mask、档位嵌入和上下文路径继续向 logits 回传梯度。

增大 β 通常会增加降低期望符号数的压力，但不能保证达到某个剪枝率或固定总预算。Gumbel 温度只影响反向连续近似，不自动保证原始 softmax 概率最终变成 one-hot；测试必须同时看采样期望成本和 argmax 硬决策成本。

训练的码率损失只优化期望 JSCC 负载。完整 2-bit q 图被视作固定长度控制信息；实际 zlib 压缩字节不可微，没有用来给分配器反传。`transmit/evaluate` 会测量实际压缩元数据并将其折算开销加到负载上。可靠元数据假设与之前相同；其中只有场景级包围盒，没有逐点坐标。分配参数是发送端状态，接收端不用获得 logits 或 `route2.pt`。

## 服务器运行

```bash
cd /root/MaskGaussian-main/MaskGaussian-main
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PLY="$PWD/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply"
SCENE="$PWD/data/tandt_db/tandt/truck"
OUT="$PWD/output/truck_route2_joint"

python -m unittest discover -s tests -p 'test*.py' -v

CUDA_VISIBLE_DEVICES=0 python -u -m gaussian_jscc train-route2 \
  --ply "$PLY" --source "$SCENE" --out "$OUT" \
  --warmup-steps 20000 --joint-steps 2000 \
  --rates 0 8 16 32 --snr-range 0 20 \
  --condition-snr --beta 0.01 --mask-lr 0.001 \
  --tau-start 1.0 --tau-end 0.3 \
  --resolution 2 --block-size 4096
```

前段只预热 codec，随机采样正档，防止尚未学会恢复的网络给分配器提供失真的初始信号；后段每步采样训练视角和 SNR，对整个场景联合反传。迭代数、β 和温度是可调整超参数，尚不是实测最优配置。全场景渲染、零档候选和梯度都占显存，4096 分块不代表只渲染一个块。每块使用激活重计算节省中间激活，但仍会额外耗时。

已有完整 xyz codec 时可加 `--codec-init /path/codec.pt --warmup-steps 0`，模型结构和归一化统计取自该权重。可选 `--existence-prior /path/p_exist.npy` 使用原始行顺序的 `[N]` 存在概率初始化四档分布；没有该文件也能从默认分布学习。原始二元 `_mask_score` 不能直接加载到四档参数。

只检查 CPU 联合反向通路时：

```bash
python -m gaussian_jscc train-route2 \
  --ply "$PLY" --out output/route2_cpu_check \
  --device cpu --attribute-only --warmup-steps 2 --joint-steps 3
```

`--attribute-only` 用局部参数代理损失验证梯度和接口；它不等价于渲染贡献学习，不能用来宣称视觉性能。正式训练不要加此参数。

输出 `codec.pt` 与 `route2.pt` 必须配对；定期快照也是同后缀配对。`loss.jsonl` 记录渲染/诊断损失、期望符号数、采样档位数量、温度、SNR、codec 和 mask 梯度范数。所有输出目录要求不存在。检查点可部署，但当前联合训练未提供恢复 Adam 状态的严格断点续训。

## 测试与导出

直接在不同 SNR 使用学习到的分配：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m gaussian_jscc evaluate \
  --ply "$PLY" --source "$SCENE" \
  --checkpoint "$OUT/codec.pt" --allocation "$OUT/route2.pt" \
  --snrs 0 5 10 15 20 --trials 3 --resolution 2 --save-images \
  --out "$PWD/output/truck_route2_evaluation"
```

每个 SNR 用 argmax 生成确定的 q；若启用 SNR 斜率，q 可随 SNR 变化。每个 trial 只重采样信道噪声。保留点数、档位分布和实际总信道次数均在 `results.json`，渲染质量相对完整输入 PLY 与相同测试视角计算，包含 q0 丢点的影响。

也可以保存供检查的重要性分布：

```bash
python -m gaussian_jscc export-route2 \
  --ply "$PLY" --checkpoint "$OUT/codec.pt" --allocation "$OUT/route2.pt" \
  --snr 10 --out "$PWD/output/truck_route2_map_snr10"
```

得到原始 PLY 行顺序的 `probabilities.npy [N,4]`、`tiers.npy [N]` 和成本报告。存在概率可定义为 `1-pi[:,0]`，资源需求可用 `sum(pi*K)` 表示；二者均不等于原始 MaskGaussian 的剪枝概率，意义由当前通信目标决定。

## 自动统计图

`train-route2`、`evaluate` 和 `export-route2` 完成后会分别在各自输出目录的 `charts/` 下自动生成 PNG 与 SVG。训练图包含损失与梯度趋势、期望复符号数和四档采样占比；评估图包含质量—SNR、信道开销—SNR、部署档位占比、率失真散点和 Gaussian 参数误差；分配图包含硬档位构成、存在概率 / 期望符号数分布，以及 XYZ 三组空间投影。每组图同时保存用于复核的 CSV 数据与 `charts_manifest.json`。

对已经完成的实验可以直接补画，不需要重新训练：

```bash
python -m gaussian_jscc plot-stats \
  --training "$OUT" \
  --evaluation "$PWD/output/truck_route2_evaluation" \
  --allocation "$PWD/output/truck_route2_map_snr10" \
  --ply "$PLY" --checkpoint "$OUT/codec.pt" \
  --out "$PWD/output/truck_route2_charts"
```

三个输入目录可以只提供其中一部分。`--ply` 只用于分配结果的空间投影，`--checkpoint` 用于读取实际档位符号表。图表依赖 `matplotlib>=3.7`；若训练环境缺少它，核心实验仍会保存完成，终端给出警告，安装依赖后再运行 `plot-stats` 即可。

独立发送与解码：

```bash
python -m gaussian_jscc transmit \
  --ply "$PLY" --checkpoint "$OUT/codec.pt" --allocation "$OUT/route2.pt" \
  --snr 10 --out "$PWD/output/truck_route2_packet"
python -m gaussian_jscc decode \
  --checkpoint "$OUT/codec.pt" --packet "$PWD/output/truck_route2_packet" \
  --out "$PWD/output/truck_route2_received.ply"
```

输入 PLY 内容或顺序不同会被校验拒绝。解码命令不需要分配器权重、原始 PLY 或训练图像。当前场景原始候选保持固定，不支持将同一分配表直接套给另一个场景。

## 当前验证记录

CPU 自动测试 20 项通过，2 项 CUDA 渲染测试因本机无 CUDA 跳过。测试包含四档梯度、零档剔除下的训练 / 真实打包一致性、带噪 Gumbel 激活重计算、源 PLY 顺序与模型身份校验，以及联合训练→分配导出→逐 SNR 收发→独立解码。

实际 Truck 的 883438 个候选还完成了 2 步 codec 预热、3 步局部属性诊断和全量概率导出，记录了非零 mask 梯度。该短程检查没有进行渲染优化，硬 argmax 仍全部为高档；不能据此判断重要性质量、剪枝收益或通信质量。4090 上的正式渲染联合训练与峰值显存仍待验证。
