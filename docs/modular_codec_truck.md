# Truck 四模块编解码实验

独立 `modular_jscc_v1`，不覆盖原 learned_joint 或残差增强实验。

发送侧：属性/XYZ → SemanticEncoder → 未归一化 latent y（默认48维）→ ChannelEncoder → 4/12/24复符号前缀。

接收侧：收到的符号、接收掩码、q/SNR → ChannelDecoder → y_hat → SemanticDecoder → 属性。
SemanticDecoder 保留可靠16bit XYZ的嵌入与相对注意力辅助；坐标输出固定为交付值，不学习坐标。
语义模块不接收 q/SNR。clean 路径直接将 y 送入语义解码器，完全绕过通信模块；none 无噪声仍走两通信模块。

| 阶段 | 默认真实更新数 | 更新模块 | 目标 |
|---|---:|---|---|
| semantic_bootstrap | 2000 | 两语义模块 | 原有局部响应 RGB MSE |
| semantic_render | 1000 | 两语义模块 | 原PLY场景渲染 RGB MSE |
| channel_bootstrap | 1000 | 两通信模块 | 原始语义特征与恢复特征的 MSE |
| channel_render | 1000 | 两通信模块、语义解码器 | 原PLY渲染 RGB MSE + λ × 特征 MSE |

所有模块默认固定1e-4，不裁剪，无自动衰减。阶段切换重建Adam，阶段内精确续训保留Adam/RNG。
本版训练目标标记为 `latent_reconstruction_v2`，网络结构与传输格式不变。两个通信阶段均冻结语义编码器，保持目标 `y` 稳定。
信道初始化直接优化 `mean((y_hat - stop_gradient(y))²)`，不运行语义解码器，不使用局部响应损失。
联合阶段保留同一特征 MSE，并允许语义解码器配合残余通信误差调整。特征目标不做批内重归一化，填充点不参与平均。
`--channel-latent-weight` 默认 `1.0`，仅控制联合阶段的特征项，是公开可调的工程系数，不宣称最优；必须为正且有限。
MSE 按有效点数和特征维度取平均，replay 重算保留该项的梯度，不增加实际传输数据。
训练不含概率表、删除、beta或新属性权重；通信阶段循环q1/q2/q3/mixed，真实点全部保留。
默认10dB AWGN；`--train-snrs 0 5 10 15 20`可按每次实际更新均匀抽一个SNR，全场景/replay使用同一值。

## 后台启动

SOURCE_RUN只读取training.json与scenes/truck/scene.json中的路径、维度、PLY指纹和相机划分，绝不读取其权重。
推荐用之前XYZ实验的`xyz`子目录，或完整多场景输出根目录。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/truck_xyz_context_gpu2_20261012_154519/xyz"
OUT="$PWD/output/truck_modular_latent_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_modular_codec_truck.sh "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

GPU=2仅为示例，启动前确认空闲。nohup运行不依赖本地终端；Ctrl+C退出tail不终止训练。
迁移后的路径用`--manifest /absolute/path/manifest.json`覆盖；PLY与相机划分不一致会拒绝运行。

## 保存、继续训练与分叉

旧目标的训练不能通过 `--resume` 静默切换成新目标；更改特征权重也不能作为精确续训。
可以复用旧版 `end_semantic_render/codec.pt`，只保留语义模块与属性统计，重新随机初始化两个通信模块与 Adam。
以下受控实验复用上一轮的同一语义表示，默认只跑 1000 步信道初始化 + 1000 步通信渲染，保持 10 dB 不变：

```bash
OLD="$PWD/output/truck_modular_gpu0_20261012_183141"
RUN="$PWD/output/truck_xyz_context_gpu2_20261012_154519/xyz"
OUT="$PWD/output/truck_modular_latent_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_modular_codec_truck.sh "$RUN" "$OUT" \
  --init-semantic "$OLD/checkpoints/end_semantic_render/codec.pt" \
  --channel-latent-weight 1.0 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

结果集中在一个根目录：checkpoints、metrics、charts、images；场景元数据位于metrics/scenes/truck。
每500步验证/保存latest，另保存四阶段端点和final。`progress.json`不是完成标记，只有`complete.json`说明最终测试成功。
图片只保留初始/最终相同两验证视角的clean及10dB q1/q2/q3/mixed。最终数值覆盖全部test相机、5个SNR与2次噪声。

```bash
# 精确续训：使用同一OUT、原参数；脚本默认值未改变时如下即可。
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_modular_codec_truck.sh "$RUN" "$OUT" \
  --resume "$OUT/checkpoints/latest/codec.pt" >>"${OUT}.log" 2>&1 &

# 从已完成的干净语义表示另开通信实验，不是精确续训。
NEW_OUT="$PWD/output/truck_modular_multisnr_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_modular_codec_truck.sh "$RUN" "$NEW_OUT" \
  --init-semantic "$OUT/checkpoints/end_semantic_render/codec.pt" \
  --train-snrs 0 5 10 15 20 >"${NEW_OUT}.log" 2>&1 &
```

四阶段步数用`--semantic-bootstrap-steps`、`--semantic-render-steps`、`--channel-bootstrap-steps`、`--channel-render-steps`调整。
四模块LR用`--semantic-encoder-lr`、`--semantic-decoder-lr`、`--channel-encoder-lr`、`--channel-decoder-lr`调整。
`--decoder-geometry none`为无坐标辅助对照；`--latent-dim`控制表示宽度。
`--with-direct-control`额外运行原direct骨干3000属性+2000渲染，从头初始化，相同场景与XYZ辅助；不声称容量/FLOP严格匹配。

## 如何读结果

metrics/loss.jsonl按阶段记录真实更新、四模块梯度/更新/冻结检查、SNR和秒数。
metrics/validation.jsonl与test.jsonl明确标记clean、finite_noiseless、awgn；通信未训练时channel_trained=false。
若clean本身弱，优先修表示；若clean强而finite_noiseless弱，检查通信映射/预算；后者强而AWGN弱才主要检查抗噪。
channel_render 改变语义解码器，clean 只是兼容性诊断，不是必然上界，也不是本版训练的直接目标。
检查每个档位的实际通信 PSNR 与特征 MSE 是否同时改善；有限预算和噪声下，不要求特征完全无损恢复。
日志新增 `latent_mse`、`latent_target_power`、`latent_nmse`、训练批内 `latent_mse_by_tier`；不存在的档位记录 null。
NMSE = 特征 MSE / 原始特征均方功率，仅用于诊断，不替代优化用的原始 MSE。
联合阶段分别记录 `render_loss`、`weighted_latent_loss` 与总 `loss`，不能跨新旧目标直接比较总 loss。
验证覆盖固定 q1/q2/q3/mixed；关注低档重建与高档渲染是否出现折衷，不只观察 q3。

通信成本：payload复符号 + 实测16bit delta_zlib等级6 XYZ字节 + 实测包元数据字节。
数字侧信息换算沿用source run的net_bits_per_use（默认2）假设，非FEC实测；模型预共享，不计传权重。
clean没有4/12/24传输成本，latent浮点存储不是通信码率。

本机测试使用真实PyTorch网络/信道/包传输与合成可微图像，不能证明Truck CUDA画质已经改善。

结束时自动生成按四阶段分开的loss图（联合阶段分解两项）、模块梯度/更新图、耗时图、验证三路径曲线、特征MSE/NMSE曲线和最终PSNR–SNR图。
也可以在中途只读取现有日志画图，不启动训练或渲染：

```bash
python -m gaussian_jscc.modular_plots "$OUT"
```

PSNR为逐视角平均，最终误差棒是两次噪声试验的视角均值标准差；不是相机之间的标准差。
通信模块未训练时有限预算结果带`channel_trained=false`，不能用初始化结果判断通信模块上限。
