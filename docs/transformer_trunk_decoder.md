# Transformer 主干解码 + 多层位置读出

本版是独立实验入口，不覆盖 `baseline/q3-noiseless-v1`、已有 checkpoint 或结果目录。
新入口从随机权重训练，不把旧 Context 权重硬迁移进新主干。

## 实际结构

```
每点接收符号 + 有效符号 mask
  -> 逐点 MLP + 档位/SNR conditioning
  -> 4 层块内全连接自注意力（pre-LN、残差、4倍宽 GELU FFN）
      -> 第 1 / 2 / 4 层分别 LayerNorm -> 拼接 -> MLP -> XYZ
      -> 最后一层 LayerNorm -> 各自线性头 -> opacity / logcov / DC / SH
```

- 默认 hidden=96、4 heads、decoder_depth=4；这是工程起点，不是论文最优参数。
- `--decoder-depth` 独立于编码器 `--depth`。至少 3 层，读取第 1、居中和最后一层。
- 所有输出都经过 Transformer。保留层内残差，但不再保留主干外的逐点输出旁路或门控 Context 输出修正。
- 不使用源 XYZ、源 kNN、源块中心、手工参考符号，也没有新增坐标旁路。
- 无槽位位置编码或接收端硬 kNN，接收端在一个固定块内对 token 重排等变；这不意味着整个编码器或跨块划分也重排等变。
- 每个输入 token 对应一个输出 Gaussian。q0/padding 不能被有效 token 读到；全空块有安全 mask，不产生全 -inf softmax。
- attention 只覆盖一个 codec block（默认 256 点），不是整个场景。计算随块长度平方增长，显存/速度需服务器实测。
- 默认总参数量由 1,698,615 变为 1,445,787；取消了旧解码端的双流多尺度 Context，不是简单堆大模型。

## 不变项及诊断

编码器结构及同种子初始权重不变，依旧单一 JSCC payload；q3 每点 32 复符号。
固定 q3、无噪声、SNR=10（只作条件输入），随机初始化，固定 LR=2e-4，不裁剪梯度。
保留 `spatial_logcov_v1` 一阶段目标、logcov 输出、数据块和特征统计流程，不加入第二阶段训练。
属性和位置一起训练，并非只训练 XYZ。

`training.json` 记录主干结构和实际 XYZ 层号；checkpoint 保存 `decoder_attention=transformer_trunk` 与 `decoder_depth`。
现有每步梯度/更新日志将 XYZ 多层读出、各 Transformer 层、logcov 和其他属性头分别统计。
`diagnose_position_path.py` 支持新结构的完整位置误差分解；不再伪造不适用的 self_only/Context 分解。

旧窗口/feature_point 模型仍可用于复现实验。`xyz_decoder=additive` 在新分支只是通用配置占位；实际使用多层读出，不执行旧 additive 路径。旧 center/symbol_skip 等组合会明确拒绝。

## 服务器后台入口

确认所选 GPU 空闲后执行，下面 GPU=2 仅为示例，不代表实时空闲：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
OUT="$PWD/output/truck_transformer_trunk_q3_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 BOOTSTRAP_STEPS=10000 SAVE_EVERY=500 \
  nohup bash scripts/test_transformer_trunk_q3.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

可通过 `PLY` / `SCENE` 指定数据路径、`BLOCKS_PER_BATCH` 调整批量、`DECODER_DEPTH` 调整主干深度。
默认每 500 步保存 checkpoint；训练结束后自动在 **同一目录** `render_history/` 中生成每 500 步的图像、PSNR/SSIM 等指标。
这不是每 500 步实时渲染。`RENDER_HISTORY=0` 仅关闭历史渲染，不改变训练。
`Ctrl+C` 退出 tail 不会终止 nohup 训练；不要重复启动同一个输出路径。

## 本地对照

```bash
python scripts/compare_xyz_decoders_local.py \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --out output/transformer_trunk_local_new --steps 300 --every 50 \
  --seeds 42 43 --modes additive --decoder-attentions window transformer_trunk
```

真实 Truck 8 个 256 点块，4 个拟合、4 个留出，两个随机种子；相同采样/监督方向序列、目标、学习率和编码器初始化。
解码器初始化和参数量不同。这是结构对照，不是严格等参数量实验；300 步也不能作为网络能力上限。
本地 CPU 数字是位置/属性误差，不是 CUDA 场景渲染 PSNR。

### 本次实测（2026-09-21）

结果位于 `output/transformer_trunk_local_20260921/`，汇总图表和数据在其中 `comparison/`。
下表使用 200/250/300 步验证值平均；XYZ 是平均 MSE 开平方后的世界单位 RMSE。

| 模型 | seed | 拟合总目标 | 留出总目标 | 拟合 XYZ RMSE | 留出 XYZ RMSE | 留出块内相对 RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧窗口 Context | 42 | 2.0820 | 4.2578 | 2.4708 | 9.8018 | 3.6320 |
| Transformer 主干 | 42 | 1.6189 | 3.3130 | 3.6591 | 8.8458 | 1.4254 |
| 旧窗口 Context | 43 | 2.1500 | 4.3661 | 2.3548 | 10.9630 | 6.3084 |
| Transformer 主干 | 43 | 1.6580 | 3.5505 | 3.9473 | 9.3446 | 1.9414 |

新结构在两种子上降低了拟合/留出总目标、形状误差以及留出位置误差，但拟合位置误差反而更高。
留出共同位置 RMSE 仍约 8.7–9.1，不能称位置问题已经解决，也未通过“拟合与留出 XYZ 同时不退步”的保守筛选。
这比上一轮单纯替换 Context 的结果更有继续研究的价值，但仍没有真实场景 PSNR 改善证据。

最后 100 步梯度范数均值：旧版 seed42/43 为 145.64 / 139.23，新版为 100.81 / 107.62。
这只是相同目标下的小规模观测，不能据此断言梯度爆炸已解决；两版参数结构也不同。
没有非有限梯度，训练未启用裁剪。

验证：完整回归 214 项，210 项通过、4 项跳过；包括独立接收/打包重载、混合档位、全空块、块内重排等变、各层梯度、direct/replay/checkpoint 一致性、训练 CLI 与后台脚本参数。未在本机验证 CUDA 渲染质量或 RTX 4090 显存。

## 设计参考与边界

- [FreeSplatter 的主干实现](https://github.com/TencentARC/FreeSplatter/blob/main/freesplatter/models/transformer.py)：连续 Transformer 后直接读出 Gaussian。
- [NoPoSplat 的位置/属性读出](https://github.com/cvg/NoPoSplat/blob/main/src/model/encoder/encoder_noposplat.py)：独立几何头读取解码特征。

本实现是针对接收符号的迁移设计，不是两篇论文的复现；未引入图像预训练权重、图像 DPT 上采样、相机射线或新损失。
