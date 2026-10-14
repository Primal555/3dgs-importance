# 冻结语义解码器的 latent 误差敏感性检查

本实验只比较既有 PointNet++ 式空间上下文四组实验中的 `baseline` 和 `both`。不训练、不固定 q3 重训、不改变多档位或前缀方案，不调用信道两模块。默认使用 `end_semantic_render`（原实验第 3000 步）的 checkpoint；可追加 `--stages semantic_render channel_render` 同时检查原实验第 7000 步的最终解码器。

## 输入与公平性

- 必须读取与原实验完全匹配的剪枝 PLY，默认来自 baseline 的 training.json；迁移时可用 `--ply` 改路径，但点数、SH 阶数、PLY fingerprint 必须匹配。不能拿本地另一份 Truck 或原始 254 万点版本替代 711,691 点训练源。
- 每个 checkpoint 的场景身份、几何元数据、相机划分、非上下文配置和属性归一化参数必须一致。
- 使用相同的原 Morton 分块与均匀选取的 32 个处理块，每组看到同一批点。默认三次扰动试验、八个固定均匀球面方向。最后不足 256 点的块保留真实点，不加入填充点。
- 每组使用自己的语义编码器生成 clean latent，不互换 latent：不同模型的隐空间基底不能假定语义对齐。
- 每点独立规定 `RMS(delta_i) / RMS(y_i) = epsilon`，默认 epsilon 为 0、0.05、0.1、0.2、0.35、0.5。相同种子、试验与点槽位使用相同的随机扰动方向；跨档幅度线性缩放。零 latent 行不扰动。相对强度包含 latent 均值分量，未做去均值或逐维白化。
- 保留可靠 16bit XYZ；接收侧位置固定。诊断中的正值 q 只表示点有效，不选择通信前缀，因此这不是固定档位训练或码率实验。

## 衡量什么

1. **方向颜色变化**：DC+SH 合成的截断前 RGB，比较扰动输出与该模型自身 clean 输出。没有透明度或形状因子。
2. **局部响应变化**：在同一原始高斯 footprint 网格（25 个探针、黑/白背景）上，比较扰动输出与自身 clean 输出。包含颜色、形状和透明度变化，不混入两组不同的预测采样网格。
3. **与原场景的误差**：另外保存 clean / 扰动输出相对原始高斯的两项误差，避免把本来就不准的模型误判为更鲁棒。
4. **单点扩散**：每个所选块、每次试验只扰动一个点，其他输入 latent 完全保持不变；默认目标点相对 RMS 0.2。记录目标点颜色 MSE、其他点颜色 MSE、颜色及响应的非目标能量占比，以及其他点 RGB RMSE 超过 1e-4 的数量。1e-4 是明确的诊断阈值，非显著性或最优阈值。

每个模型在同一时刻的权重和归一化统计做逐项快照校验；诊断结束必须完全不变。没有优化器，没有信道调用，没有全场景 CUDA 光栅化，也不修改输入 checkpoint。只有元数据所要求的运行时几何绑定。

## 输出与解释

一个目录内保存 `plan.json`、`summary.json`、`curves.csv`、`per_block.csv`、`spread.csv`、`sensitivity.png`、`complete.json`；不保存模型或大批图片。全点曲线的平均值按真实点数加权，P95 从点级误差合并计算，包含扰动试验的重复观测，不把它们称作独立训练样本。单点扩散的采样为每块每次一个目标，不代表整个场景的严格删点/重要性分布。

相同相对扰动下，both 的颜色或响应变化若显著更大，支持其解码映射更敏感。若单点扰动改变较多其他点，支持跨点误差传播。两种现象不相同，不能仅凭扩散点数判断总损伤更大。必须结合 clean 基础误差、曲线及块间差异；本实验不自动下显著性结论。

均匀潜变量扰动并不等同于真实信道 `y_hat-y` 的方向性、偏置、点间相关性。局部响应也没有真实场景遮挡、透视及像素滤波，所以结果不是场景 PSNR，不能独立证明真实信道退化的唯一原因。

## 服务器运行

可以直接用 CPU（默认四线程），不占用正在跑颜色训练的显卡：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/truck_pointnet_context_20261013_151335"
OUT="$PWD/output/truck_latent_sensitivity_$(date +%Y%m%d_%H%M%S)"
DEVICE=cpu THREADS=4 nohup bash scripts/test_latent_decoder_sensitivity.sh \
  "$RUN" "$OUT" --stages semantic_render channel_render >"${OUT}.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -F "${OUT}.log"
```

如只检查初始化前模型，去掉 `--stages semantic_render channel_render` 即可。需要空闲 GPU 加速时，在 nohup 前使用 `CUDA_VISIBLE_DEVICES=空闲卡 DEVICE=cuda`，其他命令不变。
