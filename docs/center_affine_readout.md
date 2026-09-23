# Transformer 中心坐标读出：去掉数据依赖的归一化

本试验只修改独立中心 Transformer 的三个 XYZ 特征读出点：

- 基线 `layernorm`：`gamma * (h - mean(h)) / sqrt(var(h) + eps) + beta`。
- 试验 `affine`：`gamma * h + beta`；gamma 初始化为 1，beta 初始化为 0。

`affine` 是逐通道仿射变换，不是 RMSNorm，也不是新的归一化算法。
它保留特征均值和幅值，且与原 LayerNorm 的可学习参数量、名称和初始化相同。
同一随机种子下所有模型初始参数及后续采样 RNG 均不变，但初始预测自然会不同。
中心 Transformer 内部 Pre-LN、注意力、前馈网络、三层特征拼接及 XYZ 头均不变。
编码器、属性网络和原中心世界距离损失不变；无直接 XYZ 旁路，无额外坐标传输。

需要检验的假设是读出前标准化是否妨碍精确数值恢复，不预先断言其为根因。
移除标准化也可能放大特征和梯度；固定 `2e-4`、默认不裁剪，记录梯度，
遇到非有限损失/梯度/参数会报错，不能把异常当作成功训练。

## 5000 步后台启动

先确认空闲显卡，将下面示例中的 `2` 改为实际空闲 GPU 编号：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main || exit 1

OUT="$PWD/output/truck_center_affine_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 STEPS=5000 BLOCKS_PER_BATCH=32 \
  nohup bash scripts/test_center_affine_readout.sh "$OUT" \
  > "$OUT/console.log" 2>&1 &
echo $! > "$OUT/run.pid"
tail -f "$OUT/console.log"
```

退出 tail 或断开本地电脑不会停止后台训练。随机初始化，不读取旧模型。
只更新中心编码器与中心解码器，固定跑满 5000 步；不提前过门槛、不训练属性、
不进入联合渲染阶段，不模拟 JSCC 信道。本测试的步数是优化步骤，不是全数据 epoch。
批次 32 块，每块 256 点，hidden96，中心 latent32，seed42。

每 500 步保存检查点、四个固定验证视角的渲染图、PSNR/SSIM、拟合块抽样与
留出块中心 RMSE；第 0 步也保存。所有产物在同一 OUT 下：

- `images/000500/view_00/center_only.png`：预测中心＋原始属性，本试验主要观察图。
- `images/000500/view_00/comparison.png`：照片、源场景、完整输出、中心诊断、12bit 诊断。
- `charts/validation.png`：中心 RMSE、PSNR、SSIM 曲线。
- `charts/training_by_phase.png`：中心 loss、模块梯度曲线。
- `validation.jsonl`：`centers` 为留出块，`fitted_centers` 为固定 32 个训练块抽样；
  `render.center_only.source_psnr` 是中心诊断对源渲染 PSNR。
- `loss.jsonl`：每步 loss、梯度、学习率、采样块；定期记录实际参数更新。
- `codec_5000.pt` / `codec_last.pt`：最后一步；`codec_best_center.pt` / `codec_center.pt`：
  固定验证指标选择的模型，不能混淆最后一步与选优结果。

`full.png` 和 full PSNR 包含未训练的属性，不用它们判断本轮中心精度。
原始属性只用于标记清楚的验证诊断，不参与中心优化。

恢复同一轮训练：

```bash
CUDA_VISIBLE_DEVICES=2 nohup python -u -m gaussian_jscc train-center-attributes \
  --resume "$OUT/training_state.pt" > "$OUT/resume.log" 2>&1 &
```

恢复时使用保存的配置、优化器和 RNG，不会把旧 layernorm 训练静默改成 affine。
一般训练入口仍默认 `--center-readout-norm layernorm`；只有本脚本默认 affine。
若需严格 5000 步基线对照，对新目录运行同一脚本并设置 `READOUT_NORM=layernorm`。
不要把 600 步 CPU/batch4 与 5000 步 GPU/batch32 的差异全部归因于归一化。

## 本机初步验证（2026-09-23）

真实 Truck PLY，seed42、batch4、600 步，无相机渲染。与已有 Transformer
基线逐项检查：初始模型张量完全相同，600 步实际采样序列、场景指纹及划分相同。

| XYZ 读出 | 第 400 步留出 RMSE | 第 600 步留出 RMSE | 第 600 步拟合块抽样 RMSE |
| --- | ---: | ---: | ---: |
| LayerNorm 基线 | 4.43739 | 4.59229 | 2.77703 |
| affine 实验 | 4.24482 | 4.72238 | 3.04162 |

完成 600 步，无非有限数值；但末尾精度没有改善，不能宣称已解决定位问题。
此结果支持继续把它作为可证伪的单变量假设测试，而非替换既有默认模型。
5000 步服务器试验需结合中心诊断图像与 PSNR 判断；本机无 CUDA 渲染结论。
本机产物位于 `output/center_affine_readout_cpu_20260923/`，不提交大模型文件。
