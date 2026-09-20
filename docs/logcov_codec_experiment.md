# 对数协方差分流编解码器实验

本实验通过 `learned_split_logcov` 显式启用，不替换既有对照模型。
必须随机初始化；旧的 scale/quaternion checkpoint 不能直接加载为新结构。
目的：检验近球形形状预测导致旋转路径缺乏有效梯度这一机制，而不是承诺已经解决位置误差或渲染质量。

## 结构与损失

发送端从原始 PLY 的尺度和旋转直接构造

`S* = R diag(2 log(s)) R^T`。

几何编码器输入从 XYZ + scale3 + quaternion4 改成 XYZ + S 的六个独立元素。
接收端几何分支输出 XYZ + S6，不再有独立 scale/rotation 头。
外观分支继续输出 alpha、DC、SH，仍接收几何隐特征；两分支共用一个学习得到的信道符号流。

```text
接收符号 → 几何局部注意力 → XYZ、对称 S → matrix_exp(S) → 协方差 Σ
                 ↓ 几何隐特征                              ↓
           外观局部注意力 → alpha、DC、SH ─────────→ Gaussian 渲染器
```

- q0 不发送该点；q1/q2/q3 仍为 8/16/32 个复符号，块内允许不同档位。
- XYZ 仍通过学习型 JSCC 恢复，不发送逐点粗坐标，不增加几何符号保留区。
- 原有公共模型、全局 bbox 和档位元信息假设不变；不能把 payload 当成全部系统成本。
- 第一阶段替换形状项，不叠加四元数、轴比例、法向量惩罚。

`L = (L_position_coarse + fine_weight * L_position_fine + ||S-S*||_F²/9 + L_centered_RGB) / 3`

上述 Frobenius 平方对点求平均，非对角元素计两次。损失在物理对数协方差域计算，
不是六个标准化输出数值的直接 MSE；feature mean/std 只用于网络输入输出的预处理。
位置项及 centered RGB 的定义保留；三组等权、`fine_weight=1` 仍是明确的实验超参数，
不是已经从论文证明的最优权重，也不意味着梯度贡献相等。
新目标命名 `spatial_logcov_v1`，不能将它的总 loss 与旧目标数值直接比较。

## 梯度和数据边界

训练 feature 布局：`XYZ3 | alpha1 | logcov6 | DC3 | SH`。
训练渲染 scene 布局：`XYZ3 | alpha1 | covariance6 | DC3 | SH`。
对称元素顺序均为 `xx,xy,xz,yy,yz,zz`。SH3 时共 58 维，旧 PLY 布局 59 维。
`to_features` 做发送端变换；`to_scene` 生成可微协方差；不要把 feature 直接送入渲染器。

- `matrix_exp` 是矩阵指数，非逐元素指数；在 float64 中计算后转回模型精度，并检查有限值。
- 渲染器直接使用 `cov3D_precomp`，不先分解成尺度与四元数。
- 局部 RGB 响应用协方差投影和 2×2 Cholesky，以 trace 做平滑尺度归一化；
  保留相对 `1e-5` 的二维数值稳定项。该项不注入最终场景渲染。
- 训练路径没有对预测特征向量的 `eigh` 反传。诊断中可在 no-grad 下读取特征值。
- `to_raw`/独立 packet 接收保留标准 PLY 输出接口，只有推理导出时做 `eigh`、轴向修正和四元数转换。
  在带梯度路径误用该导出会明确报错，防止悄悄截断训练。
- 独立 packet 接收可设置 `native_scene=True` 返回协方差 scene；codec benchmark 渲染也采用此路径，
  只有标准 PLY 保存和传统参数指标才转换尺度/旋转。混合消融支持源 PLY 与协方差 scene 配对。
- full-scene direct/replay/checkpoint 都在协方差 scene 上连接渲染梯度；replay 保持同一信道噪声。
- 后续可从本模型 checkpoint 以 `--bootstrap-steps 0 --render-steps ...` 接回现有渲染训练，
  其目标仍为渲染 RGB MSE。当前短实验不自动加入第二阶段或 mask 优化。

这减少了旋转参数化的退化路径，但不能保证预测不会趋于平均形状、不会产生大梯度，
也不能保证 8 个复符号足以恢复精细 Gaussian。异常数值会停止，不靠静默裁剪掩盖。
有限精度仍可能影响极扁协方差；需要实际 CUDA 渲染验证。

## 服务器启动

确认所选卡空闲后执行；示例中的 `2` 按实际空闲卡修改。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_logcov_codec_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 BOOTSTRAP_STEPS=5000 SAVE_EVERY=500 \
  nohup bash scripts/test_logcov_codec_bootstrap.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

默认随机初始化、10 dB AWGN、固定 LR `2e-4`、无梯度裁剪、32 个 block/batch。
训练 5000 步，每 100 步固定局部验证、每 500 步保存 checkpoint。
**训练结束后**自动按 500、1000、…、5000 步的 checkpoint 做固定视角渲染；
不是每训练到 500 步就实时生成图片。默认 8 个测试视角、2 个信道噪声 trial、resolution=4。
可设 `BOOTSTRAP_STEPS=1000` 快速检查，但不能据此判断最终收敛。

结果统一保存在一个实验目录（仅 nohup 控制台日志和 PID 文件沿用同名前缀放在旁边）：

```text
truck_logcov_codec_时间/
  training.json, loss.jsonl, bootstrap_validation.jsonl
  codec_500.pt, ..., codec.pt
  charts/
    bootstrap_logcov_shape.png
    bootstrap_position_validation.png
    training_objectives.png
    ...分支梯度与实际更新统计...
  render_history/
    metrics.csv, results.json, quality_vs_step.png
    validation_images/000500/{1,2,3,mixed}_view00.png
    ...
```

重点观察：同样预算下 PSNR/SSIM 和轮廓是否改善、位置误差、预测与源 Gaussian 的
最大/最小轴比、近球形占比，以及 `covariance_head` 梯度和实际更新。
轴比指标是固定验证块中位数再平均，不是全场景中位数。
独立尺度/四元数误差会受等价轴排列影响，比较形状时优先看协方差和渲染指标。

## 本机验证入口

```bash
python -m unittest discover -s tests -p test_logcov_codec.py -v
python diagnose_split_codec.py \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --out output/logcov_cpu_diagnostic \
  --architecture learned_split_logcov --steps 300 --train-channel awgn
```

CPU 测试覆盖重复特征值处梯度、协方差/PLY 等价、混合档位和独立 packet、
direct/replay/checkpoint 梯度一致性、原生协方差渲染接口、随机短训练与检查点历史输出。
实际 CUDA 栅格化测试在无 CUDA 环境跳过；CPU renderer stand-in 不构成真实 PSNR 证据。
本轮全量回归共123项：119项通过、4项因无 CUDA 跳过；最终 packet/benchmark 接口修改后，
另外复跑了 logcov 专项（9项通过、1项 CUDA 跳过）和 benchmark 专项（4项通过）。

## 已执行的 300 步 CPU 检查（2026-09-20）

本地目录：`output/logcov_cpu_diagnostic_20260920_v2/`，不上传实验数据到 Git。
使用真实 Truck PLY，均匀抽取 4 个完整 Morton 块，每块 256 点：2 块拟合、2 块留出。
随机权重、hidden=96、depth=2、窗口32、10 dB AWGN、LR=2e-4、不裁剪，训练循环交替 q1/q2/q3/mixed。
下表是固定种子下 **q3、10 dB AWGN** 的初始值和 300 步结果；不是全场景指标或跨场景泛化。

| 指标 | 拟合块：初始 → 300步 | 留出块：初始 → 300步 |
|---|---:|---:|
| XYZ RMSE（场景单位） | 34.998 → 6.886 | 43.004 → 25.836 |
| 对数协方差形状 MSE | 10.052 → 6.768 | 10.042 → 10.822 |
| 预测最大/最小轴比中位数 | 2.154 → 15.947 | 2.157 → 10.752 |

这组实验的源轴比中位数约为 54.2（拟合）和 48.7（留出）。输出不再长期局限在近球形，
协方差头有有限非零梯度和实际更新；但形状仍未恢复到源分布，**留出块形状 MSE 反而上升**。
不能把各向异性变大直接视为表面方向学对了，也不能从训练块误差下降推断渲染提升。
300 步整体梯度范数中位数约39.18、最大约95.49，未出现非有限梯度；
这个短记录不能保证长训练不会发生梯度爆炸。

因此这次检查确认实现可训练及机制路径改变，尚未证实真实场景 PSNR 优于旧模型。
下一步按上述启动脚本进行全场景采样训练和固定视角历史渲染，不应只比较新旧总 loss。
