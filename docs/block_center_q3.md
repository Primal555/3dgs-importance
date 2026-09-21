# 无噪声 q3：块中心与逐点偏移解码（2026-09-21）

## 为什么改这里

冻结基线仍保存在 `baseline/q3-noiseless-v1`（`1eccfc2`）和本地 `output/code_snapshots/q3_noiseless_v1`。原实验文件不覆盖。

上一轮全场景诊断：XYZ RMSE=0.455905；块共同偏移占位置 SSE 的 45.45%。去掉最终 XYZ Context 后，块内相对 RMSE 仅从 0.336735 变为 0.345757，但共同偏移 RMSE 从 0.307342 变为 1.260242。Context 与自身路径已经共同适应，这不能证明 Context 无用；它提示应明确区分共同位置与局部细节。

## 这次结构

旧：`XYZ_i = own(g_i) + gate * context(c_i)`，两个头都可移动整块。

新：

```
接收符号 -> 原有逐点几何特征 g_i
                  ├─ 有效点平均 -> MLP -> 学习得到的块中心 C
                  ├─ 逐点 XYZ 头 -> 减去块内均值 -> 偏移 u_i
                  └─ 原有多尺度 Context -> XYZ 头 -> 减去块内均值 -> 修正 v_i
XYZ_i = C + u_i + tanh(gate) * v_i
```

均值只计入非零档位；填充点/q0 不参与。单点块只有中心分支，空块输出零。去均值让 `mean(XYZ)=C`，Context 的最终 XYZ 修正不能改变共同位置。中心与偏移由同一个原有位置目标共同学习；**没有新增中心监督项、手工坐标或损失权重**。

中心只来自接收符号，不接收真实 XYZ、源块中心或额外粗坐标。所有位置数据仍在原有每点 32 个复符号中；既有元数据协议不变。虽然现在使用块级聚合，档位协议仍为逐点，不改成整块共用档位。这一轮固定所有有效点 q3，仅用于排除干扰。

新选项 `--xyz-decoder block_center`；旧模型默认 `additive`，旧配置哈希不变。两种结构不可直接交叉加载初始化权重，实验从随机权重开始。编码器、logcov/属性路径、功率归一化、`spatial_logcov_v1` 目标、学习率 2e-4、不裁剪、训练/验证划分均保持原协议。相同随机种子下，两者的编码器与非 XYZ 输出初始权重一致。

限制：中心支路现在需要从平均接收特征提取绝对位置；这本身仍可能学不好。自身 XYZ 路径也不再严格逐点独立，因为块内去均值和中心聚合会耦合点。零均值约束不保证梯度范数变小，也不保证渲染提升。

## 后台启动

先确认选定 GPU 空闲，再运行；以下 GPU 编号只是示例。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_block_center_q3_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 BOOTSTRAP_STEPS=10000 nohup bash scripts/test_block_center_q3.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

默认每 100 步固定块验证，每 500 步保存 checkpoint。训练结束后，脚本自动渲染每 500 步 checkpoint 的测试视角；图片和 PSNR 等指标在同一输出目录的 `render_history/`，不是训练期间实时生成。SNR=10 仅是网络条件，信道为 none。

新增 `charts/bootstrap_xyz_decomposition.png`：块共同偏移 MSE、块内相对 MSE，均为固定验证块/试验的等权均值（世界单位平方），只记录不优化。原 CSV 同时加入这两项。它们不是全场景点数加权统计，也不是像素误差。中心头梯度单独归为 `xyz_center_head`。

需要与冻结基线做全场景同口径对比时：

```bash
python diagnose_position_path.py \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --checkpoint "$OUT/codec_10000.pt" \
  --out "$OUT/position_path_full" --max-blocks 0 --blocks-per-batch 8 --device cpu
```

新模型诊断的 `self_only` 是“学习中心 + 自身零均值偏移”，只移除最终 XYZ Context 修正。oracle 中心/仿射校正仅用于解释误差，不是接收端能力或可报告的渲染质量。

## 若收益不够，怎样继续

1. **中心误差降、局部误差不降**：优先改善接收端逐点几何细节，例如用初步预测位置建立几何邻域进行残差细化，保留自身符号路径；不使用真实邻域作为接收端输入。先验证接收端预测邻域足够可靠，避免重新依赖错误 Context。
2. **两者都不降**：固定编码器，对完全相同的接收符号训练一个位置专用读出器；若它能恢复而联合解码器不能，才优先改解码器容量/任务耦合。读出失败不能单独证明编码端丢失信息，还需对比归一化前后潜变量及更宽中间表示，定位可恢复性瓶颈。
3. **拟合块好、留出块差**：优先检查块分布、离群块、特征统计和泛化，而不是继续扩大网络。坚持同一固定验证集合，并同时看分位数与点加权误差。
4. **位置指标降、渲染仍差**：查看误差相对 Gaussian 尺寸和像素投影的尺度，以及形状/透明度/遮挡影响。只有基础重建已经可用，才考虑恢复渲染优化阶段；本轮不改目标来混淆比较。

贯穿以上各项检查分支梯度、实际参数更新和收敛曲线。大范数不等于已经证明梯度爆炸；也不能以固定阈值 1 裁剪代替定位。最终判据仍是同一测试视角的图像和 PSNR/SSIM，而不是中心损失单独下降。

## 本机检查结果

完整 unittest：162 项，4 项跳过，其余通过。覆盖 q0/填充与单点块、Context 零均值、中心梯度、相同种子的编码器/非 XYZ 输出一致、收发协议、checkpoint 保存加载、direct/replay/checkpoint 梯度一致、CLI 固定 q3 无噪声训练与离线图像历史、新诊断图和实际 Bash 参数构造。

另在真实 Truck PLY 上做了 **100 步 CPU 小块拟合**，不是完整训练或 CUDA 渲染：Morton 排序后抽取 8 个 256 点块 `[40,440,840,1240,1640,2040,2440,2840]`，交替项划分 4 个拟合块和 4 个留出块；每步交替两块，随机权重 seed=42、hidden96/depth2、q3/none、Adam 2e-4、不裁剪。仅复用基线 checkpoint 的输入特征统计，不复用训练权重；损失沿用 spatial_logcov_v1，外观方向固定为三坐标轴。两种 XYZ 头结构和初始输出不同，因此这不是匹配初始误差的消融。

| 100 步后，世界单位 RMSE | 旧 additive | 新 block_center |
| --- | ---: | ---: |
| 拟合块总位置 | 7.326 | 11.992 |
| 拟合块共同偏移 | 4.735 | 11.344 |
| 拟合块相对位置 | 5.590 | 3.886 |
| 留出块总位置 | 9.325 | 7.697 |
| 留出块共同偏移 | 7.538 | 7.573 |
| 留出块相对位置 | 5.491 | 1.378 |

新结构并非全面获胜：本小测试中相对误差更低，但共同位置仍未学好，拟合块总误差更高；不能据此宣称全场景 PSNR 提升。所有检查到的训练梯度有限，没有据此证明梯度爆炸已解决。也不能把这些只拟合四块的 100 步数值与全场景 10000 步基线直接比较。
