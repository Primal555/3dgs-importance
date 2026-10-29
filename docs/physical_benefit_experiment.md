# 物理特征收益评估实验

## 运行内容

本实验使用已训练的多场景编解码器，训练 Truck 的收益评估网络和逐点档位表。编解码器全程冻结，保留 10 dB、0/8/24/48 档位和当前可靠坐标路径。

入口：`gaussian_jscc.physical_benefit_allocation`。

### 评估网络

每个 q0–q3 状态有 158 维输入，共享 64／32 隐藏层的 MLP，共 12322 个参数。输入包括原始属性及近邻统计、基线可见贡献、参考颜色和当前颜色残差、候选属性，以及整块邻居的有符号变化。真实 XYZ 的 32 近邻一次查询后缓存；每个窗口的每个相机只累计一次可见性统计。

q0 点使用存在权重为零的源属性探针，查询其真实位置附近的可见性和参考颜色；探针不改变基线图像。遮挡前 alpha 独立于 early exit 累计，完全遮挡点也保留覆盖量。实际可见贡献仍遵循渲染器的深度顺序和提前终止。

对档位 q，网络预测：

`v_q = [(MSE_q - MSE_0) / D_ref, lambda_s × (S_q - S_0) / D_ref]`

其中 S 是 `1-SSIM`。三组二维输出以 q0 为零点；负值代表失真降低。选择代价为 `sum(v_q) + beta × ΔC_q / C_ref`，越小越好。q0 的代价固定为零，网络不接收档位编号和一阶偏导。

真实标签沿用整块换挡与相同噪声，重新投影并准确处理遮挡、跨 tile 和 SSIM 邻域。标签生成后立即积分，replay 保存 `3×2` 数值及紧凑特征。同一候选与当前场景完全相同时，直接复用当前图像。

### 训练

| 项目 | 默认值 |
| --- | --- |
| 预训练 | 200 个窗口 |
| 固定尺度拟合 | 前 50 窗口，只使用训练记录 |
| 每窗口完整标签组 | 64，覆盖 4 个视角 |
| 每窗口网络更新 | 16 次，每次 4 组 |
| 分配训练 | 2000 步，显式另行启动 |
| 优化器 | AdamW，LR 1e-4，weight decay 1e-4 |
| 正则化 | 跨状态共享 dropout 0.05，梯度范数上限 10 |
| 损失 | 分量 Huber 回归＋代价加权 RD 成对排序 |
| 排序权重／平局容差 | 1／固定训练 RD 尺度的 0.02 |

排序以同组多视角平均后的四动作成本为监督。误选的真实代价越大，排序权重越大。接近平局的标签不参与排序。上述超参数是本实验默认设置，可通过 `--decision-weight`、`--tie-fraction`、`--dropout`、`--weight-decay` 修改。

统计特征会丢失空间细节；基线可见性在候选形状变化时提供近似条件。真实教师保留完整干预。实验重点是验证紧凑表征能否改善泛化，需比较验证 regret、方向误判、高代价误判占比，以及最终部署率失真结果。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
python -m pip install 'scipy>=1.9'

RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_physical_benefit_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_physical_benefit_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
printf 'OUT=%s\nLOG=%s.log\n' "$OUT" "$OUT"
tail -n 40 -F "${OUT}.log"
```

先检查 `pretrain_complete.json` 和 `metrics/benefit_validation.jsonl`。日志 `metrics/loss.jsonl` 增加 `regression_loss`、`decision_loss` 和 `train_regret`。`grad_norm` 记录裁剪前范数。

预训练完成后，使用同一个 OUT 显式继续分配：

```bash
CUDA_VISIBLE_DEVICES=2 RESUME=1 nohup bash scripts/test_physical_benefit_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 --phase allocation >"${OUT}.allocation.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.allocation.pid"
tail -n 40 -F "${OUT}.allocation.log"
```

重开终端时请重新设置 RUN 和实际 OUT。中途恢复预训练使用 `RESUME=1` 并保留原参数。两阶段均保存当前最新权重；恢复时必须保留完整 `replay` 和 `checkpoints` 目录。此实验使用独立 checkpoint family，需从新 OUT 开始。

首次启动会编译扩展并检查 CUDA 渲染、标签积分和可见性统计一致性。当前本地验证范围为 CPU；GPU 时间和峰值显存由服务器日志记录。CUDA 原子累加可能有末位浮点差异，CPU 的逐项相同恢复测试不代表 GPU 位级复现。
