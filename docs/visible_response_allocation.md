# 可见贡献评估器实验

本版本集中测试共同保留价值和换挡差值的监督尺度，以及遮挡和邻居响应输入。沿用冻结的编解码器、10 dB 信道、0/8/24/48 档位、原始 RD 目标和加速候选重建。

## 实现内容

### 1．独立监督尺度

网络继续预测相对 q0 的三档价值，每档包含归一化 MSE 和加权结构损失两个分量。设三个正档位的标签为 `V1,V2,V3`，分别监督：

- 共同价值：`mean(V1,V2,V3)`。
- 档位相对价值：`Vq - mean(V1,V2,V3)`。
- 换挡差值：`Vnew - Vold`。

各分量分别采用训练标签绝对值的 90% 分位数作为尺度。中心化量和换挡量的尺度下限为共同价值尺度的 `1e-3`，绝对下限为 `1e-10`。共同价值、中心化量使用等权 SmoothL1；换挡记录独立计算 SmoothL1。两个记录分支仍等权采样和平均。仅损失中的误差进行缩放，预测值、通信成本和最终 RD 判断保留原单位。分量独立归一化也会调整两类分量在评估器拟合中的相对梯度，实际选档仍使用原有的 MSE＋加权 SSIM＋通信成本。

前 50 个窗口仅使用训练部分拟合输入及监督尺度，之后固定。验证标签始终隔离。

### 2．可见贡献输入

每个训练视角把当前有效高斯按中心深度分为 12 层。4 次 RGB 渲染分别累计这些层的真实 alpha 合成权重。累计结果给出每层前后的剩余透射率，从而描述目标深度的遮挡程度。层内精确次序由上下界保留；q0 候选使用正档位重建几何查询相同上下文。

在原有 677 维视角特征后新增 272 维：

| 特征 | 维度 |
| --- | ---: |
| 三个正档位相对 q0 的整块 RGB 贡献变化，上下界分别保留，池化到 3×3 | 162 |
| 整块投影区域的当前 RGB 与参考残差，池化到 3×3 | 54 |
| 选中点在三个正档位的可见 alpha 上下界，池化到 3×3 | 54 |
| 整块投影区域的归一化宽、高 | 2 |

整块贡献包含当前点换挡导致的邻居属性变化，并保留颜色变化的正负方向。贡献估计使用基线透射率；改变几何和透明度后的精确遮挡关系仍由完整干预渲染监督。输入在标签产生前固定。

## 训练与开销

- 默认 200 个预训练窗口，前 50 个窗口拟合尺度，之后每窗口更新网络 4 次。
- 概率表在预训练期间固定，编解码器全程冻结。
- 默认 32 次干预预算生成 10 个完整四档组，实际进行 30 次干预渲染。
- 4 个训练视角额外进行 16 次可见性渲染。每个视角仅构建一次深度上下文，供全部候选复用。
- 日志新增 `visibility_context_render_calls`，计时包括 `visibility_context_seconds` 和 `visibility_features_seconds`。服务器单步耗时需要实测。
- 新特征需要重新进行 200 个预训练窗口。旧实验和原加速版入口保留。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

RUN="$PWD/output/successive_mip360_gpu1_20261023_142948"
OUT="$PWD/output/truck_visible_response_gpu1_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_visible_response_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
printf 'PID：%s\n输出：%s\n日志：%s\n' "$!" "$OUT" "${OUT}.log"
tail -n 30 -F "${OUT}.log"
```

完成后查看 `comparison/benefit_pretrain.json` 和 `comparison/benefit_validation.json`。重点比较同一验证集上的共同误差、换挡 regret、四档选择 regret、固定档位对照以及最严重 10 次误判的代价占比。新增 `supervision_scales` 字段记录实际尺度。

预训练检查后，如需继续分配，在同一 `OUT` 下显式运行：

```bash
CUDA_VISIBLE_DEVICES=1 RESUME=1 nohup bash scripts/test_visible_response_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 --phase allocation >"${OUT}_allocation.log" 2>&1 &
```

## 旧 replay 的 CPU 对照

不占用 GPU，不进行新渲染。需要原实验检查点与原 PLY；PLY 换路径时可用 `--ply` 指定相同文件。

```bash
OLD="$PWD/output/truck_response_value_fast_gpu1_20261029_113406"
CHECK_OUT="${OLD}_scales_$(date +%Y%m%d_%H%M%S)"
nohup python -u -m gaussian_jscc.response_supervision_audit \
  --run "$OLD" --out "$CHECK_OUT" --updates 1000 \
  >"${CHECK_OUT}.log" 2>&1 &
tail -n 30 -F "${CHECK_OUT}.log"
```

两组从同一份选中网络权重开始，使用相同固定 replay、相同采样状态和全新的 Adam。一组保留原损失，另一组拟合训练 replay 的独立监督尺度。该对照只改变损失，不添加新可见性特征。

- `label_scales.json`：两类标签、中心化差值和成对差值的分位数、均方根及拟合尺度。
- `history.jsonl`：固定训练集和验证集的逐阶段误差、regret 和最严重误判。
- `summary.json`：初末结果及源检查点未改变的验证。
- `original.pt`、`separate_scales.pt`：CPU 对照网络，仅供诊断。

将 `--updates` 设为 `0` 可只计算尺度和初始误差。两组损失的数值采用不同归一化方式，应比较同单位的验证误差与 regret。
