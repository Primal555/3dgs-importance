# Geometry-first Gaussian JSCC 架构升级

## 实际改变

本实现保留四档概率表、Morton 分块、局部网格聚合和可变长度 JSCC。
不新增逐 Gaussian 粗坐标，不把完整 PLY 或发送端 Context 交给接收端。

```text
冻结的源 Gaussian → 四档选择 q → 发送端局部 Context / 条件编码
                                    │
                  ┌─────────────────┴─────────────────┐
               几何分支                            属性分支
                  └──────── 同一 k(q) 预算 ────────────┘
                                  ↓
                  块内统一平均功率归一化 → 信道
                                  ↓
             几何符号 → Geometry Decoder → 最终 XYZ
                                               │
             属性符号 → 特征 → 用恢复 XYZ 构建局部 Context
                                               ↓
                               opacity / scale / rotation / SH
                                               ↓
                             渲染 → 更新 codec 与四档概率
```

Geometry Decoder 不读取属性符号，也不经过接收端空间 Context。
发送端几何编码仍能使用共享上下文和原始 XYZ，因此并未禁止几何冗余建模。
属性端训练、推理都使用解码位置，不以干净位置作为训练时的隐含帮助。
XYZ 对属性损失仍可微：前向依赖是单向的，不代表属性损失不能反向影响几何编码。
网格计划只复用于一次 forward 内，不能缓存旧的预测坐标。

## 符号布局与资源记账

所有长度均为**复信道符号数**，不是位数或 Gaussian 参数个数。

| 档位 | 默认总符号 | 默认几何符号 | 默认属性符号 |
|---|---:|---:|---:|
| q0 | 0 | 0 | 0 |
| q1 | 8 | 4 | 4 |
| q2 | 16 | 8 | 8 |
| q3 | 32 | 16 | 16 |

默认二等分是可配置的工程起点，**没有已验证的最优性、几何精度下限或质量单调性保证**。
可以用 `--rates 0 8 16 32 --geometry-rates 0 4 8 16` 显式配置。
两路各自的累计长度须在 q1..q3 为正且非递减；长度划分不是每个 Gaussian 再预测一次。

实际固定布局按档位增量排列：`G1 A1 ΔG2 ΔA2 ΔG3 ΔA3`。
因此旧的 pack/unpack 前缀机制仍然精确选出 k(q) 个符号；接收端通过 checkpoint 中的配置
和已有 q 元数据确定分路，无须传输第二套逐点长度信息。
两路合在一起做每块平均复符号能量为 1 的归一化；不存在额外、免费的几何发送能量。

接收包仍包括六个全局归一化浮点数（包围盒 lower/span）和档位/配置等元数据，
**这些元数据并非零开销**，按已有可靠元数据假设计入通信成本。
`stats.json` 新增 `geometry_complex_symbols`、`attribute_complex_symbols` 和 `geometry_rates`，
前两项之和严格等于 `payload_complex_symbols`。共享模型权重仍假设预先部署。

## 新损失

实现见 `gaussian_jscc/losses.py`。当前默认 `--loss-profile balanced_v2`；
修正原因、继续训练命令见 [损失修正说明](codec_loss_correction.md)。

辅助目标为：

```text
L_aux = w_geo D_geo + w_shape D_shape + w_scale D_scale + w_opacity D_opacity
        + w_dc D_dc + w_sh D_sh
```

- `D_geo`：位置误差先投影到源 Gaussian 的旋转坐标系，再除以局部轴尺度，
  使用 `log1p(Mahalanobis距离平方)`。最小轴尺度通过各轴方差加 `(bbox_diagonal × floor)^2` 限制。
- `D_shape`：对数协方差差的 Frobenius 范数平方除以 9。
  `log(Σ)=R diag(2 log(s)) Rᵀ` 直接计算，不需要矩阵求逆或特征分解；
  同时旋转坐标系、四元数变号和等价主轴交换均不影响该损失。
- `D_scale`：排序后的物理对数轴尺度的 Smooth L1；放大与缩小同等倍数具有相同惩罚。
- `D_opacity`：激活后 alpha 的 Smooth L1，加 `0.1 ×` 未激活 opacity logit 的 Smooth L1。
  尺度/logit 约束读取安全渲染截断之前的预测，避免越界时辅助梯度为零。
- `D_dc`、`D_sh`：使用冻结的 codec 属性统计归一化后分别计算 Smooth L1。

默认权重按 geometry / shape / scale / opacity / DC / SH 顺序为
`1 / 0.25 / 1 / 1 / 1 / 0.25`，`geometry_floor=1e-4`。
这是明确记录的辅助约束配置，不是论文保证的最优系数，也不是可自由降低的可学习 loss 权重。
各项可通过 `--geometry-weight`、`--shape-weight` 等参数调整。
同一 loss profile 下未指定的项保留 checkpoint 配置；切换 profile 时重置到该 profile 默认值，
再应用显式覆盖，并打印迁移提示。旧 `physical_v1` 的计算保持可复现。
修改几何符号划分则需要新架构实例，不能在加载权重时偷偷更改。

codec 预训练：`L = L_aux`。
codec 渲染训练：`L = D_render + attr_weight × L_aux`。
`attr_weight` 现在默认 1，而非 0.1；`train` 渲染阶段学习率默认为 `0.25 × lr`，
可通过 `--render-lr` 设置。同一次训练切换学习率不清空 Adam 状态。
四档联合训练：再加入 `beta × E[k(q)] / k_max`，仍是软率惩罚，不保证硬预算上限。
辅助损失使用 detached 的存在门权重、按源点数归一化，避免通过门的直接导数奖励删除点；
码流路径的 ST 梯度仍然存在。决定删除代价的完整渲染目标保留所有候选点，q0 通过 mask 光栅器处理。

`D_render = 0.8 L1 + 0.2 (1 - SSIM)`，默认参考为冻结源 PLY 的渲染：
`--render-target source`。原始相机图像作为目标需要显式设置 `--render-target images`。
源图按训练相机惰性生成并缓存于 CPU，只在训练端使用，不占用接收端资源，不保留计算图。
首次访问相机会多一次无梯度源渲染；长期缓存占用 CPU 内存。
原始输入 Gaussian 始终冻结，不通过移动源点降低通信失真。

## 并行与兼容性

- `train` 和 `train-route2` 的完整场景阶段都支持 `--blocks-per-batch 32 --render-backward replay`。
- 联合 replay 的分界张量同时包含恢复参数和四档选择，因此能保留 inactive-mask 对 q0 的梯度。
- 同次 replay 复用 Gumbel 与信道 RNG，不重新抽取另一组档位/噪声。
- 新 checkpoint 为 version 3、`architecture=geometry_first`；旧 version 2 的结构和模型哈希保持兼容。
- legacy/version 2 checkpoint 仍可评估，但不能初始化 geometry-first。
  已经完成的 geometry-first/version 3 checkpoint 可以通过 `--init` / `--codec-init`
  继续优化，不需要重新训练网络结构。旧模型仅加载评估时保留原配置和模型哈希。
- `--seed-position-weight` 仅保留参数兼容，在新架构中不生效。
  `return_seed` 返回的是最终 XYZ；旧 seed 消融对新架构没有独立含义。
- 归一化统计仍从训练源场景获得。此升级不等于完成了跨场景泛化训练。
- 训练步数仍是用户指定的迭代预算；本次没有新增自动早停，也没有认定 2000 次渲染是必要量。

## 服务器运行

环境和数据集已经就绪时，无须重新安装原有 CUDA 子模块。
先在仓库根目录设置路径，使用新的输出目录保护旧结果：

```bash
PROJECT="$(pwd)"
PLY="$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply"
SCENE="$PROJECT/data/tandt_db/tandt/truck"
OUT="$PROJECT/output/truck_geometry_first_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=0 python -u -m gaussian_jscc train \
  --ply "$PLY" --out "$OUT" --device cuda \
  --steps 20000 --render-steps 0 \
  --rates 0 8 16 32 --geometry-rates 0 4 8 16 \
  --snr-range 0 20 --channel awgn --attribute-drop 0 \
  --save-every 1000
```

这一步重新训练新 codec，不加载旧 `truck_codec_attr_retrain/codec.pt`。
上述 20000 是示例迭代预算，不是证明充分训练的停止条件。

随后可对新 codec 做渲染优化；例如预留 1000 步预算：

```bash
RENDER_OUT="${OUT}_render"
CUDA_VISIBLE_DEVICES=0 python -u -m gaussian_jscc train \
  --ply "$PLY" --source "$SCENE" --init "$OUT/codec.pt" \
  --out "$RENDER_OUT" --steps 0 --render-steps 1000 \
  --render-target source --blocks-per-batch 32 --render-backward replay \
  --training-data-device cuda --resolution 2 --device cuda \
  --snr-range 0 20 --channel awgn --save-every 100
```

要同时优化四档 mask，则使用匹配的新 codec（可以是上一步产物）：

```bash
JOINT_OUT="${OUT}_joint"
CUDA_VISIBLE_DEVICES=0 python -u -m gaussian_jscc train-route2 \
  --ply "$PLY" --source "$SCENE" --codec-init "$RENDER_OUT/codec.pt" \
  --out "$JOINT_OUT" --warmup-steps 0 --joint-steps 1000 \
  --render-target source --blocks-per-batch 32 --render-backward replay \
  --training-data-device cuda --condition-snr --snr-range 0 20 \
  --channel awgn --resolution 2 --device cuda --save-every 100
```

codec 渲染优化与联合优化不是必须都串行执行；需要直接联合学习时，
`--codec-init` 可直接指向新预训练的 `$OUT/codec.pt`。
`--init` 仅加载模型，不恢复 Adam 或步数状态。上述步数都可以按实际训练预算调整。

## 输出与软件验证

`loss.jsonl` 增加 geometry/shape/opacity/DC/SH 分项（未乘各项权重），
渲染阶段还记录总渲染误差、辅助误差和 replay 耗时。
`training.json` 保存实际 codec 配置及渲染目标。

```bash
python -m gaussian_jscc plot-stats --training "$OUT" --out "${OUT}_charts"
```

可查看 `training_objectives.png` 和 `training_physical_losses.png`。
物理损失量纲与旧平均 loss 不同，不应把新旧总 loss 数值直接比较。

本地 CPU 软件检查覆盖符号预算、独立解码、位置对属性 Context 的独立性、
物理损失、旧包兼容，以及带 Gumbel/信道噪声的联合 replay 与完整反向传播一致性。
它们不是新的画质消融。真实 CUDA 光栅器、恢复画质、显存与训练时间仍需服务器验证，
不能沿用旧架构的 2.27 秒/步作为新架构实测。
