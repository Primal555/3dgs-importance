# 中心预训练、属性适配、联合渲染微调

入口：`python -m gaussian_jscc train-center-attributes`，启动脚本：
`scripts/test_center_attribute_codec.sh`。这是独立的 **clean 表示重建实验**，
不覆盖原来的 `train-representation`、teacher-axis 或 12 bit 对照。
随机初始化，不从之前失败的实验挑初始化权重。

## 结构与信息边界

```text
发送端中心 XYZ ──独立中心编码器── z_xyz ──独立中心解码器──预测中心
                                                            │
发送端其他属性──独立属性编码器── z_attr ──独立属性解码器 ◀─────┘
                                               │
                                 预测中心＋恢复属性 → 渲染
```

- 中心编码器只读取 XYZ 及从 XYZ 建立的邻域关系。不读取 logcov、方向、
  大小、扁平度、颜色、不透明度或 SH。中心解码器只读取中心 latent。
- 两条路径的编码器、Transformer、输出层完全独立，不只是不同输出头。
- 两个编码器都保留逐点自身路径和发送端多尺度几何 Context。属性编码器
  可以用发送端 XYZ 建立邻域，但其属性输入不再与中心 latent 混合。
- 接收侧属性解码器以**预测中心**作为条件，不接收原始坐标。联合阶段
  这个条件不 detach，属性与渲染梯度都可以通过它影响中心路径。
- 形状继续用 logcov 表示，独立输出不透明度、DC、SH。没有轴尺度加权
  位置误差，没有手工粗坐标/定位 token，没有 12 bit 坐标旁路。
- 默认 latent 总宽度 64，中心 32、属性 32 个实数；通过 `LATENT_DIM` 和
  `CENTER_LATENT_DIM` 调整。这是实验容量分配，不是通信符号数，也不声称
  中心 latent 本身比原始三个坐标更省比特。
- 不训练或模拟 JSCC 适配器、信道噪声、功率约束和档位。对这个检查点
  调用通信编码/解码会明确报错，而不是把 clean 结果冒充通信恢复。

## 三个阶段

| 阶段 | 更新参数 | 目标 | 中心来源 |
| --- | --- | --- | --- |
| A center | 中心编码器、中心解码器 | 世界坐标中的中心距离 | 网络恢复 |
| B attribute | 属性编码器、属性解码器 | 小批量 logcov 形状与局部颜色响应重建 | 冻结的位置路径实际预测 |
| C joint | 四个模块全部解冻 | 原始 PLY 渲染图与恢复图的 MSE | 可被渲染梯度微调的预测中心 |

A 的唯一损失是：

```text
delta = XYZ_pred_world - XYZ_source_world
L_center = mean(sqrt(sum(delta**2) + tau**2) - tau)
```

`tau` 默认为 `0.001` 场景世界单位，是公开的平滑超参数，可通过
`CENTER_SMOOTHING` 修改。它不是原始椭球厚度，误差不再除以轴尺度或
场景对角线。单点对世界中心的梯度范数不超过 1（求平均前），但归一化
坐标转换和网络 Jacobian 仍可能放大**参数梯度**，不承诺杜绝梯度爆炸。

B 阶段只在拟合块上随机取小批量，**不调用全场景光栅器，也不 replay**：

```text
L_shape = mean(||log(Sigma_pred) - log(Sigma_source)||_F^2 / 9)
L_response = mean((isolated_RGB_pred - isolated_RGB_source)^2)
L_B = (L_shape + L_response) / 2
```

形状项使用真实物理 logcov 矩阵，不是标准化特征的 MSE；矩阵的非对角项
按完整 3×3 矩阵计数。响应项复用已有 `centered_response_loss`：每步随机
采样 4 个观察方向，比较单椭球在黑/白背景、原始及预测 footprint 探针
处的颜色响应，监督不透明度、DC、SH 和形状。`ATTRIBUTE_VIEWS` 可调整方向数。
响应计算为局部解析计算，不涉及相机、遮挡排序或整场景渲染。

两项的等权平均是**公开的工程选择，不是自动梯度平衡，也不是论文最优权重**。
日志同时记录未加权分量、加权贡献，以及 profile 步各分量对属性模块的
梯度范数，便于发现形状项是否压制颜色学习。

局部损失内部把两椭球中心移到同一原点，仅用于属性比较；这不是把原始
XYZ 交给解码器。属性解码器始终看到冻结路径实际预测的 XYZ。B 没有位置
监督、不改变中心参数；局部误差下降也不保证场景 PSNR 同步上升。

C 才使用全场景、每步随机训练相机、默认 replay 分批重算。目标只有输入
PLY 渲染图的 MSE，无属性辅助加权项，无原始中心替换。照片只作验证指标。

默认所有分支学习率都固定 `2e-4`，无自动衰减、无梯度裁剪。联合阶段
位置学习率可单独设置 `JOINT_CENTER_LR`，例如 `0.00002`，但没有偷偷
降低默认值。每次切换阶段采用新的 Adam；精确恢复则保留当前 Adam 状态。

## 阶段切换与回退

步数是预算上限，不是宣称必须训练这么久：默认 A/B/C 上限 5000/1000/1000。

1. A 与固定视角的“12 bit 中心＋原始属性”对照比较。至少 500 步后，
   “预测中心＋原始属性”的平均 source PSNR 落后不超过 3 dB，可提前进入 B。
   达到上限仍不满足时保存结果并停止，不拿明显不合格的位置自动训练属性。
2. B 至少 200 步；固定留出块与固定方向的**属性重建损失**相对 B 初始值
   至少改善 5%，且连续 3 次验证未
   相对显著改善，可提前进入 C。显著改善默认指相对于上次显著改善锚点的
   0.5%，不是每次极小改善都重置计数。达到上限且达到 5% 改善也可进入 C；
   未达到则停止。模型选择和这一门槛不使用渲染 MSE。
3. C 至少 200 步；连续 5 次验证没有显著改善可提前结束，也受步数上限限制。
   C 开始前的属性阶段最优模型也参与 C 的选择，所以联合微调退步时不必
   导出退步的最后一次权重。

`MIN_*_STEPS` 控制提前切换/停止；如果主动把预算上限设得更短，仍会在
上限处检查质量门槛，而不是额外强行训练到最小步数。

这些数值全是**可调工程门槛，不是论文结论或最优设置**。A 的 3 dB 门槛
很可能严格，尤其 12 bit 对照接近原始渲染时。首次结果应检查实际差距，
不能为了跑完三阶段而直接把门槛放宽到失去意义。

阶段切换使用当前阶段的固定验证最优权重，再开始下一阶段。
阶段 C 只按完整恢复 MSE 选择；位置诊断略降不自动判失败，因为形状与
位置可能共同补偿。若完整图像也变差，则保留更好的联合初始/中间权重。

默认每 100 步验证，**第 0 步、每 500 个全局步和阶段预算末尾保存图像**。
B 的常规验证仅计算固定块属性损失和中心误差，整场景 PSNR/SSIM 仅在
图像保存时及 B 入口计算；因此提前切换时不保证有该步的 PSNR/PNG。
验证相机与训练相机来自数据集不同 split，并检查名称不重叠。
A/B 保留未参与小批量训练的块；C 渲染训练使用整个场景，这些块在 C
不再是未见 Gaussian，只有验证相机仍未用于训练。不要将它解释为跨场景泛化。

## 服务器后台运行

先确认所选 GPU 空闲；下面的 `2` 仅为示例。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

OUT="$PWD/output/truck_center_attribute_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
env -u INIT CUDA_VISIBLE_DEVICES=2 \
  CENTER_STEPS=5000 ATTRIBUTE_STEPS=1000 JOINT_STEPS=1000 \
  nohup bash scripts/test_center_attribute_codec.sh "$OUT" \
  > "$OUT/console.log" 2>&1 &
echo $! > "$OUT/run.pid"
tail -f "$OUT/console.log"
```

`Ctrl+C` 退出 tail 不会停止后台训练。要先检查较短流程，可设置
`CENTER_STEPS=1000 ATTRIBUTE_STEPS=300 JOINT_STEPS=300`，但位置未通过
门槛时仍会停在 A。如果只研究中心，设置 `ATTRIBUTE_STEPS=0 JOINT_STEPS=0`。
显存不足可将 `RENDER_BLOCKS_PER_BATCH=32`；这只改变批处理，不改变损失。

中断后精确恢复，保存的配置优先于命令行其他参数：

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m gaussian_jscc train-center-attributes \
  --resume "$OUT/training_state.pt"
```

门槛停止/已完成的状态再次 resume 不会暗中放宽门槛或增加预算。
本次恢复状态格式为 `center_attribute_training_v2`。旧 v1 的 B 是渲染 MSE，
不能用新代码静默续训；旧实验须在提交 `fdbbc35` 上恢复。本次使用新目录随机开始。

## 一个目录内的结果

- `training.json`：结构、分支 latent 宽度、损失、学习率、门槛、相机与块划分。
- `loss.jsonl`：逐步 loss、四个模块真实梯度范数、每 10 步参数更新量、
  各分支学习率；B 的 `terms`、`stats` 和 `objective_module_grad_norms` 分解
  两项贡献；C 包含渲染与 replay 耗时诊断。
- `validation.jsonl`：中心 world RMSE、距离分位数；完整恢复、位置诊断、
  12 bit 对照分别相对原始渲染/照片的 PSNR、SSIM、MSE、L1。
  `attributes` 保存固定方向属性验证；B 非渲染验证时 `render` 为 null。
- `images/000500/view_00/`：`full.png`、`center_only.png`、`quantized12.png`、
  `source.png`、`photo.png` 和标注对比图。后两种混合恢复是诊断，不是部署效果。
- `charts/training_by_phase.png`：阶段目标使用独立坐标轴，不把中心距离与
  图像 MSE 的数值断点当成学习改善。`charts/validation.png` 显示定位与画质。
  `charts/attribute_validation.png` 显示固定块属性损失及未加权分量。
- `transitions.jsonl`：切换/停止原因、门槛结果、选用的权重步数。
- `codec_500.pt` 等：该步真正训练出的权重，不会在回退时替换成更早权重。
- `codec_best_center.pt`、`codec_best_attribute.pt`、`codec_best_joint.pt`：各阶段验证最优。
- `codec_center.pt` 等阶段导出，以及最终 `codec.pt`：选中的最优权重；
  `summary.json/export_selected_step` 给出实际权重步数。
- `codec_last.pt`：最后一个实际优化步骤；`training_state.pt`：精确恢复状态。
  阶段边界恢复状态包含已选回的权重和下一阶段位置，新阶段使用新 Adam。

原始属性只出现在监督/诊断里，不能把 `center_only.png` 当成完整模型恢复图。

## 已执行的本机检查

- 数学与结构测试覆盖中心不读取属性、独立模块更新、属性阶段位置严格冻结、
  联合阶段渲染/属性条件能向位置反传、直接反传与 replay 梯度一致、padding、
  检查点往返、三阶段门槛、以及在 A/B/C 中断后精确恢复。
- B 使用真实局部属性损失；C 集成测试使用**可微模拟渲染器**验证流程，
  不代表真实 CUDA 画质。测试额外检查 B 不调用光栅器/replay、XYZ 不影响
  局部目标、监督目标不接收梯度、固定验证不消耗训练随机数、旧状态拒绝恢复。
- 真实 Truck PLY 抽取 8192 个点，hidden48，中心单独训练 500 步：留出块
  world XYZ RMSE 从 **18.3256 降至 13.7569**。梯度范数范围 **25.67～435.65**，
  全部有限；属性模块梯度始终为零。此结果仍远非精确定位，也不是与旧架构
  同初值同参数量的受控优劣比较。
- 本机没有 CUDA 光栅器，尚未验证真实全场景 PSNR 或三阶段训练速度。

本次 B 改动另跑了真实 Truck 8192 点、hidden48、A 300+B 300 步 CPU 检查。
为了测试 B 路径，**显式放宽无相机诊断门槛到 1000000**；不是通过了
12 bit 渲染门槛。A 留出中心 RMSE 为 16.7154，B 全程保持不变。
B 留出属性损失 **8.64124 → 3.53540**，其中 logcov MSE
**17.20343 → 6.99879**，局部响应 MSE **0.07905 → 0.07201**。
中心模块梯度/更新为零，B 总梯度范数 **3.53～90.43**，未启用裁剪且全部有限。
CPU B 步平均约 0.235 秒（含 profile 步，不含验证/保存），不能外推为服务器速度。
形状项仍主导损失和梯度，响应改善有限；这只是路径与可优化性检查，
**不能用总 loss 的下降宣称颜色或最终画质已经恢复良好**。

本机复现：

```bash
python scripts/check_center_attribute_local.py \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --out output/center_attribute_cpu_trial --steps 500
```

仅作本机 A+B 路径检查时可以加 `--attribute-steps 300` 和显式
`--center-max-world-rmse <诊断阈值>`，并保持无 `--source`。此门槛只用于
无相机 CPU 诊断，不等价于通过渲染质量门槛，不能宣称中心已经合格。
真实服务器脚本仍默认使用固定相机的 12 bit 渲染对照门槛。
