# Codec 整体损失修正：balanced_v2

这次修正针对已经观察到的“位置误差改善，但属性和最终渲染仍严重失真”。
不增加 JSCC payload 的符号预算、不额外发送粗 XYZ、不把接收端 Context 改回位置解码器。
新增 loss profile 配置可能轻微改变已有元数据的字节数，仍按原记账规则计费。
目前是代码与数值回归验证，不是已经完成 Truck 场景的画质验证。

## 修正依据与边界

旧 `physical_v1` 的白化协方差损失存在明确的数学缺陷：忽略稳定性 floor，
当预测轴尺度统一为源尺度的 c 倍、旋转准确时，

```text
D_shape(c) = log(1 + (c² - 1)² / 3)
```

c 趋向 0 时损失只趋向 `log(4/3)`，对 log(c) 的梯度趋向 0。
因此它不适合作为唯一尺度约束。这个缺陷可以由公式和数值测试确认，
但之前的无符号 RMSE 本身不能证明已训练模型一定发生了缩小。

现在使用以下共同重建目标，贯穿 codec 预训练和完整场景微调：

```text
L_rec = D_position + 0.25 D_logcov + D_logscale
        + D_opacity + D_DC + 0.25 D_SH

D_logcov = ||R̂ diag(2 log ŝ) R̂ᵀ - R diag(2 log s) Rᵀ||²_F / 9
D_logscale = mean SmoothL1(sort(log ŝ), sort(log s))
D_opacity = SmoothL1(sigmoid(ô), sigmoid(o)) + 0.1 SmoothL1(ô, o)
```

位置项保留源局部尺度归一化；DC/SH 保留冻结归一化统计后的分组 Smooth L1。
logcov 使用解析公式，不增加逐点矩阵求逆或特征分解，保留四元数符号、
等价轴排列、同一坐标系旋转下的形状一致性。排序尺度损失不是强制参数主轴的编号一致。
渲染端的安全 clamp 保留，辅助尺度/logit 约束读取 clamp 之前的输出，以保留越界修复梯度。

这些目标及权重是针对当前实现缺陷设计的工程选择，不是照抄某篇论文的验证配方。
固定权重也不保证各分支梯度平衡，更不保证码率或 SNR 提高时每张图都更好。
logcov 的平方损失仍可能对离群尺度产生大梯度，现有非有限值检查和梯度裁剪继续保留。

## 两个训练阶段现在如何衔接

- 属性/几何恢复阶段：`L = L_rec`，随机局部块、随机档位和 SNR。
- 渲染阶段：`L = D_render + λ_rec L_rec`，完整场景、随机训练相机和 SNR。
  `D_render = 0.8 L1 + 0.2 (1-SSIM)`，默认参考冻结源 PLY 渲染。
  `λ_rec` 默认从旧的 0.1 改为 1，避免切换阶段时再把属性约束整体削弱十倍。
- `train` 支持 `--render-lr`，默认 `0.25 × --lr`；脚本显式用 `5e-5 → 1e-5`。
  同一进程阶段切换不重置 Adam；`--init` 只加载模型/归一化统计，不恢复旧 Adam/步数。
- `train-route2` 使用同一重建目标及默认 `λ_rec=1`，再加原有期望符号成本项。
  本次没有训练四档分配器，也没有把均匀档位 codec 评估称为自适应资源分配。

两个阶段采样范围和完整目标不同，因此不能把总 loss 的跳变解释为画质跃升。
损失函数修复不等于可以自动判定训练结束：本次仍是显式迭代预算，未增加验证集早停。

## 现有 checkpoint 可以继续使用

- version 2 / legacy：仍能解码、评估，不支持初始化 geometry-first。
- version 3 / geometry_first / 原 physical_v1：直接加载评估仍保持旧配置和模型哈希；
  训练默认显式迁移到 balanced_v2，打印提示，初始化网络权重和归一化统计不变。
- profile 切换会采用新 profile 权重，再应用明确传入的 `--*-weight`。
  同 profile 继续训练则保留 checkpoint 权重，除非显式覆盖。
- 若要复现旧训练目标，传 `--loss-profile physical_v1 --attr-weight 0.1`
  以及 `--render-lr` 等于原 `--lr`。不要用新 loss 标度直接比较旧 loss 数字。
- 微调后的模型身份会改变，发送和接收必须使用同一个新 checkpoint；
  不要混用旧 `route2.pt` 或旧通信包。已生成的旧包仍应由原模型解码。

## 服务器继续训练

已有环境无需重装，先进入项目并激活 `maskgs`。下面使用当前已完成的 checkpoint，
3000 步修复 + 1000 步渲染只是本轮预算，不是收敛保证。脚本保留 32 blocks/replay。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs

INIT="$PWD/output/truck_geometry_first_20260915_095729/codec.pt"
OUT="$PWD/output/truck_codec_balanced_$(date +%Y%m%d_%H%M%S)"
LOG="${OUT}.log"
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_codec_balanced.sh "$INIT" "$OUT" > "$LOG" 2>&1 &
TRAIN_PID=$!
printf 'PID=%s\nOUT=%s\nLOG=%s\n' "$TRAIN_PID" "$OUT" "$LOG"
tail -f "$LOG"
```

`Ctrl+C` 只退出日志跟踪，不停止 nohup 训练。目录已存在时脚本拒绝覆盖。
若需要改变步数，可在运行脚本前 `export REPAIR_STEPS=... RENDER_STEPS=...`。
不在活跃服务器上自动启动这项耗时任务，GPU 使用仍由操作者确认。

训练结束后，使用同一 PLY、相机划分、分辨率和码率表做与旧结果一致的评估：

```bash
EVAL="${OUT}_eval"
CUDA_VISIBLE_DEVICES=0 python -u benchmark_codec.py \
  --ply "$PWD/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply" \
  --checkpoint "$OUT/codec.pt" --source "$PWD/data/tandt_db/tandt/truck" \
  --out "$EVAL" --tiers 1 2 3 --snrs 0 5 10 15 20 \
  --channels awgn --trials 3 --resolution 2 --device cuda --save-images --lpips
```

至少同时观察接收渲染相对源渲染的误差、相对照片的 PSNR/SSIM/LPIPS、属性误差和位置误差，
不要只依据训练 loss 判定成功。不同模型归一化损失不一定可比，物理误差和同口径渲染指标更合适。

## 输出图与统计

代码保留 PNG/SVG 静态输出，图表契约如下：

| 文件 | 回答的问题 | 表达方式 |
|---|---|---|
| `training_objectives` | 各阶段内部是否改善？ | 按连续 phase/profile 分面，总目标与实际加权辅助项，梯度为裁剪前值 |
| `training_physical_losses` | 哪些恢复分项变化？ | 六个独立面板，不把不同量纲放在一条共用标尺上 |
| `training_weighted_contributions` | 每项给总目标贡献多少？ | 分阶段曲线；不把标量贡献解读为梯度贡献 |
| `codec_parameter_errors_vs_snr` | 传输后的各参数误差如何？ | 保留原指标，加排序尺度、对数协方差 RMSE、有符号 log-volume bias |

平滑不跨阶段或 loss profile 边界；缺失历史分项不编造。只有一个记录时用点表示。
多条曲线只在同类趋势问题中使用，颜色配合线型区分；PNG/SVG 需验证图例与标题无遮挡。
`training_chart_data.csv` 现在导出所有标量字段，包括各分项、实际贡献、学习率。

`log_volume_bias = mean(sum(log ŝ - log s))`，负值表示几何平均体积比偏小，正值表示偏大；
不是逐 Gaussian 体积都同方向变化的证明。对数协方差指标比单独四元数角度更能识别等价形状。

旧日志可用 `python -m gaussian_jscc plot-stats --training OLD_TRAIN_DIR --out NEW_CHART_DIR`
重新分阶段绘图，但旧日志没有的 `scale_loss`、加权贡献及新评估指标不会自动补出。

## 本地验证范围

- 48 项单元/集成测试中 45 项通过；3 项需要 CUDA 光栅器或 CUDA RNG 的测试跳过。
- 验证尺度扩张/收缩对称性、越过渲染 clamp 后的辅助梯度、等价轴交换、旧模型哈希和通信包兼容。
- 验证从旧 geometry-first 权重切换目标、两阶段学习率变化、分项贡献之和，及 replay/普通反向传播一致性。
- 已检查 PNG 排版及 CSV 导出。用于排版的合成记录不作为真实训练效果证据。
- Bash 启动脚本通过语法检查。完整 Truck 的 GPU 训练时间和修正后画质尚需服务器运行验证。
