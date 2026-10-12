# Truck 残差前缀：逐层冻结对照

检验的问题是：高档继续更新共享网络和低档路径，究竟是在帮助协同恢复，还是在牺牲已经学好的低档前缀？本实验不预先假定冻结更好，也不保证 q3 的画质必然优于 q2。

## 两组只改变更新范围

两组从**同一残差前缀 checkpoint** 初始化，都依次训练 q1 → q2 → q3，每阶段默认 **1000 次实际参数更新**，每组 3000 次，合计 6000 次。学习率固定 `1e-5`。之前的独立单步测试表明 `1e-4` 容易造成画质大幅下降，因此这次使用较小学习率作为受控起点，并非宣称最优。

| 阶段 | joint 对照组 | frozen 实验组 |
|---|---|---|
| q1（4 复符号） | 完整属性编解码器允许更新，未参与前向的增强层无梯度 | 共享骨干、基础属性头、首 4 符号路径 |
| q2（12 复符号） | q2 渲染 MSE 向共享网络和前两层反传 | 仅新增 8 符号编码行、对应解码输入列、q2 条件嵌入及第二层修正模块 |
| q3（24 复符号） | q3 渲染 MSE 向共享网络及三层反传 | 仅新增 12 符号编码行、对应解码输入列、q3 条件嵌入及第三层修正模块 |

可靠 16bit XYZ 在两组中都保持不变。损失仍是**完整场景渲染与原始 PLY 渲染的图像 MSE**，不是照片监督；不引入属性辅助损失、通信惩罚、分配器、剪点、梯度裁剪或步长拒绝。

相同顺序、实际更新数、初始权重、训练相机、完整符号槽 AWGN 随机数、验证相机及验证噪声用于两组。固定沿用源实验的 SNR、信道、空间分组及输入归一化。每个阶段开始，两组都**重新创建 Adam**，不继承源实验或上一阶段的动量。这是从旧权重开始的对照微调，不是精确续训。

## 冻结不仅是 detach

共享编码器一旦变化，前 4/12 个符号也会变化；所以只冻结低档解码头不足以保护低档。本实现同时冻结共享编码骨干、共享解码骨干及已有符号路径。

`symbol_head` 的输出按每个复符号两个实数排列。新增层只开放对应输出行。`dec_in` 输入包含接收值和存在掩码；新增层只开放相应两段输入列。共享张量中的其他元素通过梯度掩码冻结；每阶段 Adam 从零动量开始，且无权重衰减，避免被历史动量偷偷更新。每次参数更新后，所有冻结参数与 buffer 都与阶段起点逐元素检查，不一致即停止并报错，**不静默修补**。

验证时额外检查固定 4 个局部块：q2 阶段与起点比较 q1；q3 阶段比较 q1/q2。相同输入、相同噪声下记录发送前缀和恢复属性的最大变化。frozen 组容忍度为 `1e-6`；joint 组只记录漂移不拦截。功能抽检不是全场景逐点证明；完整参数冻结审计覆盖所有冻结元素。

已有结构将各增强层独立功率归一化，因此修改新增符号不会重新缩放低档符号。模型结构和 checkpoint schema 不变，此脚本独立于正式训练流程。

## 服务器启动

先激活环境、拉取代码，再确认指定显卡有足够空余。以下用物理 GPU 2；按实际空余卡修改 `CUDA_VISIBLE_DEVICES`。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_residual_prefix_gpu2_20261012_095255"
OUT="$PWD/output/truck_residual_freeze_gpu2_$(date +%Y%m%d_%H%M%S)"

# 建议先检查路径、源实验配置；此步骤不会训练或创建输出目录。
CUDA_VISIBLE_DEVICES=2 bash scripts/test_residual_freeze_controls.sh "$RUN" "$OUT" --dry-run

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_residual_freeze_controls.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

所有结果位于同一个 `OUT` 内，原 checkpoint 不覆盖。`RUN` 既可指向成对实验根目录，也可指向其 `residual` 子目录。默认加载 `residual/checkpoints/final/codec.pt`。

可选环境变量：`STAGE_STEPS="1000 1000 1000"`、`LR=1e-5`、`VALIDATE_EVERY=100`、`VIEWS_PER_STEP=2`、`HELDOUT_VIEWS=4`、`VALIDATION_TRIALS=2`、`IMAGE_VIEWS=1`、`AUDIT_BLOCKS=4`、`BLOCKS_PER_BATCH=64`。默认 batch 配置继承原实验。迁移数据路径时用 `MANIFEST=/absolute/path/manifest.json`；场景指纹及训练/留出相机划分必须与源实验一致。

脚本会拒绝覆盖输出或在源实验目录内部保存结果。缺少 CUDA 显卡变量、源文件或场景不匹配会明确报错。`--dry-run` 仅检查元数据与文件路径；实际启动时还会检查模型结构、场景身份和相机划分。

## 结果阅读

- `charts/loss_comparison.png`：两组相同更新横轴的原始 MSE，以及阶段内 20 步尾随平均。竖线标出阶段切换，平滑不跨阶段。
- `charts/validation_comparison.png`：q1/q2/q3 的 PSNR、SSIM；每 100 步及阶段结束都验证三档，默认 4 个留出相机 × 2 次固定噪声。
- `joint/`、`frozen/`：逐步 `loss.jsonl`、`validation.jsonl`、`freeze_audits.jsonl`、逐阶段 trainable 范围，以及 q1/q2/q3 各自的 codec 和 Adam 文件。
- `images/initial/`：公共初始图；`images/joint/final/`、`images/frozen/final/`：两组最终三档图。只保存初始和最终，不保存中间图。面板依次为 source PLY、received、AbsError×4；默认每档一个相机、第一噪声试次。
- `sampling_plan.jsonl`：两组共用的训练相机/噪声计划；`plan.json`：源文件 hash、源模型 ID 和配置；`summary.json`：初始/最终指标及更新数；只有全部训练和源文件检查成功后才写 `complete.json`。

主要看 q2/q3 训练期间低档是否保住，以及 q3 在留出相机上是否真正超过 q2。冻结组虽然保护了低档，但也缩小了高档的可训练容量，可能使 q3 的优化收益更小；这正是对照要衡量的取舍。不能只比较当前训练档位的 loss，也不能把“冻结成功”当作“恢复质量提升”。

阶段 checkpoint 保存更新数从本次实验起算；另存的 Adam 文件便于检查动量，但此独立脚本**没有实现中断精确恢复**。它默认重新从指定源权重开始，两组各阶段使用新 Adam。

## 前缀审计索引兼容性修复

旧审计使用 `signal[valid, :prefix_width]` 同时进行二维布尔索引和切片。在 PyTorch 2.4.1 上，切片先落在点维度，256 点块进入 q2 的 q1 审计时会报掩码形状不匹配；本机 PyTorch 2.9 没有这一行为，旧 8 点测试也未覆盖真实块大小。现改为先用 `signal[valid]` 获取有效点，再对结果的符号维度切片。新增 256 点、多块、尾部填充测试，覆盖三档审计及两组跨阶段更新。无需升级服务器 PyTorch。

此错误发生在 q1 阶段完成、进入 q2 的审计准备环节，不会改变之前 q1 的训练更新。修复仅涉及审计索引，不改变学习率、损失、冻结范围或阶段步数；也不能据此认为 q1 的收敛瓶颈已解决。重新启动请使用新 OUT，保留失败运行用于比较；本入口不支持断点精确续训。

判断 q1 是否在 300 步后停滞，应查看该次运行 `joint/validation.jsonl` 中训练阶段为 q1 的固定相机/噪声指标，并结合 `joint/loss.jsonl` 的梯度、更新范数。此实验从已经训练过的残差 checkpoint 开始，不是随机冷启动。进入 frozen 组 q2/q3 后，q1 保持不变则是冻结设计的预期结果，不等于 q1 训练未执行。未取得该次运行日志前，不以早期视觉印象直接改变两个对照组的训练预算。
