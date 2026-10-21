# 块归一化续训与联合梯度定位

本次是故障定位实验，不是稳定性修复方案。保留网络、原有损失、学习率和Adam；不裁剪梯度、不回退更新、不使用已经退化的joint/latest。

## 默认运行流程

1. 从原六组实验的`block_awgn10/checkpoints/final`续训4000步信道恢复：LR=1e-4，语义两端冻结，继承Adam，原始y MSE。默认累计到8000步。复用上一轮的训练批次/档位/噪声调度。
2. 和已有逐点8000步端点做公平比较：同一实际test相机名单（Truck为32个）、4次配对10dB噪声，记录对PLY和照片的PSNR/SSIM/LPIPS、无噪声与latent NMSE，以及实际平均功率。与原4视角validation分开。
3. 逐点、块各自从自己的8000步信道端点，分别启动`plain`和`instrumented`两个独立短探针，每个默认16步。不是在plain结束后接着跑instrumented。覆盖此前第4、9步的异常。
4. 四条短探针都保持原联合目标：图像MSE + (1-SSIM)/3 + 原始y MSE；信道两端LR=1e-4，语义解码器LR=1e-5，语义编码器冻结。保留旧信道Adam，语义解码器Adam新建；q1/q2/q3/mixed轮换、相机及噪声与原入口一致。

`plain`原样反向，记录真实更新和更新后的复测。`instrumented`额外进行只读的`autograd.grad`；实际参数更新仍走原来的组合反向，诊断不写入`.grad`。CPU测试核对组合梯度、实际Adam更新与原路径一致。CUDA原子累加本身仍可能有微小不确定性，不保证两次训练逐bit相等。

## 检查记录

每个探针的`steps.jsonl`包含：

- `before`、`after`：同一档位、同一相机、同一噪声下，更新前后的图像MSE、SSIM损失、latent MSE和总损失。不能用相邻随机视角的loss变化代替这个检查。
- `updates`：各模块实际参数更新范数和相对更新量；`gradient_groups`为未裁剪的真实梯度。
- `decomposition.parameter_norms`：图像MSE、已经乘1/3的SSIM、latent三项分别在信道编码器、信道解码器、语义解码器上的完整场景梯度范数。
- `decomposition.parameter_cosine`：完整场景分目标梯度夹角余弦；是先跨批累加再比较，不是把局部范数相加。
- `gradient_sum_relative_error`：三项诊断梯度之和与实际组合梯度的相对偏差，用来发现诊断改变了语义或CUDA反向数值不稳定。不是自动拒绝更新的阈值。
- `raw_render_gradient`与`raw_component_gradients`：光栅化输入处，XYZ、不透明度、log大小、旋转、DC、SH的梯度范数和最大绝对值；后者分别检查MSE、SSIM。
- 每项最大的16个Gaussian：原始source ID、位置、大小、长短轴比、不透明度、旋转、各属性梯度；组合梯度的异常点还记录当前相机空间深度。
- 语义解码器属性输出、尚未做反标准化/旋转归一化的梯度；旋转四元数归一化前最小模长和低于1e-4的数量。这是诊断计数，不是训练约束。
- `image_gradient_norms`：图像MSE与加权SSIM对渲染图像的梯度范数，用于区分图像目标和Gaussian投影反向的放大。

所有raw属性梯度是在固定XYZ也视为光栅化输入变量时测得的；XYZ梯度大不代表本实验更新了坐标。实际传输坐标路径仍为16bit可靠交付。

诊断需要额外反向和同输入前向，因此比正常训练慢；仍采用完整场景replay，逐相机释放光栅化图，不改为保留整个编解码器场景图。额外显存包括两份完整场景FP32分目标梯度、瞬时FP64统计、三个参数梯度向量和权重快照；是有界的额外开销，不保证等于原训练峰值。只保存短探针initial/latest检查点、结构化日志和比较图，不保存逐轮大图像。

若出现非有限梯度，记录错误并停止该探针，不污染另一个起点；若只是很大但有限，则按原机制运行最多16步，以便复现。若更新后渲染失败，保留本步更新前的诊断记录并结束该探针。

## 输出和恢复

```text
NEW_OUT/
  plan.json, status.json, complete.json（仅全部完成时）
  block_continuation.log
  block_continuation/block.log
  block_continuation/block/（续训、验证与端点检查点）
  point_plain.log, point_instrumented.log, block_plain.log, block_instrumented.log
  audit/{point,block}/{plain,instrumented}/
    control.json, steps.jsonl, complete.json, error.json（若报错）
    checkpoints/{initial,latest}/codec.pt
  comparison/
    channel_endpoints.csv, channel_endpoints.json
    joint_probes.csv, joint_probes.json, joint_probes.png
```

每条探针的`complete.json`表示该探针已结束，必须看其`status`；`probe_error`不表示训练成功。顶层`status.json`会汇总部分失败。某条探针失败不阻止其他独立探针。根目录`complete.json`仅在块续训及四条探针均成功执行完预算时写入；不表示画质稳定或梯度没有爆炸。

续跑加`--resume-suite`：块信道续训使用原检查点精确恢复；成功执行完预算的探针跳过，**失败或被中断的短探针从8000步起点重跑**，覆盖其自有诊断日志，不是接着可能退化的latest训练。旧实验目录完全不写入。服务器上的四条探针各用独立进程，即使某条遇到CUDA上下文错误，也不阻止下一条初始化；比较表生成失败也不会阻止探针。

随机种子修复仅将NumPy的seed映射到uint32；Python和PyTorch保留完整seed，已有uint32内的序列不变。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

POWER="$PWD/output/truck_power_controls_20261020_184354"
FOLLOWUP="$PWD/output/truck_power_followup_20261020_202814"
OUT="$PWD/output/truck_power_stability_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_power_stability_truck.sh \
  "$POWER" "$FOLLOWUP" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -F "$OUT/block_continuation/block.log"
```

GPU3需换为当时空闲卡。之后看`$OUT/point_instrumented.log`或`$OUT/block_instrumented.log`，对应的plain日志用于确认原样路径。三项工作在同一个输出目录中、同一GPU依次运行，避免显存竞争。可在脚本末尾添加`--dry-run`只检查源检查点和计划，不读取PLY、不创建输出；`--manifest`可指定迁移路径，但PLY指纹及相机划分必须一致。
