# 信道功率归一化对照：同一语义教师，六组训练

入口：`scripts/test_channel_normalization_truck.sh TEACHER_RUN NEW_OUT [extra options]`。
这是独立实验，不覆盖历史训练入口。输入要求：含 `training.json` 和完整模块化训练检查点的 Truck 目录；预算必须为 8/24/48，信道解码器为 progressive。默认读取 `checkpoints/final/codec.pt`，可用 `--checkpoint` 指定其他兼容检查点。

## 比较对象

| 目录名 | 发端功率处理 | 训练信道 |
|---|---|---|
| point_noiseless | 旧的逐点逐段能量归一化，floor=.01 | 无噪声 |
| point_awgn10 | 同上 | 10 dB AWGN |
| statistical_noiseless | 每段公共能量统计；保留逐点幅度差异 | 无噪声 |
| statistical_awgn10 | 同上 | 10 dB AWGN |
| block_noiseless | 每个原始空间块逐段共用尺度 | 无噪声 |
| block_awgn10 | 同上 | 10 dB AWGN |

六组只复制同一份语义编码器、语义解码器和属性标准化参数；信道两端及其已有补偿模块均重新随机初始化。对应参数的随机初值相同，Adam 全部重新建立。源检查点的信道权重、动量和训练步数都不继承。原有 full-prefix、逐点补偿、空间补偿等结构按源配置保留，避免同时改变结构。

默认每组 4000 次真正的优化器更新，LR=1e-4，q1/q2/q3 均衡轮换。它们是用于本轮定位的工程预算，不是论文最优值。优化原始教师 y 的 MSE，不改变 y 的尺度，不使用渲染或颜色损失，不训练分配器。无噪声组也使用相同的 10 dB 网络条件输入，差别只有物理噪声。

## 功率和尺度协议

每个复符号为相邻两个实数槽，能量为二者平方和。目标平均复符号功率 P=1；10 dB AWGN 每个实数分量方差固定为 .05，不根据点、块或当前发送幅度重设噪声。

- point：x=z/sqrt(E_point+.01)，保持原有前向和历史检查点格式。
- statistical：x=z/sqrt(E_running+.01)。每段一个统计量；训练 forward 用当前有效点原始能量更新 detached EMA，默认 momentum=.01。评估时冻结统计量。
- block：x=z/sqrt(E_block+.01)。每行始终为原始 Morton 空间块，默认256点；沿点轴计算，不沿 batch 轴计算。填充点不参与。改变每批块数不改变一个块的输出。

统计组在第一次训练前，以及每1000步验证前，用**全部场景点的冻结教师 y**和当前信道编码器重新计算原始发送能量，写入公共统计量。校准是明确记录并保存到检查点的发端状态更新，随后继续训练；不是额外梯度更新。中途校准可能造成曲线变化，应结合 `population_energy_before/after` 判断。接收端直接恢复 y，不收到每个点/块的原始能量或逆缩放系数。

每次验证计算全场景逐段和累计前缀的平均功率、点能量分布、符号 PAPR及有效SNR。大于单位功率2%的结果标记超预算；同档位各完成组平均功率相差超过5%时，横向质量比较标记为不满足功率匹配。这些容差仅为报告判据。低功率组不会被自动补功率，超预算组不会被静默修正；不能把不匹配结果直接解释为结构收益。

该实验研究均匀正档位。将来使用学习分配或q0后，需要重新检查按内容选择带来的能量偏置，不能直接沿用这里的全点平均功率结论。

## 数据、验证与输出

语义教师在原始块上前向一次，y、可靠量化XYZ和有效点掩码缓存到CPU。y的主内存开销约为 `N*latent_dim*4` 字节，另加XYZ、掩码和少量填充；不改变邻域上下文，也不把采样点重新拼成块。

每1000步验证：

- 固定32个完整空间块、8次配对噪声的原始y误差及逐维偏差/随机方差。
- 逐层主干和补偿前后误差；从选中的块中均匀抽8块检查方向RGB、形状和不透明度损伤，对照clean属性与原场景属性。
- 原有留出相机中的固定16个视角，记录原PLY和照片PSNR、SSIM、VGG LPIPS、颜色误差。

初始化和最后一步使用全部原验证相机。三个档位同时报告clean、无噪声通信、有噪声通信；额外mixed仅为固定随机正档位诊断，不是学习分配部署。图像仅保存初始化和最终的代表视角，含无噪声通信图；不保存中间轮次图像。

所有结果位于一个父目录：

```text
NEW_OUT/
  plan.json
  point_noiseless.log, point_awgn10.log, ...
  各组/
    training.json, initialization.json, control.json, complete.json
    metrics/loss.jsonl, validation.jsonl, power.jsonl
    diagnostics/initial, step_001000, ..., final/
      latent_errors.json, latent_dimensions.csv, stage_metrics.csv
    images/initial, final/
    checkpoints/initial, latest, final/codec.pt
  comparison/summary.csv, summary.json, curves.png
  complete.json
```

`comparison/summary.csv`逐档给出画质、latent误差和真实功率；以完成组为范围检查功率匹配。每完成一组更新一次，不必等全部训练结束。`curves.png`区分无噪声/10dB测试，所有组使用相同数据和条件。中间画质只用16视角，最终用全部验证视角，因此比较最后一点时要注意视角集合差异。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
RUN="$PWD/output/truck_channel_continue6000_gpu2_20261020_153617"
OUT="$PWD/output/truck_power_controls_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_channel_normalization_truck.sh "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -f "${OUT}.log"
```

父日志展示当前运行组，细节查看该组日志，例如 `tail -n 30 -f "$OUT/point_noiseless.log"`。六组在指定的一张卡上顺序执行，可通过 `--arms statistical_noiseless statistical_awgn10` 选择子集；对比时保留相同源检查点、随机种子和训练预算。

先预检查：在启动命令后附加 `--dry-run`，只在CPU验证元数据和检查点，不创建结果目录、不加载场景、不训练。可用 `--no-lpips` 明确关闭LPIPS；默认启用，服务器需已有对应权重或可下载权重。

中断后使用完全相同的 RUN、OUT 和参数，附加 `--resume-suite`。已完成组跳过，当前组恢复latest的全部权重、统计尺度、Adam、RNG和真实更新游标。只在既定验证节点校准，不因续跑额外校准；未保存的日志更新会回退到检查点游标。

## 解释边界

实验不预设公共统计或块尺度更优。若无噪声训练改善而有噪声不改善，支持进一步研究幅度保护；若两者都没有改善，降低归一化作为主因的优先级。诊断里的有限试验均值含采样误差，逐维编号不对应固定颜色含义；逐层贡献是当前前向的条件观测，不能当作独立重训消融的因果结论。
