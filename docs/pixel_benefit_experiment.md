# 像素级收益评估实验

入口：`gaussian_jscc.pixel_benefit_allocation`。启动脚本：`scripts/test_pixel_benefit_allocation.sh`。

## 本轮实现

冻结现有四模块编解码器，在 10 dB、q0/q1/q2/q3＝0/8/24/48 下学习逐点换挡收益。每次干预重新解码该点所在的完整块，保留相邻点随档位改变产生的响应。q0 改变存在掩码时重新计算语义表示；四个候选沿用原批次的噪声槽位。

局部渲染器复用 MaskGaussian 的 CUDA 前处理。每个窗口缓存完整深度列表、投影参数和混合场景图像。候选扩大覆盖范围、移除前景后显露背景，以及 SSIM 窗口跨 tile 的影响均参与标签计算。标签为 q1–q3 相对 q0 的逐像素 MSE 与加权结构误差，使用双精度积分恢复整图失真差。

网络读取 16×16 原像素区域及 5 像素上下文、当前图像残差、完整深度序列和候选投影参数。场景和候选序列使用批量 GRU 编码，像素查询读取全部有效 token；卷积感受野覆盖完整 SSIM 上下文。标签图像只用于监督。

损失包含逐像素误差和完整干预组的积分误差。尺度由前 50 个窗口的训练标签拟合后固定，两种误差分量各用一个跨档位共享尺度。训练采用两遍分块计算：先计算完整组积分误差，再逐块反传对应梯度。训练标签保持原始精度。

## 训练与验证

- 预训练 200 个窗口。前 50 个窗口收集标签及拟合尺度，后 150 个窗口每窗口更新网络 16 次，每次抽取 4 个完整组。
- 每窗口抽取 64 个完整干预组。采样分布由均匀覆盖、投影影响和训练误差组成，记录并校正采样概率。重复抽到的组复用候选和标签计算。
- 留出完整空间块。固定验证库最多 512 组，在尺度拟合结束时冻结；后续评估使用同一批标签。每组包含窗口内各视角，验证时先平均多视角收益，再计算方向错误和 regret。
- 预训练期间概率表固定。显式启动分配阶段后，执行 2000 次概率更新，同时继续收集标签、训练收益网络。
- `latest` 和 `final` 保存实际末步状态，包括网络、Adam、概率表、回放索引及各随机状态。分配阶段每 250 步评估真实部署质量；初始和最终保存渲染图。

## 计算复用与存储

候选按完整块批量解码；场景索引只建立一次。每个相机的完整投影和排序每窗口构建一次。相同图块的上下文索引与场景编码使用有界缓存，窗口切换或参数更新后重新计算。

局部合成按 128 个 tile 分批，密集 SSIM 标签按 16 个 tile 分批。变长深度序列仅在当前网络批次内填充。标签逐视角转到 CPU，回放分片保存在磁盘。

回放默认保留 8192 个训练组及固定验证组。检查点旁的 `replay.json` 记录分片引用；保存完成后，只回收已经逐出且未被现存检查点引用的本轮缓存分片。继续训练需要保留整个 `replay` 目录及检查点。仅下载报告时可取 `metrics`、`evaluation`、`deployment` 和完成标记。

## 服务器运行

首次运行会单独编译 CUDA 扩展，需要现有 mask rasterizer 源码、CUDA 编译环境和 Ninja。随后使用真实场景检查局部渲染与原渲染器、候选配对解码、密集标签积分的一致性。结果保存为 `preflight.json`；任何检查失败都会停止训练。

新增图块覆盖使用独立的 64×64 三高斯样例检查，并验证新增区域确实产生像素变化。真实场景的两个编解码块仍执行四档全图对照。该样例避免真实块已经覆盖全部图块时无法构造扩展测试，图像与失真误差阈值保持不变。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_pixel_benefit_gpu1_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_pixel_benefit_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
printf 'PID：%s\n输出：%s\n' "$!" "$OUT"
tail -n 40 -F "${OUT}.log"
```

200 个窗口完成后，检查 `metrics/benefit_validation.jsonl` 中固定验证集的方向误判、三档选择 regret、共同误差和最严重误判。末步状态位于 `checkpoints/latest/state.pt`。

启动后续分配时，使用同一 `RUN`、`OUT` 和参数：

```bash
CUDA_VISIBLE_DEVICES=1 RESUME=1 nohup bash scripts/test_pixel_benefit_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 --phase allocation >"${OUT}_allocation.log" 2>&1 &
tail -n 40 -F "${OUT}_allocation.log"
```

中断预训练后，使用 `RESUME=1` 并保留 `--phase pretrain`。分配中断后继续使用 `--phase allocation`。

## 检查输出

`metrics/loss.jsonl` 记录解码、投影、标签、特征和网络耗时、评价组数、tile 数、缓存命中、网络更新数、档位占比、通信代价、回放字节量及 GPU 峰值。分段时间为包含同步点的主机墙钟时间，整步时间在 GPU 同步后统计。

`metrics/benefit_validation.jsonl` 记录固定验证集收益预测结果；正档位指标包含每组三个档位之间的六个有向比较。`metrics/validation.jsonl` 记录真实部署的 PSNR、SSIM、LPIPS、通信开销和归一化率失真目标。

本地验证覆盖 CPU 数学参照、真实小编解码器联调、分块梯度一致性、断点恢复和缓存回收。CUDA 编译、GPU 数值一致性及 Truck 全规模吞吐由服务器启动检查和训练日志确认。
