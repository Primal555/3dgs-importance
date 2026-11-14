# 源场景监督的多SNR适应实验

## 起点与训练任务

入口为 `gaussian_jscc.multisnr_adaptation`，启动脚本为 `scripts/train_successive_multisnr_source.sh`。从 `successive_mip360_gpu1_20261023_142948/training` 的10 dB多场景共享模型初始化。原 `train_successive_multisnr.sh` 保留既有训练设置。

本轮针对两项已观察到的问题：信道恢复阶段的latent误差下降与场景画质提升不一致，以及多SNR适应后10 dB附近和高SNR的精度损失。训练仍使用原9个场景和相机划分，room只用于初始、最终的留出测试。语义编码器、属性归一化统计、点序和空间分块固定。

| 阶段 | 每场景步数 | 9场景总步数 | 可训练模块 | 学习率 |
| --- | ---: | ---: | --- | --- |
| 信道适应 | 1000 | 9000 | 信道编码器、信道解码器及邻居修复 | 1e-5 |
| 联合适应 | 250 | 2250 | 信道两端、语义解码器 | 1e-5 |

总计11250次优化器更新。进入每个阶段时新建Adam，各场景共享当前阶段的优化器。信道阶段学习率由上一轮的1e-4调整为1e-5，适用于已有联合训练权重的续训。

## 源场景监督与10 dB保持

信道阶段每步选择一个原始Morton块批次，复用冻结编码器的48维latent。对同一批点独立执行q1、q2、q3完整收发流程；每档按照自己的发送集合生成块尺度和邻居条件，共用标准化后的AWGN样本。混合档位窗口另外监督其混合布局。

主干恢复与邻居修复后的恢复均受监督。每个恢复结果的基本目标为：

\[
L_{\mathrm{local}}=L_y+0.1L_{\mathrm{response,source}}+w_c L_{\mathrm{color,source}}.
\]

局部响应和方向颜色的目标均来自原PLY属性。颜色辅助沿用每4步、梯度比例上限0.25的调度，统计从250步开始、500步后启用。每4步增加无噪声约束，包含latent误差和源属性响应，权重0.25，并轮换正档位。各布局和主干／修复输出等权平均。

信道阶段每场景第1步及每4步加入一次完整场景渲染反传，以源PLY渲染为目标；这部分使用当前窗口的SNR和档位布局。通过冻结的语义解码器反传到信道两端。该次渲染不重复添加全场景latent项，块级latent已在本步监督。

联合阶段每步使用完整场景源RGB MSE与latent MSE。语义解码器此时与信道两端一起更新，语义编码器保持固定。

两个阶段均每场景每4步加入10 dB保持约束，独立轮换q1、q2、q3和mixed。固定的初始模型与当前模型使用相同点块、相同10 dB噪声和相同档位。保持目标是初始模型的接收latent和最终解码属性：

\[
L_{10}=\|\hat y_{10}-\hat y_{10}^{\mathrm{initial}}\|^2
       +\|\hat a_{10}-\hat a_{10}^{\mathrm{initial}}\|^2,
\qquad w_{10}=0.1.
\]

属性误差在原有归一化属性空间计算，XYZ不纳入该项。保持目标保留初始模型已形成的信道与语义解码配合，后续可从曲线判断约束是否足够。该权重、学习率和周期来自本次实验设计。

训练SNR继续在5–15 dB内分层连续采样，每场景维护独立且可恢复的SNR区间×布局序列。源渲染缓存、latent缓存及多视角反向重放继续复用。主干与修复的局部损失、渲染损失、保持损失、梯度及逐档恢复误差分别写入日志。

## 低SNR数字侧信息

采用公开规则 `snr_qpsk_half_v1`，根据本次交付的已知SNR选择三路侧信息的相同协议。调用 `comparison_baselines/side_link_policy.py`，实际传输与预算计算共用该配置。

| 实际SNR | LDPC | 调制 | 每码字有效文件载荷 | 每码字复信道使用 |
| --- | --- | --- | ---: | ---: |
| 小于10 dB | k=4096，n=8192 | QPSK | 508字节 | 4096 |
| 大于等于10 dB | k=4096，n=6144 | 16QAM | 508字节 | 1536 |

两种配置均保持48字节文件头、每码字4字节CRC和尾部补齐。XYZ、完整元数据、块尺度分别成帧，无ARQ；属性模拟符号的功率和SNR保持原定义。低SNR侧信息成本为原配置的8/3倍，总成本增加幅度由各场景的侧信息占比决定。

初始、最终及uniform q1/q2/q3均按这套规则重新计算实际字节成本。本轮编解码器训练不包含RD选模或沿用历史C_ref；后续接入分配器时，必须按相同SNR的协议重新计算C_ref。与传统数字基线比较时，两侧使用同一公开MCS规则。

默认实际交付验证仍为Truck全q2，5、7.5、10、12.5、15 dB各4次。每个文件保留CRC、SHA256和成本记录，三路全部成功后计算画质，失败试验照常计费。可用 `--delivery-layouts 1 2 3 mixed` 扩展布局。收发两端的SNR/MCS为共享配置，接收端不访问源侧信息文件。

本地使用Sionna2.2.0、PyTorch2.9.0 CPU、20次LDPC迭代、同一份130000字节随机文件，完成256码字的真实链路检查。详细记录见 `multisnr_side_link_cpu_check.json`。5和7.5 dB下原配置均256个CRC失败码字，新配置均零误码并通过文件校验；10 dB原配置成功。服务器仍执行完整场景文件和原有多次噪声试验。

## 曲线与评测

时间轴在本轮定义为优化器更新数及累计优化耗时。累计耗时包含训练前后GPU同步，排除数据准备、评测和数字链路验证。

- **训练曲线**：默认监测Truck、Playroom的固定校准视角；三个档位、五个SNR、每个条件2次固定噪声。初始、第25和100个场景更新、随后每250个场景更新及每阶段末尾评测。绘制相对源PLY与相对照片的PSNR，并保留SSIM、LPIPS和latent误差原始数据。
- **最终SNR曲线**：9个训练场景及room留出场景，使用原测试视角和4次配对噪声，绘制初始／最终模型的PSNR、SSIM、LPIPS。
- **实际交付曲线**：数字文件交付成功率与总复信道使用次数随SNR变化。图像质量矩阵采用可靠侧信息假设，实际交付结果单列。

初始模型使用固定10 dB条件，实际噪声覆盖全部测试SNR；适应后模型使用匹配的条件SNR。曲线及JSON记录这一区别。校准曲线只用于报告，训练固定执行末步权重，不根据测试画质挑选检查点。

`figures/`包含PNG、PDF和CSV，`definitions.json`说明横轴、视角和通信口径。`review.tar.gz`包含曲线、指标和代表图，不包含模型权重与临时信号。

## 运行

```bash
conda activate maskgs
cd /data/home/zhangyueheng/projects/3dgs-importance
git switch experiment/feedforward-student
git pull --ff-only origin experiment/feedforward-student
GPU=3
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/successive_multisnr_source_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 nohup bash scripts/train_successive_multisnr_source.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

恢复使用相同RUN、OUT和参数，设置 `RESUME=1`。源10 dB检查点需保留，用于恢复固定保持教师。

## 参考流程

Compatible NTSCC+从10 dB模型继续进行多SNR训练；SwinJSCC采用固定条件、多码率、多SNR的分阶段流程。本轮把这两个原则用于已有四模块系统，具体源场景监督和10 dB保持约束由本项目的诊断结果确定。

- Compatible NTSCC+，IEEE JSTSP，2023，第VI-B节：https://arxiv.org/html/2303.14637v2
- SwinJSCC，IEEE TCCN，2024，第V-A4节：https://arxiv.org/html/2308.09361v2
