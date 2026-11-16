# 逐层精修后的混合档位续训

本轮从 `successive_tier_refine_views32_gpu3_20261115_161150/checkpoints/final/codec.pt` 出发，检查逐层精修取得的收益能否在混合交付训练后保留，并进一步改善混合交付质量。

## 训练设置

－原9个训练场景各150次更新，共1350次更新；场景轮换。Room仅评测。
－固定10 dB、8／24／48个复符号档位。每次场景更新为每个有效点独立、等概率抽取q1／q2／q3；q0仅标记补齐行。
－三层信道编码器和信道解码器共同训练，重新初始化Adam，学习率1e-5。语义编解码器、邻居修复模块、属性统计及原始高斯属性冻结。
－损失沿用源PLY渲染RGB MSE＋latent MSE，权重均为1。每次更新使用32个不同训练视角，先对这些视角的渲染损失取平均，再反向传播。
－沿用上一组的训练、16个验证及测试视角划分。训练视角持续轮换，记录覆盖次数；每50次场景更新评测统一q1、q2、q3及固定混合分配，各用4次配对噪声。
－固定末步作为结果。验证和测试指标不选择权重；通过完整曲线判断基础档位是否退化。

混合档位训练允许各层一起调整。q1收益的保持需要由结果验证；本轮没有添加单独的q1保持损失。随机混合分配用于检验编解码器对混合接收上下文的适应能力，后续学习分配器产生的分布仍需单独评测。

## 一致性与效率

复用冻结语义编码器输出，保留原Morton块、补齐位置、邻居上下文及尺度归一化。每个场景更新的档位图由场景名称、种子和访问次数确定；前向与反向重算复用档位图和信道噪声，下一次更新产生新样本。信道输出和尺度随当前参数重新计算。

训练前后评测相同视角、固定混合档位图和噪声种子，同时记录相对源PLY及照片的PSNR、SSIM、LPIPS和latent误差。分别查看9场景与Room的结果，重点检查：混合交付是否改善；统一q1的收益是否保持；q2／q3是否回退；无噪声和10 dB的变化是否一致。

通信成本采用 `digital_side_file_v1`：实际属性复符号数，加XYZ、完整元数据、块尺度三路文件的LDPC4096/6144＋16QAM成帧成本，含文件头、CRC与补齐，默认无ARQ。最终Truck四种分配各做4次实际侧信息传输，记录成功率、校验及成功条件下画质。初末成本在相同档位图下必须一致。

## 运行

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git pull --ff-only origin experiment/feedforward-student
conda activate maskgs
GPU=3
RUN="$PWD/output/successive_tier_refine_views32_gpu3_20261115_161150"
OUT="$PWD/output/successive_mixed_refine_views32_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 nohup bash scripts/test_successive_mixed_refinement.sh \
  "$RUN" "$OUT" --steps-per-scene 150 --views-per-step 32 \
  >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
printf '实验目录：%s\n' "$OUT"
tail -n 40 -F "${OUT}.log"
```

续训时保持RUN、OUT和所有参数相同，将RESUME改为1。训练与评测均可恢复。检查包为 `OUT/review.tar.gz`，包含指标、最终Truck图像和实际数字链路结果；模型权重保存在 `OUT/checkpoints/final/codec.pt`。

本地验证采用合成场景渲染和数字交付替身，检查缓存梯度与完整前向一致、全部信道层收到梯度、冻结范围、精确恢复、初始评测复现前组末步以及计费一致性。真实CUDA渲染和LDPC/AWGN交付由服务器实验完成。
