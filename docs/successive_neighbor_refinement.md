# 两场景邻域恢复精修

本轮从 `successive_tier_refine_views32_gpu3_20261115_161150/checkpoints/final/codec.pt` 出发，验证逐层精修后，只调整邻域恢复能否提升混合档位画质，并保留三个单档的收益。

## 训练范围与预算

- Truck、Playroom 共享同一个模型，交替更新，各 150 步，共 300 步。
- 每个场景独立按 `mixed → q1 → mixed → q2 → mixed → q3` 轮换，共 75 步 mixed、三个单档各 25 步。mixed 中每个有效点均匀抽取 q1／q2／q3；每次更新重新抽取，前向与梯度重放使用同一份分档。
- 仅训练已有的 `channel_decoder.neighbor_refiner`，继承其权重。语义编解码器、三个信道编码阶段和三个信道解码阶段、统计量、源高斯属性固定。发送端的解码模拟也使用固定的解码阶段。
- 使用新 Adam，学习率 `1e-5`。目标为源 PLY 渲染 RGB MSE 加权重为 1 的 latent MSE。10 dB AWGN，每次更新产生新噪声。
- 每步 32 个不同训练视角，按场景轮换覆盖；保留源实验的 16 个验证视角及测试集。采用固定末步模型，每场景 50 步记录一次验证结果。

冻结发送参数保证同一场景、同一分档下发送符号保持不变。不同分档的块尺度和可用前缀仍随分档正常计算。语义 latent 按场景缓存，梯度经固定的语义解码器传回邻域模块；每个训练视角使用同一份已恢复场景。

## 判断依据

初始及最终分别评测两个场景的全 q1／q2／q3 和固定 mixed 图，包含无噪声及 AWGN；噪声评测使用 4 次配对试验。记录相对源模型与照片的 PSNR、SSIM、LPIPS，以及 latent MSE。重点检查 mixed 增益能否同时保留各单档画质。

每次保存检查全部冻结参数与持久缓冲区的摘要。初始与最终还对每个训练场景首批中的前两个完整块进行配对检查，覆盖四种分档及两种信道条件。记录发送符号、binary16 块尺度、邻域恢复前 latent 的最大变化，并记录恢复前后的 latent MSE。该抽样检查用于确认模块作用位置；全场景画质使用完整测试视角评测。

成本沿用 `digital_side_file_v1`：实际属性复符号及三个侧信息文件的分帧成本。最终对 Truck 四种分档执行实际 LDPC(4096,6144)／16QAM／AWGN 交付，每种 4 次，记录成功率、已发送成本与成功条件下画质。

## 运行与留存

```bash
CUDA_VISIBLE_DEVICES=3 RESUME=0 bash scripts/test_successive_neighbor_refinement.sh "$RUN" "$OUT"
```

`RUN` 为完成的逐层精修目录。默认无额外留出场景，本轮集中评测 Truck 与 Playroom。

查看 `metrics/loss.jsonl` 中的实际训练布局、`metrics/validation.jsonl` 中的过程指标，以及 `review_summary.json` 中的末步增量、布局次数和邻域检查。下载 `review.tar.gz` 即可分析结果。恢复训练需保留完整实验目录，尤其 `checkpoints/latest/codec.pt` 和 `metrics/neighbor_probe_initial.pt`，设置 `RESUME=1` 并使用原参数。
