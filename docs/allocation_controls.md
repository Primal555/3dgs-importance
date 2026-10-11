# 冻结编解码器的 q0 三项对照

本轮不重训编解码器，不做联合微调，也不改变正式训练默认的采样规则。
已有结果的 `checkpoints/final` 用于部署规则测试；`checkpoints/end_render/codec.pt`
用于四组全新分配器训练。两份 codec 通常不同，因此不能把第一项与第二、三项直接当成同一基座排名。

## 测试内容

1. **部署规则：10、5、3 次。** 使用相同最终概率、相同视角和噪声，分别导出硬档位。
   只有全部抽到 q0 才不发送；保留点仍选正档位众数。同一随机种子下，5/3 次使用十次抽样的前缀，
   保证删点事件嵌套。报告删点数、理论期望删点数、保留点档位变化、真实开销和画质。
   **减少抽样次数同时改变删点和正档位投票，不是单独改变删除阈值的因果实验。**
2. **训练采样：single 与 deployment。** 从相同 `end_render` codec 出发，以相同初始化、优化器参数、
   β=0.01、每场景 1000 次更新分别学习分配器。single 使用单次硬采样和单次期望成本；deployment
   使用十次硬采样和十次期望成本。最终两者统一用十次部署规则测量。
   这同时改变保留与正档位的采样和代理梯度，不能声称只改变 q0。
3. **β：0.01、0.003、0.001。** 固定 deployment 训练，从同一 codec 和全新初始化出发。
   复用第二项的 deployment/β=0.01 基准，实际只需四组分配器训练，而非五组。

两种训练形式均为已有的局部、逐点渲染反馈代理；不是无偏梯度，也没有增加人为画质损失项。
本轮不预先认定单次训练一定更好：检验 q0 能否学起来，以及在十次部署时收益是否仍能保留。
原版 MaskGaussian 的单次训练、周期性十次永久剪枝和默认 PLY 单次采样导出是不同环节，
不能直接概括为“原版训练一次、部署十次”，也不能单凭次数不同认定方法有错误。

## 服务器运行

在项目目录、`maskgs` 环境下，选择空闲 GPU。以下示例使用 GPU 0，**不是要求必须用 0**：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
RUN="$PWD/output/multiscene_pruned_aligned_gpu0_20261010_194134"
OUT="$PWD/output/allocation_controls_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_allocation_controls.sh \
  "$RUN" "$OUT" > "${OUT}.log" 2>&1 &
printf '%s\n' "$!" > "${OUT}.pid"
tail -f "${OUT}.log"
```

如果只想先在 Train 上做较快的定位：

```bash
CUDA_VISIBLE_DEVICES=0 SCENES=train TEST_VIEWS=8 TEST_TRIALS=2 \
  nohup bash scripts/test_allocation_controls.sh "$RUN" "$OUT" > "${OUT}.log" 2>&1 &
```

`TEST_VIEWS=0`（默认）评估全部测试相机，正整数是明确标注的快速子集。
`ALLOCATION_STEPS=1000` 是**每个场景、每组分配器**的实际更新次数，可改成 2000。
`VALIDATE_EVERY=250`、`IMAGE_VIEWS=2`；只在最终测试保存代表图，不保存训练过程图。
原输出中的 manifest 路径失效时，可用 `MANIFEST=/新路径/multiscene_pruned.json` 覆盖。
所有输出集中于同一 `OUT` 下，子目录区分对照组。

运行前只检查文件并打印命令，不启动 GPU 训练：

```bash
python -m gaussian_jscc.allocation_controls --run "$RUN" --out "$OUT" --dry-run
```

单独执行第一项，不学习任何分配器：

```bash
python -u -m gaussian_jscc.multiscene_experiments \
  --manifest "$PWD/output/maskgaussian_three_20261010_190401/multiscene_pruned.json" \
  --checkpoint "$RUN/checkpoints/final/codec.pt" --allocations-dir "$RUN/checkpoints/final" \
  --role train --draw-counts 10 5 3 --require-allocations --shuffle-seeds \
  --out "$OUT" --snrs 10 --trials 3 --output-profile compact --image-views 2
```

## 看哪些结果

- `comparison.csv` / `comparison.jsonl`：实际 q0–q3 数量、原 PLY 与照片各自的 PSNR/SSIM、
  payload 复符号、真实坐标和完整档位元数据字节、总信道使用。部署规则组还记录
  `changed_positive_tiers_vs_ten`，防止把正档位变化误归于删除。
- `charts/*_controls.png`：最终画质—成本与实际档位占比。
- `charts/*_q0_history.png`：单次 q0 概率、十次期望删除数与实际部署 payload 的变化。
- 各组 `loss.jsonl`：训练实际抽样档位数、keep/tier 的图像梯度与成本梯度。
- 各组 `scenes/*/allocation_history.jsonl`：单次概率期望与十次真实部署比例。
- `progress.json` 与 `complete.json`：完成了哪些任务、整套测试是否完成。

冻结组结束时强制校验 codec 权重与统计量的 hash；结果记录 `frozen_codec_verified`。
冻结对照采用独立的视角随机序列，使 single/deployment 消耗随机数不同也不会改变训练相机序列。
不同 β 的总 loss **不可直接排名**；主要看画质和部署真实成本。
`equivalent_digital_MB` 仍只是指定数字效率下的资源等效值，不是模拟符号文件大小。
本轮也不检验过信道可靠传输坐标/元数据的纠错机制。

测试完成后仍建议用其他 `--deployment-seed` 重复第一项，检查有限采样波动。
单个部署种子或多次信道噪声试验不等于独立场景的统计显著性。
