# 校准一阶分配：提速与收尾实验

本轮从头学习概率表，默认 β=0.5，使用已训练好的共享编解码器。编解码器及源高斯属性冻结。

## 执行方式

1. 在原始块划分上合并各轮候选，默认候选批次 128。原始实际场景的采样顺序不变，从相同 RNG 状态产生原批次噪声，再将噪声复制到对应候选。
2. 同一步内，坐标及保留掩码相同的候选复用 PointNet 的采样、邻居和插值权重。各候选仍重新计算特征、符号、可靠尺度与接收属性；坐标或保留掩码变化时重建邻域。
3. 每次真实纠偏只重建目标所在的原始块，替换该块全部点的输出。32 次纠偏仍分别计算渲染变化。

全点保留反馈、两轮正档位候选、历史纠偏、q0 单次采样学习、十次规则部署，以及 0.25 的更新上限均沿用旧主线。启动时在实际设备上比较少量合批候选和保留切换，结果写入 `execution_check.json`。此检查覆盖代表性输入，完整训练的耗时与最终画质由服务器实验验证。

## 收尾规则

最多 2000 步，每 100 步在原有 4 个固定校准视角上评价相同种子的部署结果。

- 1250 步前持续训练。
- 之后连续 2 次校准未比 RD 锚点改善至少 0.5%，将保留与正档位学习率都降到原来的 0.5 倍。
- 降学习率后再次连续停滞，且最近 3 次部署映射的变化不超过 0.5%、总成本漂移不超过 0.5%，结束训练。
- 达到有效 RD 改善会清零停滞计数。档位持续变化时继续训练，直到满足结束条件或达到 2000 步。

这些阈值是本轮实验设置。降低学习率后仍受原来的步长上限约束，实际减速幅度可结合触顶比例判断。`--no-early-stop` 关闭收尾，方便单独测执行提速。测试视角不参与停止和选模；最终保存校准最优、末步及全 q2/q3 的测试评价。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

GPU=3
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_first_order_fast_beta0p5_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 nohup bash scripts/test_fast_first_order_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.5 --candidate-batch 128 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

`GPU` 设为空闲卡编号。恢复使用相同 OUT 与参数，将 `RESUME=0` 改为 `RESUME=1`。

正档位限幅对照：使用新 OUT，在相同命令后添加 `--tier-max-log-odds 1.0`。此时 q1–q3 的限幅系数为 1.0，保留分支维持 0.25，实际限幅仍乘以各点的 trust。日志额外记录正档位触顶比例、实际对数优势变化、更新前后概率峰值。未提供此参数时，现有实验的默认配置与恢复口径不变。

## 查看结果

- `metrics/loss.jsonl`：整步及候选解码时间、真实纠偏时间、候选调用数、邻域复用次数、有效学习率和更新触顶比例。
- `metrics/stopping.jsonl`：校准 RD、学习率调整、档位变化及成本漂移。
- `complete.json`：实际步数、结束原因、选中步数。
- `review_summary.json`：源渲染与照片指标、成本、RD、相对全 q2 的结果。
- 下载 `review.tar.gz` 即可检查结果。包内包含日志、汇总、图像和部署档位，检查点与概率数组留在服务器上。恢复另需 `checkpoints/latest/state.pt`、原编解码器及场景输入。

本轮首先比较单位时间内完成的等价评价数量，再检查收尾是否缩短总时长及最终 RD 是否保持。服务器尚未运行，因此当前没有 GPU 提速倍数。
# 64视角教师与正档位打乱对照

入口：`scripts/test_teacher_views64_shuffle.sh CODEC_RUN NEW_OUT`。每步使用64个训练视角，β=0.5，正档位限幅2，保留分支限幅0.25，候选批量128，最多2000步。沿用校准平台期收尾机制；4个校准视角、32个测试视角、4次配对AWGN试验，10 dB。

本组启用`digital_side_file_v1`成本：属性复符号加三路真实侧信息字节的LDPC4096/6144＋16QAM成本。C_ref为同协议全q3成本，选模使用该口径的RD。训练侧信息梯度使用每保留点平均成本代理，日志记录cost_proxy_scope；部署评价重新计量实际文件。历史32视角组采用旧训练计费，故64组与其比较同时涉及视角数与训练计费变化，历史结果需先重算统一成本再比较，不能将全部差异归为视角收益。

训练成功后自动执行`gaussian_jscc.first_order_positive_shuffle`。读取selected档位图，固定q0掩码，在所有保留点中随机置换q1/q2/q3，默认一次置换，种子20261112，可用环境变量SHUFFLE_SEED指定。各档数量和属性符号总数不变，XYZ文件相同；元数据及块尺度重新计费。保持模型冻结，使用相同测试相机、噪声种子、试验次数和图像指标，分别渲染selected与shuffled_positive。

结果位于`OUT/positive_shuffle/review_summary.json`，给出selected减去shuffled的配对画质差、总成本差、RD差和实际更换正档位的比例。该结果检验这一次置换下的空间档位分配收益。summary中明确区分计费与实际侧信息接收测试；本对照不执行额外优化。

下载`OUT/review.tar.gz`和`OUT/positive_shuffle/review.tar.gz`分别检查教师与打乱结果。训练可按相同OUT和RESUME=1恢复；打乱检查会复用配置一致的已完成评价。也可对一个已完成实验单独运行：

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m gaussian_jscc.first_order_positive_shuffle \
  --allocation "$OUT" --out "$OUT/positive_shuffle" --shuffle-seed 20261112
```
