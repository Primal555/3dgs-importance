# 宽度 256＋完整历史标签实验

入口：`scripts/test_neural_capacity_history.sh CODEC_RUN truck NEW_OUT`。

基于有效的 `neural_correction_allocation`，从新收益网络和新概率表开始。默认宽度 256，可训练参数 206,082；原版宽度 64，参数 26,946。点特征、邻居池化、层数、损失和原版在线纠偏均沿用。

训练 700 个预训练窗口，其中前 50 个窗口拟合固定尺度，随后每窗口 64 次网络更新；完成后自动进行 2000 步分配，每步在线学习。编解码器冻结，默认 β=0.1、10 dB、q1/q2/q3=8/24/48，保留分支密集更新，正档位步长上限 1.0，保留分支上限 0.25。

## 完整历史回放

- 预训练与在线阶段均保留全部训练标签，包含完整四态教师和真实换挡标签。历史标签继续描述采集时的相机、背景和候选属性。
- `--replay-capacity` 是继承的兼容设置，在本入口中不淘汰标签。原有入口保持有限容量行为。
- 抽样仍是 25% 均匀、50% 最少访问、25% 困难样本，重要性权重为 `1/(N*p)`。新样本沿用旧样本优先级的下中位数。
- 优先级树和访问桶使单次抽样、优先级更新为预期 O(log N)，记录定位为 O(1)。诊断导出和检查点保存仍随历史数量增长。
- 磁盘保存完整标签，内存读取缓存由 `--replay-cache-mib` 控制，默认 256 MiB。历史不被容量或 GC 淘汰。完整历史磁盘占用会随训练增长。
- 保存随机数、优先级、访问桶顺序，恢复后继续相同的样本抽样顺序。新回放策略必须使用新的输出目录。

## 服务器运行

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
GPU=1
OUT="$PWD/output/truck_neural_capacity_history_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 nohup bash scripts/test_neural_capacity_history.sh \
  "$RUN" truck "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

700 窗口＋2000 分配步，共 2700 个窗口；网络默认累计 169,600 次更新。训练后下载 `OUT/review.tar.gz`。续训保留整个实验目录，包括 `pretrain/replay`、`allocation/neural_replay`、检查点与 `benefit_network.pt`。

与旧组比较固定测试视角下的 PSNR、SSIM、LPIPS、RD 和实际通信开销，并查看 `allocation/metrics/network_validation.jsonl` 中后期网络是否退化。旧实验侧信息计费与新成本口径存在差异时，优先报告各自相对全 q2 的收益，统一计费后再横向比较绝对成本。
