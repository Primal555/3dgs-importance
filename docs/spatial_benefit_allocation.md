# 空间配对收益评估实验

入口：`gaussian_jscc.spatial_benefit_allocation`。

## 当前配置

- 复用并冻结 `successive_mip360_gpu1_20261023_142948/training` 的编解码器。
- Truck，10 dB AWGN，q0/q1/q2/q3 对应 0/8/24/48 个复符号；沿用可靠坐标与块尺度交付。
- 200 个预训练窗口。前 50 个窗口收集标签并拟合固定尺度；后 150 个窗口各执行 64 次网络更新、每批 8 组，合计 9600 次网络更新、76800 组次标签训练。每窗口真实标签预算维持 64 组。
- 每组包含同一点四档整块重建在当前相机窗口的真实失真变化。所有状态共用噪声槽，原精确渲染教师保持不变。
- 预训练阶段固定概率表。预训练后显式启动 2000 个分配窗口，继续收集标签、更新网络与逐点概率表。
- 始终保存实际末步网络，独立实验族，不加载旧版收益网络断点。

## 三项调整

### 空间对应

每个块内邻居保留独立 token，包含可见性、基线残差、绝对／相对位置、深度关系、目标点标记与有序足迹采样。候选 token 包含各档属性、投影中心及中心处的目标／残差 RGB。共享点网络先处理配对输入，再作加权聚合和最大聚合。完整块参与计算，不按距离截取邻居。

基线可见性每相机计算一次，足迹采样按块缓存。训练按组、相机批量计算；在线推理按 `--candidate-batch` 合并源点及其全部相机，默认 16 个候选。合并仅改变网络调用粒度，真实干预及其噪声、成本对应关系保持不变。

### 标签复用

replay 采样概率为：25% 活动池均匀、50% 当前最少更新记录均匀、25% 困难度采样。每次返回真实混合概率，并使用 `1/(N*p)` 修正，同时保留源点提议分布的重要性权重。只有成功完成 optimizer 更新后才增加访问次数；重复抽中分别计数。

逐记录访问次数保存在 checkpoint 的 `replay.usage`，记录移出活动池后计数继续保留。CPU LRU 缓存默认 256 MiB。验证集隔离、固定，其标签不参与训练、尺度拟合和困难度更新。

### 收益监督

固定尺度 `s = RMS(ΔMSE/D_ref) + RMS(λ_s*Δ结构误差/D_ref)` 仅从训练标签拟合。两个分量共用 `s`。损失包含：逐相机绝对收益平方误差、三档共同收益平方误差、q0—q3 全部六对差值的平方误差，以及真实误判代价加权的方向损失。前三项权重均为 1，方向损失权重为 0.25。方向决策使用相机窗口平均值，显式加入各档通信成本。

标准化输入经过 `asinh`，缓和极端输入幅度；输出保持线性范围。全部特征与 replay 使用 float32，真实标签及收益监督使用 float64。

## 运行

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_spatial_benefit_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_spatial_benefit_allocation.sh \
  "$CODEC" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

完成 200 窗口后，如需进入分配阶段，保留同一个 `OUT` 和其余参数：

```bash
CUDA_VISIBLE_DEVICES=2 RESUME=1 nohup bash scripts/test_spatial_benefit_allocation.sh \
  "$CODEC" truck "$OUT" --beta 0.1 --phase allocation >"${OUT}_allocation.log" 2>&1 &
```

服务器启动时检查原渲染器、可见性和新空间输入的 CPU／CUDA 一致性；失败即停止训练。

## 查看结果

`metrics/benefit_validation.jsonl` 保留旧版 `positive/keep` 字段，并增加：

- `weighted_validation`：四档／三档 regret、固定档位基线、MSE／结构分量误差、共同／中心误差。
- `weighted_training`：确定性均匀抽取最多 256 组训练记录，再按源点提议权重统计，便于检查拟合差距。
- `replay_coverage`：活动池最小／平均／p10 访问次数、从未训练比例、累计真实组次。

`metrics/loss.jsonl` 同时记录绝对值、共同值、档间差值、方向损失与访问覆盖。用相同验证集比较第 50、100、150、200 窗口；重点关注三档选择能否优于固定 q2，以及四档决策 regret 是否持续下降。

## 存储与表示范围

空间 replay 比 158 维输入大。按四相机、256 点完整块估算，8192 个活动组约需 5.6 GiB 原始张量，另有验证组及断点保留的 shard。实际大小写入 `replay_live_bytes`。精确续训需要保留 replay 与 checkpoints；查看结果只需下载日志和 metrics。

输入中的足迹采样和基线可见性属于网络描述，不能完整表达连续图像及所有遮挡变化。标签仍来自精确整块干预。真实画质及分配效果需要本轮服务器结果确认。
