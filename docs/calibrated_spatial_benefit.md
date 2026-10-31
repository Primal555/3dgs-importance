# 软方向标定：从头训练 200 个窗口

独立入口：`gaussian_jscc.calibrated_spatial_benefit_allocation`。
独立实验族：`calibrated_spatial_benefit_allocation_v1`。

## 本轮配置

只运行新版，上一轮空间配对实验作为对照。已有编解码器复用并冻结；收益评估网络随机初始化，optimizer、replay、尺度拟合和概率表均从初始状态开始。

- Truck，10 dB AWGN，档位 0/8/24/48，β=0.1。
- 200 个预训练窗口，前 50 个收集标签并拟合固定尺度。
- 后 150 个窗口各 64 次更新、每批 8 组，共 9600 次网络更新和 76800 组次训练。
- 真实标签预算 64 组／窗口，验证 512 组，活动 replay 最多 8192 组。
- 沿用逐邻居空间输入、真实整块换挡渲染、MSE＋SSIM、重要性修正和批量推理。
- 本轮运行预训练，概率表保持冻结，保存实际末步网络。

## 唯一训练目标变更

每个状态的 RD 代价为两个失真分量之和，加上解析通信代价。六对档位的真实差值为 ΔJ，预测差值为 ΔJ_hat，除以原有固定训练尺度 s 后得到 a、e。

软目标 p=sigmoid(-a)，预测概率 q=sigmoid(-e)。方向损失为：

`mean(weight * KL(Bernoulli(p) || Bernoulli(q)))`

其中 weight 和近似平局判定沿用旧方向损失，方向权重维持 0.25。绝对值、共同值和全部档间差值平方损失保持不变。该目标在预测值等于真实标签时，方向损失与其梯度均为 0。

实现通过稳定的 BCEWithLogits 差值计算 KL，避免对 0 或 1 求对数。目标、权重停止梯度，减法舍入产生的微小负值截为 0。网络结构、输入归一化、输出尺度及随机参数初始化均未修改。

## 服务器启动

以下使用 2 号卡，输出目录必须为新目录：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_soft_direction_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 RESUME=0 nohup bash scripts/test_calibrated_spatial_benefit.sh \
  "$CODEC" truck "$OUT" --beta 0.1 --phase pretrain >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

本轮意外中断时，使用同一个输出目录和同一配置，设置 `RESUME=1` 恢复。

## 在现有 200 个窗口上续训 500 个窗口

`--pretrain-until 700` 表示预训练累计目标为 700 个窗口。固定尺度、网络、Adam、replay、验证集及随机状态均从最新 checkpoint 恢复；编解码器和概率表继续冻结，原训练配置不变。

当前第 200 个窗口已完成 9600 次网络更新，额外 500 个窗口增加 32000 次更新，最终累计 41600 次。继续每 50 个窗口验证一次，比较三档／四档 regret、保留 regret、共同误差和 MSE 分量误差的走势。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_soft_direction_gpu2_20261030_152022"
CUDA_VISIBLE_DEVICES=2 RESUME=1 nohup bash scripts/test_calibrated_spatial_benefit.sh \
  "$CODEC" truck "$OUT" --beta 0.1 --phase pretrain --pretrain-until 700 \
  >"${OUT}_continue700.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}_continue700.pid"
tail -n 40 -F "${OUT}_continue700.log"
```

确认原进程已经结束后启动。续训仍写入原目录，现有 `metrics` 记录保留，并从第 201 个窗口追加。第 200 个窗口的完成标记归档为 `pretrain_complete_000200.json`，700 个窗口完成后写入新的 `pretrain_complete.json`。原 `plan.json` 保留初始 200 窗口配置，累计目标写入 checkpoint 的 `progress.pretrain_target` 和完成标记。

中断后可重复上述启动命令；目标仍为 700。也可以省略 `--pretrain-until`，程序会恢复 checkpoint 中已保存的累计目标。其他参数必须与原实验一致。缩短目标或修改训练配置会报错。

## 查看结果

沿用 `metrics/benefit_validation.jsonl` 的全部既有字段，额外记录 `direction_objective=rd_gap_bernoulli_kl_v1`。旧版的检查命令可直接使用。

重点比较 100、150、200 窗口的共同误差 RMSE、MSE 分量 RMSE、保留／删除 regret 和四档 regret，同时检查三档 regret 能否保住已有优势。训练侧 `decision_loss` 现在是 KL，其数值不宜与旧硬方向损失直接比较。

输入与标签缓存策略保持上一版：空间 replay 体积较大，精确恢复需要保留完整 replay 和 checkpoints；分析结果可先下载日志与 metrics。

分配阶段可使用 `--allocation-batch 64` 恢复为批量投影与空间特征路径，详见 [批量执行与断点恢复](batched_spatial_allocation.md)。
