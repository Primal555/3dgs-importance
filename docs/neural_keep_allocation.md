# 全点神经保留＋空间正档位网络

独立入口：`scripts/test_neural_keep_allocation.sh`。原混合入口 `test_hybrid_tier_allocation.sh` 保持可用。

| 阶段 | 默认预算 | 学习内容 |
| --- | --- | --- |
| 空间收益预训练 | 700 窗口 | 从随机权重学习四状态空间价值；前 50 窗口拟合尺度，之后共 41600 次网络更新 |
| 档位分配 | 2000 步 | 全点保留网络、平滑保留概率与稀疏正档位表；空间正档位网络默认冻结 |

编解码器与源高斯属性全程冻结，使用 10 dB、0/8/24/48、可靠 16bit XYZ、binary16 块尺度。目标为归一化 MSE＋SSIM 失真与通信代价，默认 beta=0.1。部署继续使用十次规则，和全 q2、全 q3 同条件比较。

## 保留与正档位分工

保留网络输入共 69 维，包括缓存的属性/KNN/坐标、上一轮恢复与可见性统计、条件正档位概率以及信道/代价信息。两层 64 维 SiLU 隐藏层输出保留 logit，初始保留概率 0.99。每步对全部点提出新概率，以 0.1 权重与上步概率平滑后采样。镜像更新只写入正档位表，保留概率由网络提议更新。

保留网络每步接收全图渲染损失对存在门控的梯度，以及期望通信成本梯度。按 65536 点分块重算轻量网络，并使用执行采样时保存的相同 Gumbel 噪声回传，避免保存全场景网络激活。该路径使用 ST 局部导数，恢复属性在本次门控求导中固定；真实干预校准补充改变保留状态后邻域重编码的影响。

真实校准与原有限换挡共用每步最多 32 次渲染预算。通常为正档位预留 24 次，余量可组成 2 个完整 q0–q3 组，共 6 次渲染；当前状态复用。标签是在相同视角与噪声下，三个保留档位相对 q0 的实际渲染损伤，按当前条件正档位概率混合，再加属性/尺度边际成本与平均 XYZ 成本代理。完整报文成本在部署校准与最终评估时实测。闭点和保留点均参与覆盖与边界采样。

真实标签使用预训练固定尺度构造软方向目标，默认校准权重 0.1，保留网络 Adam 学习率 1e-4。部分空间块的标签只报告校准误差；密集渲染目标仍覆盖全部点，因此该留出统计衡量校准标签效果。正档位网络只通过三档间的价值差参与选择，其相对 q0 的共同偏移不用于保留决策。

## 行序一致性与对照

新入口在预训练、分配两个阶段统一修正原 PLY 行号与 Morton 包顺序的映射。静态特征按原行号索引，闭点投影探针按包内行号读取属性。原混合入口继续保持其已有行为，因此跨入口比较包含此修复的影响。

需要单独比较保留学习方式时，在本入口的新 OUT 增加 `--keep-estimator first_order`。两个选项具有相同的修正输入和空间网络预训练；对照分支使用全点一阶保留反馈及原保留历史、纠偏、镜像更新。默认 `--keep-estimator neural` 使用本轮神经保留分支。

## 启动与恢复

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

GPU=2
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_neural_keep_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_neural_keep_allocation.sh \
  "$CODEC" truck "$OUT" --beta 0.1 --network-mode frozen >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
echo "$OUT"
tail -n 40 -f "${OUT}.log"
```

700＋2000 自动衔接。`--network-mode frozen` 冻结空间正档位网络，保留网络在分配阶段继续学习。若显式选择 online，会另外采集正档位标签并训练空间网络，增加运行预算。

同配置恢复时重新指定 GPU、CODEC、原 OUT，并使用：

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/test_neural_keep_allocation.sh \
  "$CODEC" truck "$OUT" --beta 0.1 --network-mode frozen >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
```

## 检查与下载

- `pretrain/metrics/benefit_validation.jsonl`：空间正档位网络学习曲线。
- `allocation/metrics/loss.jsonl`：全点保留更新、梯度、十次删除期望、真实校准记录、候选预测和耗时。
- `allocation/comparison/`：部署档位、成本、画质与 q2/q3 对照。
- `review_summary.json`、`review.tar.gz`：检查结果时优先下载，排除 checkpoint 和 replay。

只保存初始与最终选中结果的图像及档位分布图；末步另外保存指标与档位数组。保留网络、Adam、上轮观测、覆盖和 RNG 随 allocation checkpoint 保存，用于精确恢复。冻结模式恢复需紧凑 benefit_network.pt、allocation checkpoint、源编解码器和场景；预训练恢复还需 pretrain replay。保留观测缓存额外约占每点 144 字节，Truck 约 98 MiB。

本地已验证块重算梯度、相同噪声、全点与全闭点反馈、真实渲染预算、行序映射、冻结网络、完整流程与断点恢复。完整相机渲染用合成函数验证流程，CUDA 最终画质及速度由服务器实验确认。
