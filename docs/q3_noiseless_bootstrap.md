# 固定 q3、全 32 复符号、无噪声一阶段诊断

入口：`scripts/test_q3_noiseless_bootstrap.sh`。

目的：排除低档位截取、跨档位轮换训练和信道噪声，检查当前链路在更宽松条件下能否学习有效恢复。不是新通信方案，也不是证明 32 个复符号一定足够或能够无损表示 Gaussian。

## 固定条件

- 当前几何近邻 Transformer 编码器：`learned_split_logcov / multiscale_self / geometric_point`。
- 解码器、logcov 输出、`spatial_logcov_v1` 目标、输入分块和功率归一化不变。
- 随机初始化；默认一阶段 10000 步；二阶段和 mask 联合优化为 0。
- 所有真实 Gaussian 固定 q3，32 个复符号全部保留；q0 仅用于 batch 补齐，不丢弃真实点。
- `channel=none` 是恒等信道，**不是提高到某个有限 SNR**。`SNR=10` 仅保留为网络条件输入，不代表实际施加 10 dB 噪声。
- 学习率固定 2e-4、默认 batch 32 个块、默认不裁剪梯度，与上一轮保持一致。
- 训练内验证仅评估 q3；每 500 步保存 checkpoint，训练结束后只渲染 q3，避免把没有训练的 q1/q2 当作本轮效果。
- 无信道噪声且无混合档位，验证 trial 固定 1；训练仍有局部块抽样、响应视角抽样，不是完全确定的全场景拟合。

启动器会覆盖继承的 `CHANNEL`、`BOOTSTRAP_TIER`、`SNR`、`VALIDATION_TRIALS`，防止误跑旧配置；其他实验默认行为不变。新参数 `--bootstrap-tier` 仅允许一阶段、drop=0，避免悄悄改变已有联合训练语义。

## 后台启动（先确认 GPU 2 空闲）

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git pull --ff-only --no-recurse-submodules origin main
export PYTHON_BIN="$(command -v python)"
export PLY="$PWD/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply"
export SCENE="$PWD/data/tandt_db/tandt/truck"
export CUDA_VISIBLE_DEVICES=2
export BOOTSTRAP_STEPS=10000 SAVE_EVERY=500 VALIDATE_EVERY=100
export BLOCKS_PER_BATCH=32 ENCODER_NEIGHBORS=16 LR=0.0002 SEED=42
export RENDER_HISTORY=1 RENDER_VIEWS=8 RESOLUTION=4
mkdir -p output
OUT="$PWD/output/truck_q3_noiseless_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/test_q3_noiseless_bootstrap.sh "$OUT" > "${OUT}.log" 2>&1 < /dev/null &
echo $! > "${OUT}.pid"
printf '结果：%s\n日志：%s\n' "$OUT" "${OUT}.log"
tail -f "${OUT}.log"
```

启动日志应出现 `Fixed q3`、`32 complex symbols`、`channel=none` 对应的无噪声说明；验证仅显示 q3。Ctrl+C 退出 tail 不会停止后台训练。

日志 `loss.jsonl` 新增 `tier_counts` 和 `symbols_per_source_gaussian`，本测试应为 `[0,0,0,N]` 和 `32`，不把补齐槽位计入 N。`training.json` 记录固定档位和恒等信道。

结果集中在同一个 `$OUT`：

- `bootstrap_validation.jsonl`：留出块位置、形状、外观误差。
- `render_history/metrics.csv`、`quality_vs_step.png`：每 500 步的 q3 渲染指标。
- `render_history/validation_images/000500/3_view00.png` 等：照片 / 原 PLY / 恢复图 / 误差图。

历史渲染是**训练完成后**自动执行，不更新任何模型权重。手工调用历史评估时须传 `--tier 3 --channel none --snr 10 --trials 1`。

## 如何解读

如果本轮明显优于此前多档位 AWGN 训练，说明被排除的因素中至少有一项很重要；这次同时去除了多档位训练与噪声，不能单凭它认定是哪一项。

如果仍然失败，优先检查位置表示、解码路径、训练目标和有限潜在容量；不能据此证明是某一个模块的问题，也不能把无噪声 32 符号当作数学上的无损上限。

CPU 测试验证了真实编解码/梯度、固定档位采样、功率掩码、恒等信道、checkpoint 和 q3-only 评估逻辑；历史渲染集成测试使用 CPU 渲染替身，不是实际 Truck 图像质量证据。
