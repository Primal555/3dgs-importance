# Gaussian-JSCC 训练性能优化

本次优化针对 `python -m gaussian_jscc train` 的完整场景渲染阶段。
保留所有待传 Gaussian、原始 Morton 分块、三种正档位、逐块功率归一化以及原有渲染/属性损失。
未引入可见性剪枝、局部渲染近似、混合精度或额外的预测分支。

## 改动与适用范围

- **网格算子向量化**：把角点特征和质量累加合并为一次 `index_add`。默认一次网格聚合由
  120 次累加变成 12 次；这是算子调用数变化，不是实测加速倍数。
- **独立块批处理**：默认一次处理 4 个块，每块保留独立网格、上下文边界和符号平均功率。
  尾块补齐使用 q0，不参与上下文、功率、属性损失或渲染。实际传输的 packet 格式不变。
- **编码端几何映射复用**：一次前向中的多个编码层复用索引和插值权重。解码端预测位置
  每层变化，仍重新建立网格；四档 mask 变化后也重新计算对应权重。
- **输入缓存**：固定源参数只归一化一次；`--training-data-device cuda` 缓存在指定 GPU，
  `cpu` 缓存在内存。缓存不是接收端辅助信息，也不缓存过期的网络输出。
- **完整场景梯度重放**：先无梯度恢复整个场景，然后对完整渲染反传获得 `dL/dG_hat`，
  最后逐批重新前向并执行向量—雅可比积，累积 codec 梯度。每批重用第一次前向的 RNG 状态，
  包括信道噪声/衰落，且恢复重放前的全局 RNG 状态。整个步骤之后只更新一次参数。
- **分段计时**：记录 codec 前向、渲染前向、渲染反向、codec 重放反向的时间，以及峰值显存。
  render 每步刷新日志和进度条，不再等待 10 个慢迭代。

梯度重放使用链式法则：

```text
G_hat = concat(codec(batch_1), ..., codec(batch_M))
v = d D_render(G_hat) / d G_hat
dL/dtheta = sum_b v_b * d codec(batch_b)/dtheta + lambda * d L_aux/dtheta
```

各局部块原本就是独立编解码的，完整渲染仍同时包含所有块，因此块间遮挡与颜色合成的梯度
仍通过 `v` 传递。重放需要额外前向，原先 checkpoint 同样需要重算；它的主要作用是将
codec 激活内存限制为一批，让批处理不必同时保留全场景的 codec 计算图。
完整场景输出、输出梯度和光栅器内存仍与场景规模有关。

共享的网格优化同时作用于 `train-route2` 和推理；**批处理/梯度重放训练循环目前只接入
独立 codec 的 `train`**。`train-route2` 仍用原有四档联合训练循环，mask 梯度回归测试保留。

已有 version 2 `codec.pt` 可直接载入，没有增加需要学习的参数。
浮点累加顺序改变，不能保证与旧代码逐 bit 相同；建议收发两端使用同一代码版本。

## 先测服务器实际耗时

脚本执行真正的完整场景前向和反向，但不执行 optimizer step，不改写 checkpoint。
默认 1 个预热步骤、3 个计时步骤，固定 SNR，采样训练视角和正档位。

```bash
cd /root/MaskGaussian-main/MaskGaussian-main
PROJECT="$(pwd)"
PLY="$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply"
SCENE="$PROJECT/data/tandt_db/tandt/truck"
INIT="$PROJECT/output/truck_codec_only/codec_20000.pt"

CUDA_VISIBLE_DEVICES=0 python -u benchmark_training.py \
  --ply "$PLY" --checkpoint "$INIT" --source "$SCENE" \
  --out "$PROJECT/output/truck_training_perf" \
  --blocks-per-batch 4 --backward replay \
  --warmup 1 --iterations 3 --snr 10 --resolution 2 --device cuda
```

`INIT` 可换为实际存在的最新检查点；输出目录必须是新目录。
结果为 `timings.jsonl`、`results.json`、`config.json`。
`results.json` 中 `summary.median_seconds` 给出排除预热后的单步时间。
测量包含梯度裁剪，不包含优化器更新、数据准备或 checkpoint 写盘。

要测同一套新代码中批量大小/反向策略的影响，可使用新输出目录并改成：

```bash
--blocks-per-batch 1 4 --backward checkpoint replay
```

这将测量 4 种组合，两种策略都使用优化后的网格，不是旧版本的完整性能基线。
不同批大小会改变随机数的消耗方式，因此有噪信道的瞬时 loss 不要求完全相同；
脚本使相机和每个 Gaussian 的档位在各配置之间一致，不更新权重。
如果 4 块批处理显存不足，可降为 2 或 1；如果仍慢，先检查分段耗时，特别是光栅化部分。
随机或尚未训练好的解码器可能生成投影范围很大的 Gaussian，导致渲染本身缓慢。

## 从已有权重继续渲染训练

不要重复前 20000 步属性训练。使用已保存权重以及 `--steps 0`：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m gaussian_jscc train \
  --ply "$PLY" --source "$SCENE" --init "$INIT" \
  --out "$PROJECT/output/truck_codec_render_fast" \
  --steps 0 --render-steps 2000 \
  --blocks-per-batch 4 --render-backward replay \
  --training-data-device cuda --profile-every 10 \
  --snr-range 0 20 --resolution 2 --save-every 25 --device cuda
```

`--init` 加载模型权重、架构和属性统计量；**不恢复 Adam 状态或原始 step 计数**。
例如加载 `codec_20000.pt` 后，新目录中 `codec_25.pt` 表示又训练了 25 步。
`2000` 是新增步骤数，可设置为计划剩余的渲染步骤；不是必须运行的固定次数。
输出目录必须未存在。每 25 步保存，减少停止任务时丢失的未保存更新。
本次没有增加 SIGTERM 自动保存功能。

日志示例字段：

```text
loss, render_loss, aux_loss, snr, step_seconds
codec_forward_seconds, render_forward_seconds, render_backward_seconds
codec_replay_backward_seconds, peak_allocated_mib, peak_reserved_mib
retained_gaussians, codec_batches
```

`--profile-every 10` 在每 10 个渲染步骤同步 GPU 计时，其他步骤只记录整体步耗时。
`--profile-every 0` 关闭分段同步。checkpoint 模式将反向耗时记录为
`combined_backward_seconds`，因为它混合了渲染反向与 codec 重算。

## 验证边界

```bash
python -m unittest discover -s tests -p test_training_performance.py -v
```

测试包括：原始循环网格与向量化网格的输出和梯度、批量块与逐块实际 pack/decode 的
等价性、包括信道随机性的普通反传/checkpoint/replay 梯度等价性，以及训练 CLI 的
日志与权重接续流程。CPU CLI 测试替代了 CUDA 光栅器，不能据此宣称渲染性能已验证。
CUDA 设备 RNG 测试在有 GPU 时运行；没有 GPU 时明确跳过。

本地环境只能验证 CPU 数值正确性，**尚无 4090 加速倍数或显存峰值实测结论**。
