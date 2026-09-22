# 512 点感受野与逐级 XYZ 细化实验

本轮是两个可分离的假设，不宣称已经修复渲染质量。原来的
`scripts/test_transformer_trunk_q3.sh` 默认仍然是 256 点纯主干；
旧 checkpoint 的默认配置和参数结构不改变。

## 结构

新增 `--decoder-refinement progressive`，只支持
`decoder_attention=transformer_trunk`、`decoder_memory=none`。
不是上一轮的 memory cross-attention，也不叠加定位 token。

```text
接收符号 + 符号 mask + 档位/SNR条件 → 接收嵌入 memory
  → 第1层全注意力 → 初始 XYZ
  → 第2层：当前状态 + 同一个 memory 的投影 + 预测 XYZ 特征
            → 带预测相对距离偏置的全注意力 → 逐点 ΔXYZ
  → 第3层：使用更新后的 XYZ 重复上述细化
  → 第4层：再次细化 → 最终 XYZ
                    → opacity / logcov / DC / SH
```

- 每点输出一组坐标和属性，不改变点数量或点与属性的对应关系。
- XYZ 特征是预测绝对坐标与预测块中心化坐标的拼接。均从接收端预测得到。
- 注意力额外偏置为 `-softplus(log_precision[head]) * ||xyz_i-xyz_j||²`。
  仍然保留全注意力，没有用早期不可靠坐标建立硬 kNN，也没有邻居截断。
- 距离在全局 bbox 的逐轴归一化坐标中计算，不冒充世界坐标各向同性距离。
- 每层输出残差 `xyz_next = xyz + delta`；坐标不 detach、不截断、不用 sigmoid。
- 接收嵌入不是新的发送数据；所有预测和修正仍依赖同一份每点 32 复符号 payload。
  原有全局 bbox 等元数据协议不变，没有新增粗坐标、源点邻域或源点中心。
- 仅最终输出使用原来的 `spatial_logcov_v1` 目标，无中间坐标辅助损失、
  投影损失或新损失权重。各层细化不保证逐层误差单调下降，必须观察日志。
- 初始头末层标准差 .02、bias .5；细化头末层标准差 .002、bias 0；
  距离 log_precision 从 0 开始学习。这些是工程初始化，不是最优参数声明。
  非零小权重让各细化路径在第一步就有反向传播路径。
- 不默认裁剪梯度，也不宣称结构修改可以保证不会梯度爆炸。

总 Transformer 深度仍为 4，但细化模块增加参数和运算，不能把改善直接解释成
“纯几何反馈的独立收益”。这是完整结构候选，与扩大块大小的对照分开。

## 感受野与公平比较

`BLOCK_SIZE` 现在传入 `--block-size`，不再在基础脚本写死 256。
此主干没有使用 `decoder_window`；`blocks_per_batch` 不是注意力窗口。

| 对照 | BLOCK_SIZE | BLOCKS_PER_BATCH | DECODER_REFINEMENT |
|---|---:|---:|---|
| 原主干重跑 | 256 | 32 | none |
| 只扩块 | 512 | 16 | none |
| 扩块 + 逐级细化 | 512 | 16 | progressive |

三组均为每步 8192 点、编码邻居 16、q3、无噪声、随机权重、学习率固定 2e-4、
默认 10000 步、只有 bootstrap 阶段。没有渲染优化或档位优化。

新入口强制使用 512 点 Morton 留出区域。`VALIDATION_BLOCKS=16` 在此代表 16 个区域：
256 版排除 32 个子块，512 版排除 16 个块，留出的原始点集合相同。
完整区域不足时会报错，不退化为训练/验证重叠。
旧实验的留出点不同，不能将旧日志直接当作这个严格对照的基线。

`bootstrap_validation.jsonl` 新增点加权的 `xyz_pooled_world_rmse`。
逐级版还记录 `xyz_stage_pooled_world_rmse`（初始、第1/2/3次细化）。
原有 block common/relative 分解仍依赖 block-size，不应直接跨 256/512 比较其占比。
对比最终 pooled RMSE、固定视角 PSNR/SSIM，以及真实图像，不能仅选最低的单次 loss。

## 服务器后台启动

确认选择的物理 GPU 空闲后执行。下面只启动新候选，不会自动占用多张卡：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

export CUDA_VISIBLE_DEVICES=2  # 按当时的空闲卡修改
OUT="$PWD/output/truck_progressive_b512_q3_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/test_progressive_xyz_q3.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

`Ctrl+C` 退出 tail 不会停止后台训练。使用其他数据路径时设置 `PLY`、`SCENE`。
每 500 步保存 checkpoint；训练结束后自动对所有 checkpoint 生成图像和 PSNR 等指标，
位于同一实验目录的 `render_history/`。不是训练过程中实时渲染。
仅 CPU 重建检查可设 `DEVICE=cpu RENDER_HISTORY=0`。

只扩大块大小：启动前 `export DECODER_REFINEMENT=none BLOCK_SIZE=512`。
原主干匹配重跑：启动前 `export DECODER_REFINEMENT=none BLOCK_SIZE=256`。
两次都调用新入口，分别设置新 OUT；若之前手动设置过 `BLOCKS_PER_BATCH`，
需 `unset BLOCKS_PER_BATCH`，让入口自动按 256→32、512→16 匹配每步点数。
新候选则 `export DECODER_REFINEMENT=progressive BLOCK_SIZE=512`。

## 本地可复现检查

```bash
python scripts/compare_progressive_xyz.py \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --out output/progressive_xyz_context_local \
  --steps 300 --every 100 --regions 64 --batch-regions 2 --threads 4
```

三组顺序运行；相同 64 个 512 点区域中 48 个拟合、16 个留出，
相同 seed、训练点采样与方向采样，每步 1024 点。用于结构运行和初步优化检查，
不等于 10000 步服务器实验，也不能推出真实渲染 PSNR。
日志、逐区域指标、最终 checkpoint 与汇总都保存在同一 OUT 内。
本地 common/relative 分解统一在 512 点区域计算，可跨组比较。

## 本次本地结果（2026-09-22）

上述真实 truck PLY、64 区域、seed42、每组 300 步检查已完成。
以下是预先指定的第 300 步，不挑选最有利的检查点。世界坐标单位：

| 模型 | 拟合 XYZ RMSE | 留出 XYZ RMSE | 留出相对位置 RMSE | 留出总目标 |
|---|---:|---:|---:|---:|
| 256 原主干 | 11.1404 | 11.9660 | 1.1711 | 3.0906 |
| 512 原主干 | 11.1864 | 12.0721 | 1.1125 | 3.1428 |
| 512 逐级细化 | 11.3984 | 12.2413 | 1.6496 | 2.9596 |

逐级版本留出集各层 MSE 为 382.106 → 228.989 → 166.451 → 149.851。
这表明细化路径确实参与优化，但不代表优于基线：初始头也可能把一部分定位责任
转交给后续层。此时总目标较低包含 shape 改善，不能解释成 XYZ 更准确。
512 原主干的相对误差略降，总定位误差并未改善。
本轮没有修改目标来追逐这个短跑结果，候选不替换默认研究基线。

参数量：原主干 1,445,787；逐级版 1,514,256（约 +4.7%）。
三组无裁剪、无非有限值中断。本机只有 CPU，没有真实 CUDA 渲染或显存测试；
300 步也不足以给出收敛结论。结果位于
`output/progressive_xyz_context_local_20260922/`，未把大型输出提交到 Git。

验证：完整回归 234 项（230 通过、4 环境跳过）；追加逐级诊断后新路径 7 项再次通过；
Bash 语法及启动参数干跑通过。测试中的模拟渲染仅验证流程，不是场景质量证据。
