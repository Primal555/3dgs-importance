# 逐层重新读取接收符号的 Transformer 解码器

在不加定位 token 的 Transformer 主干上进行独立实验。旧入口和旧 checkpoint 行为保持不变。
新入口：`scripts/test_received_memory_q3.sh`；配置：`--decoder-memory received`。

## 实际实现

```text
接收符号 + 符号 mask + 档位/SNR条件
  -> 原有输入 MLP -> 接收嵌入 memory（本次前向内不更新）
                         |
初始逐点状态 = memory    |
  -> 第1层：向 memory 做 cross-attention -> 原 self-attention + FFN
  -> 第2层：向 memory 做 cross-attention -> 原 self-attention + FFN
  -> 第3层：向 memory 做 cross-attention -> 原 self-attention + FFN
  -> 第4层：向 memory 做 cross-attention -> 原 self-attention + FFN
  -> 原第1/2/4层 XYZ 读出，以及末层 opacity/logcov/DC/SH 读出
```

每层读取模块有独立参数：查询是当前逐点状态，key/value 是同一份初始接收嵌入，使用独立的查询/记忆 LayerNorm。
memory 不被后续层覆盖，但**不 detach**，所有读取路径都能反传到输入 MLP 和编码器。
没有重新运行信道，没有第二次噪声采样，也不是多次发送同一内容；32复符号预算不变。
新分支没有共享定位 token、块中心、平移输出、源 XYZ、源邻居或新的空间旁路。

读取模块在原模块初始化后创建，各层注意力输出投影权重/偏置初始化为零：

- 同种子旧/新模型的共享权重、初始输出、第一次共享参数梯度严格一致。
- 第一更新步只有读取模块的输出投影能收到非零梯度，其前面的 Q/K/V 随后开始学习。这是有意的零启动，不是断梯度。
- q0、padding、全空块与空序列安全屏蔽；保留块内重排等变。
- 新旧结构分别随机训练，不默认加载旧 checkpoint。显式跨 memory 配置初始化会报错，避免默默漏载参数。

不修改编码器、logcov、`spatial_logcov_v1`、学习率、符号档位或训练阶段。
服务器实验仍为第一阶段、固定q3、无噪声、LR=2e-4、不裁剪；不是仅训练位置。
默认 `decoder_memory=none`，该默认字段不写入旧模型配置哈希；已有纯主干训练脚本不自动改用新实验。

## 成本及依据

默认模型新增4个cross-attention模块，增加150,528个参数；总参数从1,445,787变为1,596,315（约+10.4%）。
注意力仍仅在256点块内，新增读取为平方复杂度，**会增加计算量和显存**；不能把通信用量不变理解为计算成本不变。
原有共享 bbox/特征统计协议不变。

借鉴的是 [iLRM 的每层读取输入信息、再更新重建状态](https://arxiv.org/html/2507.23277v1) 的结构原则，
不是论文复现：iLRM 使用图像和相机射线，我们只有接收符号；我们不新增视角嵌入、不生成额外高斯，也不照搬其损失。
“深层处理中可能丢失定位信息”是动机假设，不是已确定的根因；读取同一份信息不能弥补编码器根本没有传入的信息。

## 服务器启动

确认GPU空闲、激活maskgs并更新代码后：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
OUT="$PWD/output/truck_received_memory_q3_$(date +%Y%m%d_%H%M%S)"
# GPU=2只是示例，请先确认空闲。
CUDA_VISIBLE_DEVICES=2 BOOTSTRAP_STEPS=10000 SAVE_EVERY=500 \
  BLOCKS_PER_BATCH=32 LR=0.0002 RENDER_HISTORY=1 \
  nohup bash scripts/test_received_memory_q3.sh "$OUT" > "${OUT}.log" 2>&1 < /dev/null &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

默认4层主干；`DECODER_DEPTH`可显式修改。每500步保存，训练结束后在同目录`render_history/`生成历史图和PSNR，非实时渲染。
断开SSH不终止nohup；Ctrl+C只退出tail。不要复用已有输出目录。

## 本机匹配对照

```bash
python scripts/compare_received_memory.py \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --out output/received_memory_comparison_new \
  --steps 2000 --every 200 --save-every 500 \
  --blocks 64 --blocks-per-batch 4 --threads 4 --seeds 42 43 --workers 2
```

64个256点块，48拟合/16留出，分层取全场景Morton块索引区间中点；两个种子，每模型2000步。
新旧模型相同采样顺序、随机监督方向、目标、初始共享权重，末三次检查点（1600/1800/2000）汇总，不挑最好一次。
全PLY归一化统计共享，留出块不参与梯度，这是同场景诊断，不是跨场景泛化测试。
记录每步梯度及各层memory梯度、逐块误差、共同/相对误差、形状误差和点云展开比例，保存模型及曲线。

可选 `--reference-root` 复用已完成的**纯主干**对照：核对PLY哈希、配置、分块、种子、步数、批量等协议，复制结果并记录来源和哈希；不复用候选模型。
本次复用 `output/localization_large_paired_20260921` 中的两个 `none` 对照，绝不将失败的定位token模型当作基线。
首次启动因JSON列表/内存元组比较差异被协议检查拒绝，未开始训练；修正序列表示比较后，正式结果使用 `output/received_memory_paired_20260922/`。

本机没有CUDA，CPU对照不是场景渲染PSNR；功能测试中的模拟渲染数据也不是实际画质结果。

## 2026-09-22 已完成的2000步对照

两组新模型均完成2000步，约18分钟（两个种子并行）；与前述匹配基线比较。
以下取1600/1800/2000步的MSE平均再开根号，单位为原场景坐标单位。

| 指标（越低越好） | seed42 纯主干 | seed42 逐层重读 | seed43 纯主干 | seed43 逐层重读 |
|---|---:|---:|---:|---:|
| 拟合 XYZ RMSE | 1.6325 | 2.1582 | 1.6696 | 1.9990 |
| 留出 XYZ RMSE | 6.1353 | 5.8815 | 6.1973 | 6.7003 |
| 留出共同偏移 RMSE | 6.0576 | 5.8074 | 6.1171 | 6.6249 |
| 留出相对位置 RMSE | 0.9735 | 0.9309 | 0.9942 | 1.0019 |
| 留出 shape 项 | 0.2212 | 0.2826 | 0.2809 | 0.2599 |
| 留出总目标 | 1.4253 | 1.4166 | 1.4144 | 1.5136 |
| 最后500步梯度范数中位数 | 82.71 | 90.73 | 85.32 | 81.39 |

**结论：实现可用，但未形成跨种子的稳定整体收益，不替换纯主干基线。**

- 拟合位置误差增加约32.2% / 19.7%；留出位置误差一个种子降低4.1%，另一个增加8.1%。
- 留出逐块平均MSE仅6/16、4/16个块改善；逐块RMSE中位数分别从3.645/3.841变成3.819/4.032。seed42总RMSE的小幅下降不代表多数块变好。
- 最后单次检查seed42拟合RMSE曾降到1.1749，但1800步为3.1853；不能挑2000步就宣称稳定改善。
- 四层新增读取模块最后一步均有非零梯度；全程无裁剪、未出现NaN/Inf。总体梯度量级接近旧版，不能宣称解决梯度爆炸。
- 对“没有稳定收益”的解释仍有限：不能单凭这组结果证明重复读取一定无用，也不能证明编码器或位置表示没有问题。不在本轮继续叠加损失/结构以追逐某个检查点。

本地结果和模型保存在 `output/received_memory_paired_20260922/`；`comparison.json`、六张曲线、逐步/逐块日志、参考来源及checkpoint统一在该目录下。
原`test_transformer_trunk_q3.sh`保持不启用memory的行为。新入口仅供独立实验，不把它称为已经验证的质量升级。

验证：完整回归227项（223通过、4跳过）；增加诊断报告字段后，新路径12项和旧主干11项再次通过。Bash语法及`git diff --check`通过。
