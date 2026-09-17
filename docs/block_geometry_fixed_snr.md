# 固定 10 dB 的块级几何编解码

> 本文记录 v4 历史设计。已发现其填充槽功率浪费；当前脚本使用 v5，
> 请参阅 [v5 修正与本机验证](pilot_geometry_v5.md)。不要将下面的 v4 说明
> 当作当前发送格式。

## 修改目的

`block_relative_v4` 不再依靠 `LN(hidden) -> 小权重 affine` 从中位数附近重新
建立绝对位置。它在已有几何符号中传输块参考和局部偏移，使用确定性的恢复通路
加可学习残差。未训练时，无噪声下即应恢复整个 bbox 范围；训练集中学习噪声下的
几何编码修正与局部联合去噪。它是新的工程设计，不是对某篇论文方法的复现。

## 位置载荷与功率

输入仍是现有全局 bbox 归一化坐标。每个原有 Morton 块的保留节点计算中心 c
和一个各向同性半径 s=max(abs(x-c))，s 下限 1e-4。局部坐标 r=(x-c)/s。

每个 Gaussian 的几何实数槽位：

- 0..2：2c-1；3：映射到 [-1,1] 的 log(s)。在块内重复传输、接收端聚合。
- 4..6：三个局部坐标；8..末尾：循环重复局部坐标，叠加学习到的有界编码修正。
- 7：功率补全分量。接收端不将它作为位置信息使用。

几何预算仍为 4/8/16 **复符号**；总预算仍为 8/16/32。参考信息与功率补全
消耗现有预算，**不免费、不是额外 metadata**。最低档需要至少 4 个几何复符号。
这种结构付出了参考重复和功率补全的代价，不能视作已经最优的资源分配。

若本档几何长度为 g，d 为除第 7 槽之外的 2g-1 个实数值，每个值绝对值不超过
1.25，使用已知增益 a=0.9*sqrt(g/(2g-1))/1.25，并补全
sqrt(g-sum((a*d)^2))。每行几何总能量严格为 g。接收端仅用档位即可得到 a。
属性部分独立按块归一化；不再用几何和属性混合的未知增益缩放坐标。
同一帧中不同档位不能简单截取已编码的高档信号：掩码长度仍为前缀，但编码增益
和补全值依赖实际档位，应按选定档位重新编码。

接收端仅接收符号、档位、既有全局 bbox/分块元数据、共享权重和 SNR。它从接收
信号聚合中心和半径，再恢复局部位置；从不调用发送端参考计算或读取源 PLY。
属性网格仍在恢复的 XYZ 上建立。没有新增块中心/半径字段。

当前学习残差不改动参考槽位；参考的抗噪声依赖块内重复聚合。块整体参考误差、
Morton 块跨度和离群点仍可能限制质量。低 SNR、稀疏小块、极端大块不保证高精度。

## 训练与兼容性

脚本固定 AWGN 10 dB，逐步重采样噪声。每步同一个随机块依次 q1/q2/q3，平均梯度。
只更新 `block_geometry.encoder/decoder`，其余权重、归一化统计均冻结。
旧几何网络参数保留以便审查迁移，但不参与新几何路径；没有沿用旧的位置预测。
属性权重虽保留，接收功率与预测上下文改变，**属性结果不是函数等价迁移**。

目标明确为：

`mean SmoothL1((xyz_pred-xyz_true)/source_block_radius, beta=.01)`

`+ mean SmoothL1(xyz_pred-xyz_true, beta=.001)`。

前者约束局部精度，后者约束全局放置；监督半径只用于训练损失。每项系数为 1，
是可审查的工程选择，不宣称最优。局部半径下限限制放大，分支裁剪仍是保护措施。
这里的训练日志 profile 是 `block_local_v4`；checkpoint 的 `position_v3` profile
保留给完整属性/渲染训练及通用重建评估，不是本次几何训练标量目标。

每 100 步默认固定 16 个源场景块、固定噪声种子，评估无噪声和 AWGN 10 dB。
按三档平均的**未裁剪 XYZ RMSE**选择 `codec_best.pt`，从 step 0 开始选。
连续 8 次验证不改善则结束，3000 步只是上限；没有自动进入渲染阶段。
这是固定源场景诊断/模型选择，不是独立测试集或跨场景泛化结论。

旧 checkpoint/packet 的路径与哈希保持兼容。新格式由 config 的 head 名称区分，
旧版软件不能读新 checkpoint。现阶段仅支持固定硬档位（含 q0 丢弃/填充），
遇到可学习 mask 的 straight-through choices 会明确报错，不能假装已支持联合优化。

## 服务器运行

激活 maskgs，确保文件已经同步后运行：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
INIT="$PWD/output/truck_position_v3_short_gpu3_20260915_175019/codec.pt"
OUT="$PWD/output/truck_block_geometry_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_codec_fixed_geometry.sh \
  "$INIT" "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

GPU 0 只是示例，必须换成当时获准使用的空闲卡。此次不加载相机、不做渲染反传，
因此 `blocks-per-batch 32` 不影响这段逐块几何训练，只为后续全场景接口保留。

主要查看 `position_evaluation.json`、`position_selection.json`、`loss.jsonl` 和
`charts/training_fixed_positions.png`、`charts/training_gradient_groups.png`。
`codec.pt` 是最后权重；评估优先用 `codec_best.pt`，它可能来自 step 0。
源码和训练配置变更意味着不同 profile 的总 loss 不能直接比较。

可用原有独立 codec benchmark 对 best checkpoint 做固定 10 dB 渲染评估；
它会使用完整属性解码，不会把源属性偷偷当作正常输出。几何冻结训练并不保证
完整属性/渲染指标同步变好，正式进入联合微调前必须确认这一点。

## 本地验证

```bash
python -m unittest discover -s tests -p test_block_geometry.py -v
python -m unittest discover -s tests -v
```

覆盖全范围无噪声恢复、真实预算/功率、混合档位、q0/短块、独立接收端、冻结权重、
明确迁移、完整渲染 replay 梯度一致性、固定 SNR 和 best/early-stop 记录。
CPU 测试不能替代服务器上 10 dB 的实际恢复质量与 CUDA 渲染验证。
