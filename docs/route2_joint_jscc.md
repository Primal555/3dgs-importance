# 四档 mask 与 codec 联合训练

联合训练已统一到 `train-learned --joint-steps N`。`train-route2` 及旧的 Gumbel-ST 训练循环已移除。

## 当前含义

- q0：不发送、不渲染；不构造假位置参与反向传播。
- q1/q2/q3：每个 Gaussian 独立的不同复符号预算；不代表缩小透明度。
- 同一局部块可以混合多个档位；局部块不是档位分配单位。
- mask 仍是按原始 PLY 行索引的四类可学习 logits，不是 Transformer 注意力权重。

可通过 `--existence-prior prior.npy` 初始化存在概率。文件必须与原始 PLY 行顺序一致。

## 梯度与目标

codec 对实际采样档位的多视角图像 MSE 正常反向传播，不再加入参数重建或逐点投影辅助损失。
mask 使用硬采样的 REINFORCE 梯度、独立样本的 leave-one-out 基线，以及期望 payload 码率的解析梯度。
mask 的任务奖励同样来自完整原场景渲染的图像误差；删点不能通过移除某一行的参数误差获得虚假收益。

`--mask-samples` 默认 2；样本数增加会增加全场景开销。场景级策略梯度的方差仍需观察，不能仅凭梯度非零认定重要性学习有效。
`--beta` 是期望 payload 开销的拉格朗日权重，不是严格的总符号数上限。

完整连续训练：

```bash
JOINT_STEPS=1000 CUDA_VISIBLE_DEVICES=0 bash scripts/train_codec_learned.sh
```

仅从已训练的新 codec 开始联合优化：

```bash
python -m gaussian_jscc train-learned \
  --ply /path/to/point_cloud.ply --source /path/to/scene \
  --init /path/to/learned_codec.pt --out /path/to/new_joint_output \
  --steps 0 --render-steps 0 --joint-steps 1000 --device cuda
```

上面使用的只能是 learned_joint 检查点。`--allocation-init` 可恢复与初始化 codec 匹配的 `route2.pt`；优化器和步数计划重新开始，不是精确断点续训。

结果包含配对的 `codec.pt` / `route2.pt`，以及配对的最佳联合检查点。不得跨次保存混用。

架构、验证图及预算假设见 [主说明](learned_joint_jscc.md)。
