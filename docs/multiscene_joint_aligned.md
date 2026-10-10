# 部署对齐与固定前缀保持：两项同时启用

从原实验 `checkpoints/end_allocation/codec.pt` 和同目录的逐场景分配表开始，
只训练联合阶段，默认每个训练场景 1000 次更新。不是重新随机初始化，
也不是继续旧 final。原检查点没有 Adam 状态，本次明确重新创建优化器。

## 修改内容

- 训练和部署都抽十次：十次全 q0 才删除，否则选择正档位众数；并列按
  学到的概率、再按随机数决定。训练每步重新抽样，部署保持固定种子。
- 反向仍是有偏的局部代理，不宣称离散选档或压缩字节可精确求导。
  q0 全量反事实渲染反馈保留，未发送点不进入有效属性上下文。
- 通信惩罚按十次决策后的分布计算；286 种计数模式精确累加，分块立即
  反向，避免保存全场景的大计算图。坐标与档位压缩的实测成本仍用于评估。
- 每四次联合更新，额外用同一批训练相机执行一次全 q1/q2/q3 渲染，循环
  三个前缀。该次编解码器梯度取两种布局 MSE 的平均；分配器梯度不减半。
  每步仍只有一次编解码器更新和一次分配器更新，不减少学习分配的次数。
- 保留当前代码的全 q3 **payload + XYZ + 档位代理总成本/原始点数**归一化；
  它不是固定除以 24。本轮不同时更改 beta 的实际强度，日志记录归一化值。

## 启动

```bash
conda activate maskgs
INIT="$PWD/output/multiscene_full_gpu3_20261008_181748/checkpoints/end_allocation"
OUT="$PWD/output/multiscene_joint_aligned_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/train_multiscene_joint_aligned.sh \
  "$INIT" "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

GPU 编号请改成实际空闲卡。先完成代码同步，不能直接运行旧服务器脚本。
可用 `JOINT_STEPS=50 VALIDATE_EVERY=25` 做启动检查；正式比较保持默认
1000 次/场景、3 次噪声、全部测试相机，三种置乱种子不变。

## 保存与解释

所有内容在一个 OUT 内。训练的中间验证只保存指标，不存渲染图；
`evaluation/initial` 与 `evaluation/final` 只保存四个代表视角的
`mask`、全 q1、全 q3、`mixed` 面板。两轮采用相同视角和噪声协议。

`mixed` **值得保存**：它是随机混合档位对照，不是学习分配后的部署。
`mask` 才是十次采样得到的学习部署，图片标题区分两者。q2 和所有置乱
仍完整评估，只是不默认保存面板。mixed 的完整指标保存在各场景的
`validation.jsonl`；它不冒充学习策略进入部署成本排行。

默认 compact 只输出 PNG 图表、不保存完整概率数组，不复制初始化与
end_joint 检查点；保留 latest、best_joint、final 匹配权重/表在服务器。
结束后生成 `review.zip`，包括日志指标、图表、初始/最终图片与配置，
排除权重、大数组和 PLY。只下载它即可复查本轮；需要额外推理再取检查点。
不删除已有 output。`OUTPUT_PROFILE=full` 可保留完整数组和 PNG/PDF。

日志中 `loss` 是学习布局 MSE 加通信惩罚；发生前缀保持时，实际编解码器
图像目标另见 `codec_image_objective`。比较部署画质、总通信成本、固定
前缀收益和额外耗时，不只比较单一总 loss。CPU 合成测试验证实现，
真实场景质量是否改善仍需服务器 CUDA 实验。
