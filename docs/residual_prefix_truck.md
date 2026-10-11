# 实验 3：Truck 属性渐进残差解码

这是独立实验入口 `scripts/test_residual_prefix_truck.sh`，不替换原来的直接解码基线，也不改变实验 1–2 的颜色/预训练方向对照。只测试 Truck，先判断增强层能否带来更明确的增量收益，暂不训练档位分配器。

## 与当前档位实现的区别

原来的 progressive 编码已经是“一次编码，交付不同长度的前缀”，但解码器根据收到的前缀直接预测一套完整属性。q2/q3 并没有显式接着上一档的输出进行修复。

新实验仍然使用 4/12/24 个复符号，分别对应基础层 4 个、第二层额外 8 个、第三层额外 12 个：

```text
一次编码 → 完整 24 符号
              ├─ 前 4 个  → 基础属性 a1
              ├─ 前 12 个 + a1 → 修正量 Δ2 → a2 = a1 + Δ2
              └─ 前 24 个 + a2 → 修正量 Δ3 → a3 = a2 + Δ3
```

“基础属性”包含透明度、尺度、旋转、DC 和 SH，并不是把不同属性分配给不同层。所有累加都发生在归一化属性参数空间；最终仍由 `data.to_raw` 执行原有物理约束，包括旋转归一化、尺度与透明度范围约束。它不是对渲染图像逐层相加。

XYZ 继续通过可靠的 16 bit/轴量化与压缩路径交付，不参与残差修正。冻结坐标输出头及归一化统计，并在完成时核验未变。

## 实际结构及训练

- 保留原来的属性编码器，完整符号只编码一次。当前 progressive 编码不按 q1–q3 重新生成不同表示；保留集合和 SNR 仍会影响编码。
- 基础层复用原有解码主干和属性头，严格只读取前 4 个符号。
- 两个增强层复用同一个解码主干，各自增加上一层属性的线性特征投影和属性修正头。它们不是三个独立 Transformer。
- 每个阶段截断所有邻居的可读前缀。q1 点可以提供基础上下文，但不会拿到 q2/q3 修正；q2 点不会读取邻居的第三层符号。q0 仅用于分块填充。
- 修正头零初始化，初始 q2/q3 等于基础层输出。第一次相关更新先学习修正头，随后梯度才逐渐进入增强符号与前一层特征投影，避免初始化时随机大修正破坏基础输出。
- 不分离上一层的计算图，高档位目标也可以更新基础层。继续循环训练 q1/q2/q3/mixed，保住各前缀的训练机会；但这不构成质量单调提升的数学保证。

训练目标保持不变：先用当前廉价的局部 Gaussian 响应监督做属性 bootstrap，再用原 PLY 的场景渲染 RGB MSE 优化完整属性编解码器。没有新增残差损失、单调性惩罚或人工权重。

两组均从随机权重开始，源实验目录只用于读取数据路径、配置和相机拆分，不加载历史 codec 权重。对照组 `direct` 与实验组 `residual` 使用相同公共初始化、单场景统计、批次、逐更新随机种子与训练相机采样；额外修正模块初始化不会挤占公共初始化后的随机序列。程序核验初始公共权重签名一致。

两组都保留全部点，mixed 只混合 q1–q3、不随机丢点。没有 q0 学习、档位表或联合分配微调，避免把删点收益混入增强层对照。新模式目前明确拒绝分配器软梯度接口，不能直接拿去替换完整 mask 流程。

额外参数和计算量并不匹配：q1 一次主干解码，q2 两次，q3 三次；mixed 根据出现的最高档位执行必要阶段。总场景训练时间还包含光栅化、通信与 replay，不能直接断言总时间变为三倍。`runtime_comparison.json` 会记录参数量、实际更新数及各阶段平均耗时。这是结构效果对照，不是等参数或等 FLOP 对照。

## 服务器拉取和启动

GPU 2 只是示例，启动前换成实际空闲卡号。默认每组 2000 步属性 bootstrap + 1000 步场景渲染，两组在同一张卡顺序运行，学习率固定 `1e-4`，不是两组一起抢卡。该预算用于观测趋势，不宣称足够收敛。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/multiscene_pruned_aligned_gpu0_20261010_194134"
OUT="$PWD/output/truck_residual_prefix_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_residual_prefix_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

必须在当前终端定义 `RUN`，不能沿用另一个终端的空变量。`Ctrl+C` 只退出 `tail`；关闭 SSH 不会中止 `nohup` 后台训练。

可用 `BOOTSTRAP_STEPS=5000 RENDER_STEPS=5000` 增加预算，`VALIDATE_EVERY=250 LR=1e-4` 调整验证频率与学习率。`DECODER_GROUPS=residual` 可以只跑实验组，但此时不声称完成两组配对对照。`MANIFEST=/absolute/path/manifest.json` 可覆盖迁移后的数据路径，必须仍指向同一份剪枝后 Truck PLY 和相机拆分。

启动前可去掉 `nohup` 和输出重定向，在命令末尾加 `--dry-run`，检查路径和实际命令；不会创建输出目录或训练。

## 观察和下载结果

所有结果保存在一个共同目录，组别是其子目录：

```text
plan.json / progress.json / complete.json
comparison.json / comparison.csv
prefix_gains.json / prefix_gains.csv
runtime_comparison.json
charts/                            # 两组共图的 loss、验证、最终测试和增强层收益
direct/                            # 直接解码对照
residual/                          # 累加修正实验
    stage_diagnostics.json
```

各组 `loss.jsonl` 和 `scenes/truck/validation.jsonl` 保留数值历史。每 250 步记录验证数值，训练图像只保存最初和最终两轮、每轮两个代表视角；包含随机 mixed 图片，但不是学习分配后的部署图。最终固定前缀测试位于各组 `test/`，默认全部独立测试相机、两次噪声试验，只保存两个代表视角的 q1/q2/q3 图像。

重点比较：

1. 原 PLY 与真实照片两套 PSNR/SSIM/MSE，颜色偏差诊断也沿用已有指标。
2. q1→q2、q2→q3 的配对画质收益，以及每额外 8/12 个复符号带来的 MSE 降低。
3. 有多少配对测试视角/噪声试验真的改善，而不是只看平均 PSNR。
4. 基础层是否退化、细节是否改善，以及额外参数和耗时是否值得。

`stage_diagnostics.json` 在同一次 q3 接收下检查全场景三层的归一化属性误差和实际修正量 RMS，确认增强层确实在工作。它发生在物理约束之前，使用一个完整符号噪声样本，不做可见性加权；只是诊断，不是新增 loss，也不能代替真实渲染质量。

优先下载共同目录的对照表、图表和各组数值/图像日志，不必下载所有 checkpoint。总 `complete.json` 只在全部训练、评估及配对/冻结检查成功后写出。

本机 CPU 测试使用真实编解码器、通信噪声、反向传播与 checkpoint/传输流程；集成测试的光栅化边界使用可微合成函数。它只能证明代码流程及梯度正确，不能证明 Truck 的实际 CUDA 渲染画质已提高。
