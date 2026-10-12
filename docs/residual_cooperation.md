# Truck 残差增强层协同诊断

独立入口 `scripts/test_residual_cooperation.sh`，读取已有残差解码实验的最终权重，不重新训练编解码器，不引入 detach、冻结分支或新的损失。源目录和 checkpoint 文件只读，诊断产生的更新不保存、不供后续训练使用。默认输入是实验共同目录，也接受其 `residual/` 子目录。

## 检查内容

1. **梯度关系**：同一组训练相机、原 PLY 目标、固定点/符号槽的 AWGN 分别计算 q1/q2/q3 渲染 MSE 梯度。整场景 replay 后按共享编码器、共享解码器、conditioning、基础属性头、第二/第三层输入投影与修正头统计范数、点积和余弦。未参与某档位的参数在固定参数基底上补零；某个模块范数为零时余弦为 null，不伪装成零相关或负相关。新增修正模块不会被误归到编码器。全参数余弦与各模块余弦同时保留。
2. **一次更新影响**：q1/q2/q3 分别提供更新梯度；每一档、每一学习率、每一组相机都从同一份原始权重和全新的 Adam 状态开始。更新后重新检查三档，更新所用相机与留出相机分开记录。两种学习率为 `1e-4` 和 `1e-5`。这不是连续续训，也不是恢复原有 Adam 动量。当前历史 checkpoint 只保存权重与配置，不能据此复现原训练最后一步。
3. **新增信息作用**：固定权重，比较正常 q3、不应用第三层修正、保留第三层计算但清零新增 12 个符号、打乱新增层的逐点发送信号。打乱只发生在同一局部块的有效点之间，前两层前缀和各点噪声保持不变，填充点不参与；少于两点的块无法打乱，数量在 plan 中公开。打乱与清零是异常输入敏感性检查，不独自构成新增符号利用率或因果证据。

位置始终走原有 16bit 可靠坐标路径，不参与修正。所有损失与 PSNR 基准是原 PLY 渲染；没有新增属性 MSE。PSNR 为逐视角 PSNR 的平均，MSE 使用未截断 RGB，SSIM 使用显示 RGB。相同探测组中所有档位、更新前后使用相同噪声，留出相机使用另一组配对噪声。

该噪声方式是原先验证接口的全符号槽配对，而不是按不同前缀长度紧密抽取噪声。每个档位的边际 AWGN 分布不变，但与某次历史训练的具体噪声抽样序列不同。只保留 q>0 的实际有效点，全场景不采样删点、不训练档位表。

## 拉取和后台启动

GPU 2 是示例，按空闲卡调整。`RUN` 在当前终端重新设置，不复用可能为空的旧变量：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_residual_prefix_gpu2_20261012_095255"
OUT="$PWD/output/truck_residual_cooperation_gpu2_$(date +%Y%m%d_%H%M%S)"

# 先做只读预检：打印路径、checkpoint 哈希与测试配置，不加载相机、不建输出目录。
CUDA_VISIBLE_DEVICES=2 bash scripts/test_residual_cooperation.sh "$RUN" "$OUT" --dry-run

# 预检后开始实际 CUDA 渲染诊断。
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_residual_cooperation.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

默认 8 个探测组；每组随机选 2 个训练相机，另用原训练划分的最多 4 个留出相机。相机名称和噪声 seed 都记录。每组做 3 个梯度目标和 6 次独立的一步 Adam 探测，不是 48 步连续训练。它会重复整场景解码和渲染，不能将“8 个探测组”误解为“8 次训练步”；`progress.jsonl` 与终端记录各组实际耗时。

可以先 `PROBES=1` 验证服务器流程，再用默认 8 组新建目录运行。`BLOCKS_PER_BATCH=32` 可降低编解码器激活显存；一次测试内保持分块配置不变。`PROBE_LRS="1e-4 1e-5"`、`VIEWS_PER_PROBE=2`、`HELDOUT_VIEWS=4`、`IMAGE_VIEWS=1` 可调整。场景迁移时用 `MANIFEST=/absolute/path/manifest.json`，实际运行会核验 Truck PLY 指纹、数量和原始相机划分。`--checkpoint` 可指定中间 checkpoint，但路径必须位于本次源实验的 `checkpoints/` 内；解析符号链接后的外部 checkpoint 也会拒绝，避免误用其他实验的权重。来源和哈希会记录。

`--dry-run` 是路径/元数据预检，不验证 CUDA、实际数据文件或相机身份。正式测试若失败，不会生成成功标记；换新的输出目录重试。Ctrl+C 只退出 `tail`，`nohup` 任务不会因电脑断线中断。

## 下载和判断

所有结果在一个新输出目录：

```text
plan.json / complete.json / summary.json
gradients.jsonl / gradients.csv
updates.jsonl / updates.csv
ablations.jsonl / ablations.csv
progress.jsonl
charts/
  gradient_cosines.png
  update_effect_lr0.0001.png
  update_effect_lr1e-05.png
  suffix_controls.png
images/                         # 仅第一个探测组的留出视角、四种第三层条件
scenes/truck/scene.json          # 场景与相机划分身份记录，不复制原 PLY
```

images 的三个面板从左到右是原 PLY 渲染、诊断条件恢复、绝对误差×4。默认每个条件只保存一张，最多四张；不保存修改后的模型或大规模接收符号文件。

- 梯度持续负相关且单步更新反复伤害其他档位：支持跨档位优化折衷；不能仅凭一项负余弦定性。
- 较小步长消除更新后退化：支持步长/非线性效应；留出视角仍退化则另考虑样本适应。
- 正常第三层明显优于清零/打乱：支持它确实利用新增层信息；没有明显差异则需要更细致的依赖检查，不能立刻宣称新增信息完全无用。
- 不应用第三层修正与 q2 应一致，程序会核验配对 MSE；不一致就中止，不能继续解释结果。

只记录、比较与核验，不自动冻结前两层或选择某种“胜出”更新。小规模单 checkpoint 诊断不代表收敛上限；重复噪声使用相同留出相机，不当成独立场景样本。

## 本机验证边界

CPU 测试运行真实编解码器、AWGN、整场景 replay、梯度统计和独立 Adam 更新；只有 CUDA 光栅化及相机加载在测试边界替换为可微合成图像。它能检查代码、配对与恢复机制，不能提供真实 Truck 画质结论。服务器正式入口没有使用合成渲染器。
