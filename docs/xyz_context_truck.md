# Truck：可靠 XYZ 辅助属性解码对照

独立实验入口：`scripts/test_xyz_context_truck.sh`。不覆盖正式多场景主线、残差增强或分层冻结实验。

## 对照边界

两组都从随机权重开始；源多场景目录只提供数据路径、配置及相机划分，不加载历史 codec 权重。

| 设置 | `none` 对照 | `xyz` 实验 |
|---|---|---|
| 属性解码 | 普通前缀直接解码 | 同一解码器，增加可靠 XYZ 上下文 |
| 坐标交付 | 每轴 16bit 量化，delta + zlib 无损压缩量化整数 | 完全相同 |
| 属性预算 | 4 / 12 / 24 个复符号，增量为 4 / 8 / 12 | 完全相同 |
| 编码 | 一次编码，各档截取前缀；增量独立功率归一化 | 完全相同 |
| 信道 | 沿用源运行 SNR / 信道，当前实验源为 10 dB AWGN | 完全相同 |
| 数据 | 仅剪枝后的 Truck；保留全部输入椭球 | 完全相同 |
| 档位训练 | q1 → q2 → q3 → 随机 mixed 循环，不随机丢点 | 完全相同 |
| 初始化 | 相同公共参数及统计 | 新增路径零输出初始化，不扰动公共初始化 RNG |
| 优化 | 完整属性编解码器更新，XYZ 输出头冻结 | 同时更新新增几何模块，XYZ 仍不更新 |

当前源配置：711,691 点；隐藏维度 96；编码端 2 层多尺度栅格上下文（4、8）+ MLP；解码端 2 层局部 Transformer、4 个注意力头、窗口 32；空间块 256 点；每批 64 块；最高 SH 阶数 3；图像分辨率参数 2。

不接分配器、不学习 q0、不做残差增强、前层冻结、步长拒绝或梯度裁剪；无通信惩罚。随机 mixed 不等于学习后的部署档位。

## XYZ 如何参与解码

接收端首先恢复已交付的量化坐标，再送入属性解码器，而非传入源 PLY 的未量化坐标。

1. 逐点路径：`MLP(2 * recovered_unit_XYZ - 1)` 加到接收特征上。
2. 空间关系：每个原有注意力窗口内，对点对的相对 XYZ 和距离用小型 MLP 预测每个头的 attention bias。
3. 坐标仍直接替换最终 XYZ 输出，不学习位置修正，不改变原窗口和分块。

这里的 `unit_XYZ` 使用已交付场景 bbox 做逐轴归一化；距离是归一化坐标空间中的距离，**不是世界坐标欧氏距离或 KNN 图**。新增的是空间条件，不是平均邻居颜色。原逐点残差路径保留。

逐点嵌入和相对偏置的末层权重及 bias 均初始化为零；第一步先更新末层，之后梯度进入前面的层。相同公共权重下，初始输出在数值容差内一致。新增参数和计算量不匹配，不能宣称等参数/等 FLOP 对照；每点业务符号与 XYZ 流不增加，模型参数仍假定提前共享。

q0/填充点不提供几何或注意力上下文。缺少匹配坐标、有效坐标非有限数、尝试混用残差/分配器时明确报错，不回退到源数据。

实际 packet 的接收接口也读取 `xyz.bin` 再进行属性解码；checkpoint 保存开关及新增参数。原开关为 `none` 时不增加参数，并从序列化配置中省略字段，保持原模型/packet 标识语义。

## 训练

每组默认 2000 步属性 bootstrap + 1000 步整场景渲染，共 3000 次实际参数更新；两组顺序运行，共 6000 次。均使用 Adam、固定 `1e-4`，进入渲染阶段清空动量。

- bootstrap：单 Gaussian 局部 RGB 响应 MSE，4 个随机方向、黑白背景，学习全部属性；不监督 XYZ。
- render：恢复场景与输入源 PLY 的完整图像 RGB MSE，每步 2 个训练相机；replay 反传；无属性辅助 loss。
- 原始照片只作为额外评价参考，不作为上述目标。

这里“源 PLY”是此次训练输入的剪枝后场景，不是剪枝前的完整 PLY。两组核验 PLY 指纹、相机划分、公共随机初始化；每更新使用配对随机种子。不是精确续训，不保留其他实验的 Adam 动量。

## 服务器拉取和启动

GPU 2 仅为示例，启动前确认空余 GPU。必须在当前终端定义 RUN 和 OUT。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/multiscene_pruned_aligned_gpu0_20261010_194134"
OUT="$PWD/output/truck_xyz_context_gpu2_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES=2 bash scripts/test_xyz_context_truck.sh "$RUN" "$OUT" --dry-run

CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_xyz_context_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

`Ctrl+C` 退出 tail 不会停止后台训练；SSH 断开也不影响 nohup。两组在一张卡上顺序运行，不同时争用同一张卡。

可覆盖：`BOOTSTRAP_STEPS=2000 RENDER_STEPS=1000 LR=1e-4 VALIDATE_EVERY=250 TEST_TRIALS=2 IMAGE_VIEWS=2`。增加步数会同时增加两组预算。迁移路径时使用 `MANIFEST=/absolute/path/manifest.json`，场景和相机划分仍需一致。`COLOR_ROIS` 可指定预先选定的同一份局部区域文件。

## 结果

所有结果在一个 OUT 内，分组仅是子目录：

```text
plan.json / progress.json / complete.json
comparison.json / comparison.csv
runtime_comparison.json
charts/truck_xyz_context_loss.png
charts/truck_xyz_context.png               # PSNR、颜色偏差验证曲线
charts/truck_xyz_context_ssim.png
charts/truck_xyz_context_final_test.png
none/                                     # 原 direct 解码器
xyz/                                      # XYZ 辅助解码器
    loss.jsonl
    scenes/truck/validation.jsonl
    scenes/truck/validation_images/         # 仅最初与最终，含随机 mixed
    checkpoints/initial/codec.pt
    checkpoints/final/codec.pt
    test/                                  # 独立测试相机，q1/q2/q3
```

验证每 250 步，默认 4 个留出相机 × 2 次固定噪声；最终使用全部独立测试相机 × 2 次噪声。源 PLY 和照片的 MSE/PSNR/SSIM、RGB bias/MAE、通道差分误差均记录；可选 ROI 指标沿用已有实现。`test/prefix_gains.jsonl` 记录实际增强层收益。

重点比较所有 q1/q2/q3 的画质与颜色，不能只看 q2；同时看步耗时和参数开销。跨阶段 loss 尺度不同，不直接比较 bootstrap 与 render 的数值。`complete.json` 只在两组训练、评估和公共初始化核验成功后生成。

本机 CPU 测试使用真实编解码器、量化坐标、信道、packet 和反向传播；仅光栅化边界使用合成可微图像。它能验证执行与梯度一致性，不能证明 Truck 实际画质已提升。
