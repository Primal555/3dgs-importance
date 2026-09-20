# 几何 / 外观分流编解码器实验

本实验通过 `--architecture learned_split` 显式选择。默认 `learned_joint`
保持不变，旧权重仍可用于对照；不自动迁移旧权重到新结构。
它是可训练的结构实验，不是已经证明超过 12.6 dB 的方案。

## 结构与通信边界

发送端：

```text
XYZ + log-scale + quaternion → 几何 MLP → 几何局部 Transformer ──┐
                                          ↓ 特征交换           │
alpha + DC/SH              → 外观 MLP → 外观局部 Transformer ──┤
                                                             ↓
                                       合并特征 → 一个 symbol head
                                                             ↓
                              每 Gaussian 按 q 取前缀 → 功率归一化 → 信道
```

接收端：

```text
接收符号 + 有效符号 mask + q + SNR + 块内顺序
                    ├→ 几何投影 → 局部注意力 → 几何 MLP → XYZ / scale / rotation
                    │                              ↓ 几何隐特征
                    └→ 外观投影 → 局部注意力 + 特征交换 → 外观 MLP → alpha / DC / SH
```

- 发送端使用 Morton 顺序窗口，默认 32 个点；两层使用普通 / 平移 16 槽的窗口。
  不是 kNN，不保证窗口中的所有点都是欧氏距离最近邻。窗口不跨 256 点块。
- 发送端注意力偏置由窗口内真实相对 XYZ 经 MLP 得到。使用窗口内保留点的最大轴跨度
  归一化，只在发送端计算，不作为坐标辅助流发送。绝对 XYZ 仍进入逐点几何 MLP。
- 两条路径都是 pre-norm 残差注意力，保留逐点特征。几何特征可送入外观路径，
  几何路径不在接收端读取外观路径输出；最终通信表示仍然共享。
- 接收端注意力使用接收特征和确定的顺序窗口，不根据尚不可靠的预测 XYZ 建图。
- 没有手工参考信号、粗坐标旁路、坐标重复观测或固定几何符号子预算。
  q0 不发送该点载荷；q1/q2/q3 分别使用 8/16/32 个复符号。块内允许混合档位。
- q0 不参与发送端相对坐标归一化和注意力；接收端屏蔽 q0 key，输出对应行归零。
  尾部填充、全 q0 窗口、全 q0 块均有测试。
- 和已有链路相同，模型权重、全局归一化信息和档位/分块语法仍遵循已有共享/可靠元数据假设。
  “不额外发送坐标”不等于没有任何元数据成本。
- 分流不等于梯度完全隔离：外观损失仍可通过几何隐特征交换和共享符号回传到几何路径。
  没有隐式 detach、裁剪或新几何预算。

默认 hidden=96、depth=2、SH=3 时参数量为 865899；旧结构为 311291。
这不是等参数量对照。注意力为窗口规模，不能据此沿用旧结构的实测显存/耗时结论。

## 训练约束和观测

本次只变结构。实验脚本保持：随机权重、spatial_response_v3、fine_weight=1、
固定 10 dB AWGN、LR=2e-4、无梯度裁剪、5000 步 bootstrap、无 render/joint 优化。
沿用该损失是为了与上一轮结构对照，不表示它已经被证明是最优目标。
保存每 500 步权重；每 100 步记录固定局部块验证。

`loss.jsonl` 的 gradient_groups / updates 增加几何和外观 encoder/decoder、
encoder_exchange / decoder_exchange 统计，同时保留 xyz_head、attribute_heads 等。
这些是按参数组计算的梯度，不是各损失对共享参数的梯度冲突测量。

训练结束后统一运行只读 checkpoint 渲染，每 500 步生成 PSNR/SSIM/MSE 和图像；
不是训练中每 500 步立即显示，也没有启用第二阶段训练。
默认 8 个固定 test 视角、两次噪声试验、resolution=4，方便与上一轮比较。
图像是一次噪声试验的展示，指标汇总两次。前期随机输出仍可能产生大 footprint，
resolution=4 不保证一定不会 OOM；脚本不会自动改协议后继续。

## 服务器启动

下面 GPU=2 仅为示例，先确认该卡空闲。输出目录必须全新。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main || exit 1
OUT="$PWD/output/truck_split_codec_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/output"
CUDA_VISIBLE_DEVICES=2 PYTHON_BIN="$(command -v python)" \
BOOTSTRAP_STEPS=5000 BLOCKS_PER_BATCH=32 LR=2e-4 \
nohup bash scripts/test_split_codec_bootstrap.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

若数据不在默认路径，显式设置 `PLY`、`SCENE`。可用 `RENDER_HISTORY=0` 只训练，
随后用 `evaluate_bootstrap_history.py` 单独评估。评估开启时，步数须是 SAVE_EVERY 的正整数倍。
脚本不会继承 INIT 或 POSITION_DELIVERY 来加载失败权重/开启坐标旁路。

输出位置：

- `$OUT/training.json`：确认 `codec_config.architecture=learned_split`、`init=null`。
- `$OUT/loss.jsonl`、`bootstrap_validation.jsonl`、`charts/`：训练与参数组梯度。
- `$OUT/render_history/quality_vs_step.png`、`metrics.csv`、`results.json`：渲染曲线与明细。
- `$OUT/render_history/validation_images/000500/3_view00.png` 等：photo / source PLY / received / error。

训练权重、结构化日志和评估现在都位于同一个实验目录。离线评估省略 `--out` 时，
也默认使用 `--training` 目录下的 `render_history/`；不会覆盖已有评估。
同一模型若需不同评估条件，可显式指定 `$OUT/render_history_noiseless` 等子目录。
之前下载的同级 `_render_history` 目录不会自动移动、删除或覆盖。

## 本机能力检查与结果边界

```bash
python diagnose_split_codec.py --ply "$PLY" --out output/split_fit_check \
  --steps 300 --sample-blocks 4 --device cpu
```

默认从真实 PLY 均匀抽取 4 个完整的 256 点块，交替 2 块用于训练、2 块用于验证；
无噪声训练，循环 q1/q2/q3/mixed，固定三个正交方向进行 spatial_response_v3 监督。
评价时同时记录无噪声和 10 dB AWGN，条件 SNR 输入始终为 10 dB。
该脚本不加载/保存 checkpoint，不渲染场景；配置、逐步梯度和更新、前后指标写入日志。
`--train-channel awgn` 可检查有噪声训练；`--architecture learned_joint` 可运行旧结构对照。

2026-09-20 本机 Truck，默认规模、seed=42、300 步、无裁剪：

| 指标 | learned_joint | learned_split |
|---|---:|---:|
| q3 拟合块 XYZ RMSE：开始 → 结束，无噪声 | 53.736 → 5.531 | 34.714 → 6.938 |
| q3 保留块 XYZ RMSE：开始 → 结束，无噪声 | 76.189 → 15.719 | 39.461 → 16.651 |
| 参数梯度范数：中位数 / 最大值 | 137.01 / 278.21 | 46.36 / 101.42 |

不同结构初始化不相同，参数量也不同。新结构可获得有效梯度并降低损失，但这个短测
没有显示最终位置误差优于旧结构，保留块形状也没有得到改善，不能声称解决了精度或梯度爆炸。
拟合块的位置误差仍很大。以上只支持进行有观测的实验，不支持直接延长到 20000 步。
本机没有 CUDA 渲染条件，新结构的 PSNR 收益、服务器显存与吞吐仍待实测。

检查覆盖：混合档位、发送功率/符号计数、独立接收端、q0 无泄漏、相对几何注意力、
各分支梯度、SH3、checkpoint/packet 往返、replay 一致性、离散 mask 联合梯度、
CLI 随机初始化和架构不匹配拒绝、实际 Bash 参数构造。
本轮完整测试：111 项，108 通过、3 项 CUDA 相关检查因本机条件跳过。

## Truck 20000 步服务器结果复核（173021 实验）

训练记录确认：随机初始化 learned_split、spatial_response_v3、fine_weight=1、
固定 10 dB AWGN、LR=2e-4、无裁剪；仅 20000 步 bootstrap，render_steps=joint_steps=0。
离线渲染并没有继续优化模型，不应把这些曲线称为渲染目标训练曲线。

- 最终 q3：source PSNR=13.269 dB、SSIM=0.250、全场景 XYZ RMSE=0.723。
  上一轮相同评估协议的共享结构对应 12.231 / 0.308 / 0.591。
  新结构 PSNR 改善不等于结构恢复全面改善；不是等参数量实验。
- 总梯度范数中位数 / 95 分位 / 最大值为 37.646 / 62.469 / 97.024。
  上一轮对应中位数约 225、最大值约 346；本轮未见持续增长的爆炸模式。
- 最后 1000 步，几何 decoder / 外观 decoder 的梯度范数中位数约 10.891 / 0.0039，
  但 Adam 相对参数更新约 3.39e-4 / 4.49e-4。不能把梯度范数差异解释为外观分支不更新。
- 固定验证块 q3：形状损失在 5000→20000 步仅 0.704→0.667；最大轴尺度比的
  块中位数均值约 0.384，位置误差/原始最大轴半径的块中位数均值约 22.94。
  这里不是全场景中位数，也不是球化程度指标。

额外只读 CPU 检查：从 Morton 排序场景中均匀抽取 16 个完整 256 点块，
固定 seed=142、q3、SNR 条件输入始终 10 dB，总共 4096 点：

| 指标 | 数值 |
|---|---:|
| 无噪声 XYZ RMSE（抽样点整体统计） | 3.136 |
| AWGN XYZ RMSE（抽样点整体统计） | 3.158 |
| 原始最长/最短轴比值中位数 | 44.428 |
| 解码最长/最短轴比值中位数，AWGN | 1.662 |
| 解码/原始最大轴比值中位数，AWGN | 0.394 |
| 位置距离小于原始最大轴半径的点比例，AWGN | 0.49% |

抽样 RMSE 受背景/离群点影响，不能直接和全场景或按块平均的日志数值混比。
这些样本显著趋于球形，不能从全场景渲染图单独确认的形状问题在参数层面得到了支持。

用三个正交方向检查原生形状损失，原解码为 0.620；只替换原始尺度为 0.970，
只替换原始旋转为 0.609，同时替换两者接近零。
这是有 teacher 的损失诊断，不是合法通信方案，也不是实际渲染提升证据。
尺度/旋转的配对和轴置换等价性意味着不能把单项替换解释为纯粹的旋转因果结论。
它支持联合检查协方差形状恢复，不支持直接在部署时统一放大所有椭球。
检查前后 checkpoint SHA256 一致，没有重新训练或改动权重。
