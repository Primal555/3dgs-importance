# Truck：独立多预算前缀筛选实验

用于比较不同符号预算的恢复效果、重新挑选档位，不覆盖当前三档主实验，也不重新训练语义表示或分配器。

## 固定与变化的部分

- 从同一个已训练完成的 modular checkpoint 复制语义编码器、语义解码器及属性归一化统计，两端语义模块全程冻结，结束后校验权重摘要完全一致。
- 使用源实验的同一个 PLY、空间分组、每批块数、渲染分辨率和相机划分；XYZ 仍通过可靠 16bit 压缩路径交付。
- 两个实验臂都重新随机初始化信道编码器与信道解码器、Adam；不继承原来的三档信道权重。
- 信道编码器一次产生最大长度信号，各档使用其前缀。信道解码器按前缀逐层恢复 `y_hat`，增强层重新读取已收到的完整前缀。源模型已有的逐点去噪、空间邻域修正配置保留，相关权重重新初始化。
- 固定 10 dB AWGN，学习率固定 `1e-4`，只最小化有效点及 latent 维度上的 `mean((y_hat-stop_gradient(y))²)`。不执行渲染反向、颜色专训、语义联合微调或档位分配。渲染、颜色误差、SSIM、LPIPS 都仅用于评估。

| 实验臂 | 累计复符号数／有效 Gaussian | 每档真实更新数 | 总更新数 |
|---|---|---:|---:|
| `three_tier` | 8 / 24 / 48 | 500 | 1500 |
| `fine_tiers` | 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64 | 500 | 4000 |

表中是**累计预算**，例如 24 档总共发送 24 个复符号，不是再额外发 24 个。每臂内部轮换固定预算，500 次为每档独立配额，不是把 500 步分给八档。默认不训练 mixed；仍评估固定随机正档位 mixed，不能把它当作学习分配。`--mixed-updates N` 可额外加入 N 次 mixed 更新，不侵占固定档位配额。真实点全部保留，q0 仅用于填充，无删点。

公共预算 8 / 24 / 48 使用按“预算值＋该预算更新序号”生成的同组数据和噪声种子。评估时按相同逐点虚拟 64 槽产生配对噪声，再截取实际前缀；虚拟槽不发送、不收费。

注意：八档有八级恢复、三档有三级恢复，参数量和运算量不严格相等；各预算还会共享其他预算更新带来的优化。该实验用来查看当前架构下的实际率失真曲线，**不是只改变符号数、严格隔离模型容量的因果对照**。两臂各有自己的共享信道编码器，跨臂不共享同一发送信号。

## 拉取和后台启动

Linux 服务器执行，GPU 3 仅为示例，请确认空闲。`RUN` 必须是已完成的**双编解码器 modular 实验目录**，不是早期单编解码器或多场景目录。默认读取其 `checkpoints/final/codec.pt`。如果只想复用干净语义阶段，可明确指定同一源实验的 `--checkpoint "$RUN/checkpoints/end_semantic_render/codec.pt"`。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_channel_joint_gpu3_20261016_001659"
OUT="$PWD/output/truck_channel_rate_sweep_gpu3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/output"

# 首次使用 LPIPS 时先准备预训练 VGG 权重；不含训练。
python -m gaussian_jscc.perceptual_metrics --device cpu

CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_channel_rate_sweep_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

`nohup` 训练不依赖本地连接；Ctrl+C 退出 tail 不会停止后台训练。两个实验臂按顺序执行，不会同时占两份模型显存。训练更新较快，但默认最终评估全部 test 相机、两次噪声，且八档需逐档评估，不能只用训练步耗时估算总时间。

运行前可打印计划，不创建输出、不加载 CUDA 或 PLY：

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/test_channel_rate_sweep_truck.sh "$RUN" "$OUT" --dry-run
```

可选 `--arms fine_tiers` 只跑八档；`--updates-per-rate 1000` 提高每档更新配额；`--rates 4 8 12 16 24 32 48 64` 改候选累计预算；`--no-lpips` 关闭 LPIPS，但输出不会伪填该指标。`--test-views 8` 仅用于快速诊断；正式筛选建议保留默认全部 test 相机。`--eval-snrs 5 10 15` 仅扩展最终测试，训练仍为 10 dB。

源实验路径迁移后，用 `--manifest /absolute/path/manifest.json` 替换数据路径；PLY 指纹和相机划分仍需一致。

## 日志、结果与续训

所有结果都在一个新 `OUT` 下：

```text
experiment.json                       两臂计划、每档配额、源权重摘要
three_tier/                           三档对照
fine_tiers/                           八档实验
  training.json                       实际结构、源场景、模块参数量
  metrics/loss.jsonl                  每次更新的预算、MSE/NMSE、梯度、模块更新、耗时
  metrics/validation.jsonl            起点、每500总更新、终点验证
  metrics/test.jsonl                  最终留出 test 相机逐视图、逐次噪声结果
  images/initial/                     起点：clean、各档及随机mixed，仅首尾两个验证视图
  images/final/                       同视图终点，无中间轮次图片
  checkpoints/latest/codec.pt         Adam/RNG/步数齐全，用于精确续训
  checkpoints/final/codec.pt          较小的推理权重，不含Adam/RNG
  complete.json                       本实验臂完成标记
comparison/
  rate_distortion.csv                各路径、各实际预算、原PLY/照片的质量与通信成本
  common_budget_comparison.csv       相同8/24/48预算两臂差值
  best_observed_validation.csv        每档已观测最佳画质/最低特征误差及其步数
  quality_vs_symbols.png             PSNR、SSIM、LPIPS对符号数曲线
  latent_vs_symbols.png              MSE、NMSE对符号数曲线
  training_history.png               各预算训练loss与验证PSNR
  summary.json                       汇总、每档更新次数、参数量和比较限制
complete.json                         两臂评估与汇总全部完成
```

质量曲线分别画 10 dB 实线、有限符号无噪声虚线、绕过信道的共享 clean 参考线。PSNR/SSIM 越高越好，LPIPS 越低越好；误差棒是噪声试验间标准差，不是相机置信区间。源 PLY 与真实照片分开统计，不把两者混为同一基准。

最终 CSV 还给出每增加一个符号带来的 PSNR 增益，以及是否被更低预算的 PSNR 支配。**仅 PSNR 被支配不等于所有指标被支配**，仍需检查 SSIM、LPIPS、细节和颜色。最佳验证步只是观测记录，不声称另存了对应最佳模型。最终选择预算依据实际曲线，不强制高档优于低档。

通信成本包括实际累计 payload 复信道使用、实测压缩 XYZ 字节和打包元数据字节；数字侧换算沿用源场景 `net_bits_per_use` 的显式假设。不是把一个复符号当两个 float32 文件字节，也不把模拟载荷冒称为实际 MB；模型假设预共享，未模拟数字侧 FEC/丢包。

只查看结果时，优先下载 `comparison/`、各臂 `metrics/` 和 `images/final/`，无需下载 `latest` 的 Adam/RNG。需要后续推理再下载 `checkpoints/final/codec.pt`。

中断后用**原 RUN、原 OUT、同一组训练/评估参数**续训，脚本自动寻找每臂 latest；已完成臂校验后跳过：

```bash
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_channel_rate_sweep_truck.sh \
  "$RUN" "$OUT" --resume >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

预算、配额、学习率、种子或源模型改变需另建 OUT。续训不会借用旧实验的优化器；只恢复本次实验自己的 Adam/RNG。训练中断最多回退到最近500更新的检查点；不是自动按 Ctrl+C 保存。
