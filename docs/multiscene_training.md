# 多场景共享编解码器：训练、部署与实验

## 固定研究口径

- 默认训练 Truck、Train、Playroom；DrJohnson 留出。清单位于 `configs/multiscene_tandt_db.json`，路径相对仓库根目录，可直接迁移 Linux。可用 `--train-scenes truck` 做单场景对照。
- 四个输入使用官方 30000 步完整 PLY。不是以前约 88 万点的剪枝 Truck；训练速度、通信总量和既往结果不能直接混算。
- 共享属性表示编码器、JSCC 符号映射、解码器，以及属性均值/标准差。每个训练场景等权计算一、二阶矩；留出场景不参与统计。
- 逐场景维护 bbox、Morton 排列与分组、可靠坐标流、分层 keep/tier 概率表和其 Adam 状态。概率表不是跨场景泛化网络。
- 坐标仍为每轴 16 bit 量化后 delta/zlib 无损压缩。**无损指整数压缩，不指原始浮点坐标完全无损**。默认 q0/q1/q2/q3 = 0/4/12/24 个复符号，沿用现有一次编码、分层功率归一化、按前缀交付结构。
- 不改变原有损失：属性 bootstrap 使用 isolated local-response RGB MSE；render 使用原 PLY 渲染图像 MSE；allocation/joint 再加入当前分层档位分配与归一化率惩罚。没有新加手工属性权重。

## 训练阶段与步数

所有步数是**每个场景的真实优化更新次数**，不是 batch 数或者整个循环共用的次数。

| 阶段 | 默认每场景更新 | 共享 codec | 当前场景档位表 |
|---|---:|---|---|
| bootstrap | 5000 | local-response 属性恢复 | 不更新 |
| render | 5000 | 整场景 MSE；q1/q2/q3/mixed 轮换 | 不更新 |
| allocation | 1000 | 冻结 | 逐点渲染反馈 + 率惩罚 |
| joint | 1000 | 整场景联合微调 | 联合微调 |

三个训练场景合计 36000 次更新，其中 codec 更新 33000 次。场景轮流更新、每场景自己计数轮换档位；大场景不会因点数多而获得更多更新。bootstrap→render 清空共享 Adam 动量，其他阶段延续；不按场景重置共享优化器。学习率默认固定 `1e-4`，没有自动衰减或隐式裁剪。

render/allocation/joint 沿用 replay 反向以限制显存。原始数据与特征缓存在 CPU；只有当前场景的概率表和优化器状态搬入 GPU。主机内存需要容纳三个完整场景、特征及相机/参考图像，不能将旧剪枝场景的内存、秒/步直接套用。验证、画图、保存耗时不包含在 `step_seconds` 内。

每 500 次**每场景更新**，统一验证所有训练场景并保存图片、日志、图表。验证相机从各场景 train camera split 内划出；最终实验另用 test camera split。按场景等权验证选择 `best_*`，不会用 DrJohnson 或最终 test 数据挑 checkpoint。

### 留出场景的两种不同结论

1. 未参与训练的 DrJohnson，冻结 codec/统计量，使用统一 q1/q2/q3：检查 codec 的零样本跨场景迁移。
2. 冻结同一 codec，只用 DrJohnson 自己的训练视角拟合其概率表，然后在 test 视角评估：**场景分配器适配**，不能称为整个系统零样本。

留出适配不会重新估计属性统计，也不会微调共享权重。bbox/排序和坐标由新场景输入确定，属于发送内容，不是训练集统计。

## 启动

先在服务器激活已有 CUDA 环境，确认清单中的四套 PLY 与图片/COLMAP 数据存在。指定当前真正空闲的 GPU；下面的 0 只是示例。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
OUT="$PWD/output/multiscene_short_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_multiscene_short.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

短流程是每训练场景 100 bootstrap + 100 render + 100 allocation + 50 joint = 1050 次全局更新；之后另做留出适配及少量 test 视角评估。它验证链路和趋势，不能作为收敛证据。正式流程把脚本换成 `scripts/train_multiscene_full.sh`，默认所有 test 视角、3 次 AWGN 噪声重复、3 次打乱对照。

可用环境变量覆盖 `BOOTSTRAP_STEPS`、`RENDER_STEPS`、`ALLOCATION_STEPS`、`JOINT_STEPS`、`VALIDATE_EVERY`、`BLOCKS_PER_BATCH`、`LR`、`RENDER_LR`、`BETA`、`TEST_VIEWS`、`TEST_TRIALS`。`TEST_VIEWS=0` 表示所有 test 视角；`RUN_EVALUATION=0`、`RUN_ADAPTATION=0` 可暂不运行后续实验。

输出全部位于同一 `OUT` 下：

```text
OUT/
  training.json, loss.jsonl, validation_macro.jsonl, complete.json
  scenes/<scene>/scene.json, loss.jsonl, validation.jsonl
  scenes/<scene>/validation_images/<step>/*.png
  scenes/<scene>/allocation_history.jsonl, allocation_latest/
  checkpoints/{initial,best_*,end_*,latest,final}/
    codec.pt                  # 每个快照只有一份共享 codec
    truck.pt, train.pt, playroom.pt  # 配套概率表，绑定 codec hash + PLY fingerprint
    bundle.json
  charts/*.png, *.pdf
  evaluation/{end_allocation,final,heldout_adapted}/
  heldout_adaptation/
```

`latest` 是定期覆盖的安全权重快照；`end_*` 是固定阶段边界；`best_*` 是验证选择。**当前快照保存权重/概率表，不包含可精确恢复训练的优化器与 RNG 状态；不要把 --checkpoint 的适配当作 resume。**

## 已接入的实验

统一入口：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m gaussian_jscc.multiscene_experiments \
  --checkpoint "$OUT/checkpoints/final/codec.pt" \
  --out "$OUT/evaluation/snr_sweep" --snrs 0 5 10 15 20 --trials 3
```

| 实验 | 对照与控制 | 输出 |
|---|---|---|
| 跨场景 | 3 个训练场景的 test 视角；1 个完全留出的场景 | 逐场景照片/原 PLY PSNR、SSIM、MSE |
| 分配是否有价值 | 统一 q1/q2/q3、学习表、全点随机打乱、只打乱正档位 | 同档位直方图；后者固定 q0 位置，区分删点与保留点分档 |
| 存在必要性排序 | 仅当输入确有 MaskGaussian prior 时启用，同档位数量 | `prior_ranked`；官方普通 PLY 无 prior 时明确缺失，不伪造 |
| 渐进增益 | 4→12、12→24，配对视角与完整点/符号槽噪声 | `prefix_gains.jsonl`、柱图，保留负增益 |
| 联合微调 | `end_allocation` 与 `final` 配套 codec+表 | 固定阶段边界，避免用测试集挑最优结果 |
| β 率失真 | 同一 `end_render` codec、相同新表初始化与种子，仅改变 β | 真实率失真点、硬档位比例；不比较总 loss |
| SNR 错配 | 10 dB 训练模型测试 0/5/10/15/20 dB | 固定部署表，无隐式重分配；不能称为动态 SNR 训练 |

β/SNR 批量入口（四组顺序执行；可设置单一 `BETAS` 分别放不同 GPU）：

```bash
CUDA_VISIBLE_DEVICES=1 BETAS="0 0.001 0.01 0.03" \
  bash scripts/test_multiscene_beta_snr.sh \
  "$OUT/checkpoints/end_render/codec.pt" "$OUT/beta_snr"
```

比较既有实验：

```bash
python -m gaussian_jscc.multiscene_plots \
  --runs "$OUT/evaluation/end_allocation" "$OUT/evaluation/final" \
  --out "$OUT/evaluation/joint_comparison"
```

独立单场景对照可用同一个训练入口加 `--train-scenes truck`，保持所有目标、预算和统计实现相同。此时单场景 vs 共享训练默认是**每场景曝光次数相同**，不是总计算量相同，论文需报告两者差异。其他未参与共享训练的场景在结果中记录为 `unseen-scene frozen codec`。

### 图表与原始数据

- 训练：分场景、分阶段 loss；验证照片/原 PLY PSNR；部署占比收敛；期望 payload 与实际十次抽样部署 payload。
- 实验：完整成本–PSNR 图、q0–q3 堆叠比例、payload/XYZ/metadata 成本分解、前缀增益、SNR–PSNR/SSIM。
- 每份图同时输出 PNG/PDF；`results.csv/jsonl` 保留成本口径和完整分组字段；`per_view.csv` 保留逐视角逐噪声重复指标；`paired_comparisons.jsonl` 提供与 mask 的逐对差异。
- 图表采用明确单位、固定配色和线型，按 visualize-data 的比较/组成/趋势规则设计。少量率失真点不拟合曲线，不把多次噪声/打乱重复当作独立场景样本；不跨训练/留出场景混合平均。

## 通信成本的边界

最终评估实际调用与导出相同的 `encode_positions` 和 `encode_metadata`：坐标、完整档位序列、JSON/bbox/model-id 和 CRC/framing 均计入。payload 以实际复符号数量统计，而非 `.npy` 中 I/Q 浮点存储字节数。

默认可靠数字侧流按 **2 净信息 bit / 复信道使用**折算，仅为显式工程假设，不是已经实现的 FEC。结果保存 `same_snr_awgn_capacity_bits_per_use` 与超容量标志：例如 0 dB 下复 AWGN 容量仅 1 bit/use，2 bit/use 的可靠侧流不能假定在同一信道实现。SNR 曲线首先反映属性 JSCC 的错配，完整同链路论证还需要合适码率/FEC或独立受保护侧链路。

结果同时记录：

- 原始 XYZ/metadata bytes 与十进制 MB；总复信道使用次数；相对全 q3（24 符号）的总成本节约比例。
- 假定已共享 codec 的成本，以及额外发送一次 compact codec checkpoint 的初次交付成本；后者也只是假设可靠数字传输。
- `equivalent_digital_MB` 是同等信道资源对应的数字净信息量，**不是 JSCC 编码文件大小**。

训练期率惩罚仍用现有坐标实测值和档位表 proxy，最终完整包成本单独报告，不宣称 proxy 可精确微分。打乱操作保持 payload/档位数量，却可能改变 XYZ/档位压缩大小，因此是等 payload 对照，不是严格等总信道使用。

尚未实现真正的数字压缩 + 调制/FEC/误包基线；本次不将资源折算当作该基线的实测优越性。四个场景（仅一个跨场景留出）也不足以证明广泛泛化。

## 验证情况

`tests/test_multiscene.py` 用真实 codec、优化器、量化/压缩/打包与合成可微渲染验证：共享统计、逐场景档位轮换、冻结阶段哈希不变、联合更新、配套表加载、留出不参与训练、冻结适配、受控打乱、真实包成本一致，以及图表输出。CPU 合成渲染测试不代表真实 CUDA 场景质量、速度或收敛效果。
