# Truck 空间上下文对照实验

本实验承接模块化 JSCC 与 SNR 条件修复，检验 PointNet++ 式空间聚合能否提高属性表示和接收重建质量。默认版本仍为原网格编码器、XYZ 辅助窗口解码器；新增结构须显式启用。不是完整 PointNet++ 移植，也不预先宣称增强层必然更有效。

## 四组与共同条件

| 组名 | 语义编码器上下文 | 语义解码器上下文 |
| --- | --- | --- |
| baseline | 原 4³/8³ 网格 | 原 32 点固定/移位窗口 + XYZ |
| encoder | 新空间聚合 | 原窗口 + XYZ |
| decoder | 原网格 | 新空间聚合 |
| both | 新空间聚合 | 新空间聚合 |

只训练 Truck。四组均随机初始化，复用 SOURCE_RUN 的 PLY 身份、相机划分和模型宽度，不读取其网络权重。共同层与信道两模块的初始权重逐元素一致，新增模块初始化不改变后续训练随机序列。相同样本、视角、SNR 和档位调度；参数量和计算量并不相等，日志记录其代价。

- 可靠 XYZ：每轴 16bit、delta_zlib；位置不学习，不预测残差。
- 语义 latent：每点 48 维；信道两模块沿用当前实现。
- q1/q2/q3：前 4/12/24 个复符号；一次编码、按前缀交付。
- 固定训练 10 dB，四模块学习率均为 1e-4，无梯度裁剪。
- 不训练分配器，不删除真实点。mixed 为固定随机正档位验证图，不能称为学习分配后的部署。
- 信道物理 SNR 与条件输入继续分开，测试时条件限制在已训练区间，本轮即固定 10 dB。

| 阶段 | 默认实际优化步数 | 目标与可训练模块 |
| --- | ---: | --- |
| semantic_bootstrap | 2000 | 原局部响应 MSE；语义编码器、语义解码器 |
| semantic_render | 1000 | 原 PLY 渲染 RGB MSE；语义两端 |
| channel_bootstrap | 2000 | y_hat 对冻结 y 的 latent MSE；信道两端 |
| channel_render | 2000 | 渲染 MSE + 1 × latent MSE；信道两端与语义解码器 |

每组共 7000 次更新；完整四组 28000 次。通信阶段按 q1/q2/q3/mixed 循环。两次场景渲染训练沿用 replay、原视角数量及 batch 设置，每 500 步验证和保存 latest。可以通过 `--blocks-per-batch 32` 同时减小四组显存需求。

## 空间模块的具体含义

每个原有 256 点处理块：逐点特征 → FPS 64 个内部代表点 → FPS 16 个内部代表点 → 特征插值回 64 点 → 回全部 256 点。最后保留 `自身特征 + tanh(gate) × 上下文修正`，gate 初始参数为 0.1。代表点仅用于内部计算，不是删点、额外发送的坐标或新增通信 token。

1. 将 unit XYZ 乘以已有 bbox 的三轴跨度，再除以同一个最大跨度。距离排序与世界坐标欧氏距离相同，不依赖椭球大小、旋转或属性。这里的“世界单位”是 PLY 单位，未假设为米。
2. 每层用 FPS 采样内部代表点；起点取距有效点均值最远的点。距离相同的点允许按槽位打破平局。该模块在无距离并列时对块内点重排等变；Morton 分块仍限制整体系统。
3. 每层用有效点的最近非重合点距离中位数作为基础间距，采用 2 倍、4 倍半径，两支各最多 16、32 个邻居。半径与数量是公开的实验超参数，不是论文最优值。小球内不足额时保留实际邻居，不强行补入远点。
4. 分支输入为邻居特征、按该球半径归一化的相对 XYZ 和距离，经逐点 MLP、有效邻居 max 聚合后合并。恢复层采用三个最近代表点的反距离加权插值，与原分辨率逐点特征拼接后处理。
5. 不使用 BatchNorm，不在处理块之间混合统计量；保留自身路径。新解码上下文不再添加序列正弦编码，空间坐标负责相对关系。因此这是一次完整上下文模块对照，不能把收益全部归因于距离公式。

发送端使用源坐标，接收端使用真正量化交付的 XYZ。bbox 来自本来已有的数据包元数据，图结构在两端各自建立；不传邻接表，不向接收端注入源属性或 clean latent。bbox 不写入可训练参数，解码数据包时从包内绑定。无效点、q0 和填充点都不参加采样与聚合。

搜索仅在原处理块内；没有跨块 halo。本实现是纯 PyTorch，便于现有环境运行；FPS/邻域搜索会增加计算，实际 GPU 性能以日志为准。

思想来源：[PointNet++ 原论文](https://arxiv.org/abs/1706.02413)。代码为本项目独立实现，借鉴 set abstraction、多尺度半径分组和 feature propagation，并针对逐 Gaussian 输出加入自身残差路径。

## 服务器启动

GPU 编号改成实际空闲卡。SOURCE_RUN 可为旧多场景源目录，也可直接使用最新模块化实验目录；这里只读取场景身份和配置。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_modular_latent_gpu2_20261013_113847"
OUT="$PWD/output/truck_pointnet_context_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_pointnet_context_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"${OUT}.pid"
printf 'Output: %s\nSuite log: %s\n' "$OUT" "${OUT}.log"
tail -n 40 -F "$OUT/baseline.log"
```

四组在同一张可见卡上顺序运行，每组独立进程释放显存。总日志显示当前组；`baseline.log`、`encoder.log`、`decoder.log`、`both.log` 记录每步详情。Ctrl+C 退出 tail 不影响后台训练。

只先跑基线和两端升级：在脚本参数末尾加 `--groups baseline both`。这不能区分收益来自哪一端，完整四组更适合定位。各组步数可用原 `--semantic-bootstrap-steps` 等参数覆盖，所有组统一生效。

中断后使用同一个 OUT、原来的全部参数，额外加 `--resume-suite`。已完成组跳过，其余从 latest 恢复 Adam、RNG 和进度。更改学习率、组列表、步数或路径时应新建 OUT。若某组在第一个可恢复检查点前失败，应新建输出目录。

迁移机器时可用 `--manifest /path/to/local_manifest.json` 改数据路径；PLY 身份和相机划分仍会验证。

## 输出与判读

所有组位于同一个 OUT 下。每组保存初始/最终两轮少量 clean、q1/q2/q3/mixed 对比图，不保存中间渲染图；保留各阶段末和滚动 latest/final 检查点。

- `comparison/metrics.csv`：全部最终路径、SNR、档位的 source/photo PSNR、SSIM、latent 误差、通信成本。
- `comparison/quality_by_tier.png`：10 dB、有限符号无噪声两种条件的 q1/q2/q3 绝对画质。
- `comparison/enhancement_gains.png`：q2−q1、q3−q2 的 dB 增量，允许显示负值。
- `comparison/learning_curves.png`：clean 与通信 q3 的验证学习曲线。
- `comparison/training_cost.png`：分阶段每步时间；参数量和峰值显存在 summary 中。
- `各组/metrics/spatial_neighborhoods.json`：均匀抽取最多 16 个原处理块，统计两层/两半径的实际邻居数、仅自身比例、超过邻居数量上限的比例与世界单位半径。可判断半径是否过小或邻域是否经常被截断。baseline 不生成此文件。
- `各组/metrics/test.jsonl`：最终全部测试相机、两次噪声试验、每视角数据，以及 0/5/10/15/20 dB 与 clean/finite-noiseless 路径。

额外采样诊断只在初始化时执行，不在每步 forward 中同步统计。汇总图只读取已保存数据；重绘用：

```bash
python -m gaussian_jscc.pointnet_context_controls --run "$RUN" --out "$OUT" --summarize-only
```

首先比较同档位绝对画质是否改善，再看增强层增量、颜色误差和耗时。q1 变好导致 q3−q1 缩小也可能是好结果，不能把“拉大档位差”本身作为成功标准。各模型 latent 坐标系不同，latent MSE/NMSE 只辅助解释自身信道恢复，不能替代渲染质量横向排名。单种子结果用于筛选，尚不足以宣称稳定统计收益。

## 本地预检（2026-10-13）

CPU、seed42、旧 Truck PLY `output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply`，按 Morton 序列均匀取八个 256 点块，前四个拟合、后四个留出。四组各进行 50 步 semantic_bootstrap，均为 hidden96 / latent48；这里不是最新 711691 点场景的完整训练，也没有 CUDA 渲染。

| 组 | 第 50 步拟合局部响应 loss | 随后留出块 loss | 参数量 | 50 步 CPU 用时 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.03551 | 0.02213 | 344496 | 4.35 s |
| encoder | 0.03720 | 0.02217 | 466480 | 6.09 s |
| decoder | 0.02777 | 0.01764 | 382120 | 5.72 s |
| both | 0.02909 | 0.01887 | 504104 | 7.14 s |

本地结果支持继续比较 decoder/both，但不证明全场景 PSNR 或增强层收益。逐步数据位于本地 `output/pointnet_context_cpu_probe/summary.json`，不提交训练产物。

这八块第一级小半径平均约 3.02 个邻居、仅自身占 38.3%；大半径平均约 11.46 个、仅自身占 8.8%。因此两种半径确实提供不同的聚合范围；若新场景诊断出现两个尺度都高度退化为自身，需先核查分组尺度，不能直接把没有收益归因于网络无效。CPU 计时不外推为 GPU 训练速度。
