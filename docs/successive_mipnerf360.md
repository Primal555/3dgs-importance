# Mip-NeRF 360 扩展训练：剪枝后续训共享编解码器

本轮从已完成的 `successive_multiscene_gpu3_20261022_230109` 继续训练，不重新随机初始化，也不训练档位分配器。旧三场景参与轮换，降低新增场景训练对已有恢复能力的影响。

## 数据与训练协议

| 项目 | 本轮设置 |
| --- | --- |
| 旧训练场景 | Train、Truck、Playroom；使用源实验记录中的剪枝 PLY、照片路径与相机划分 |
| 新训练场景 | bicycle、garden、stump、bonsai、counter、kitchen |
| 新留出场景 | flowers、treehill、room；共享编解码器仅在初始、最终评估 |
| 新场景预处理 | 九场景分别从 iteration_30000 的预训练 PLY 开始，顺序运行 MaskGaussian 5000 步，输出 iteration_35000 |
| 共享模型训练 | 九个训练场景各 1250 次实际参数更新，共 11250 次；逐场景轮换，使用同一个 Adam |
| 可训练模块 | 语义编码器、语义解码器、信道编码器、信道解码器 |
| 初始化 / 学习率 | 完整继承源模型权重；新建 Adam，四模块均为 1e-5 |
| 信道 / 档位 | 10 dB AWGN；q1/q2/q3 = 8/24/48 个复符号；轮换 q1、q2、q3、mixed |
| 可靠旁路 | 16bit XYZ 与 binary16 块尺度，保持源模型实现 |
| 训练目标 | 原始剪枝 PLY 的 RGB 渲染 MSE + latent MSE（系数 1） |
| 属性归一化 | 继承源模型的 mean/std，新增场景不重新拟合 |
| 验证 / 保存 | 每场景每 250 次更新验证；每场景每 100 次更新保存完整恢复点 |
| 最终测试 | 每场景最多 32 个测试视角、4 次噪声试验；PSNR、SSIM、LPIPS 与通信成本 |
| 图像保存 | 初始和最终结果图；中间验证只存数值 |

原始照片目录：`data/mipnerf360/<scene>`。预训练模型目录：`data/pretrained_3dgs/<scene>/point_cloud/iteration_30000/point_cloud.ply`。预处理清单为 `configs/mipnerf360_raw.json`。

清单中九场景均标记 `train`，该标记仅用于逐场景 MaskGaussian 预处理。后续生成的 `dataset.json` 会将 flowers/treehill/room 明确标为 `heldout`。这里的零样本泛化指**共享编解码器**未在留出场景上训练、拟合统计量或校准；场景的预训练 3DGS 和逐场景剪枝仍使用各自的训练照片。剪枝后的 PLY 是本轮通信恢复的源场景，照片指标另外记录。

## 服务器启动

在仓库根目录，激活 `maskgs` 后执行。GPU 编号按实际空闲情况选择。

```bash
RUN="$PWD/output/successive_multiscene_gpu3_20261022_230109"
GPU=3
OUT="$PWD/output/successive_mip360_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

# 只检查路径、模型来源与全部九场景数据；不创建输出，不使用 CUDA。
CUDA_VISIBLE_DEVICES="$GPU" bash scripts/train_successive_mipnerf360.sh "$RUN" "$OUT" --dry-run

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/train_successive_mipnerf360.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
printf '输出：%s\n日志：%s\n' "$OUT" "${OUT}.log"
tail -n 40 -F "${OUT}.log"
```

`nohup` 任务不依赖本地电脑或 SSH 保持连接。关闭 `tail`（Ctrl+C）不会终止训练。

总流程首先检查九场景的 SH 阶数是否与源模型兼容，并对输入 PLY、相机文件和照片计算内容哈希；这一步在启动或恢复时执行一次，不进入训练循环。之后运行九次逐场景剪枝、剪枝结果渲染与质量统计，完成后才进入共享模型训练。剪枝阶段主日志会提示具体场景日志位置。共享模型初始评估覆盖十二个场景，完成后才打印实际更新；这段时间没有训练步数增长是正常的。预处理和初始/最终测试的耗时不包含在 11250 次模型更新中。

## 输出与观测

```text
OUT/
  pipeline.json                     # 固定输入、参数、模型哈希和数据角色
  progress.json                     # 当前阶段 / 失败原因
  preprocessing/pass_000/
    plan.json / summary.json        # 顺序剪枝计划、点数与照片质量
    <scene>.log                     # 单场景剪枝/渲染/统计日志
    <scene>/point_cloud/iteration_35000/point_cloud.ply
  dataset.json                      # 9 train + 3 heldout，全部指向剪枝表示
  training/
    plan.json / training.json
    metrics/loss.jsonl              # 场景、档位、render/y loss、更新耗时、显存峰值
    checkpoints/{initial,latest,final}/codec.pt
    scenes/<scene>/evaluation/{initial,final}/
    comparison/endpoints.csv        # 按场景对比初始与最终指标
    charts/                         # 训练场景与留出场景质量曲线
    complete.json
  complete.json                     # 整条流水线成功标记
```

`mixed` 是固定随机正档位图，用于检查混合档位恢复，不是学习分配的部署结果。留出场景不参与中间验证或模型选择。训练场景使用未参与梯度更新的验证/测试相机；旧三个场景的划分沿用源实验。

所有训练场景的 CPU 点数据与相机元数据会驻留，训练及回放按批次传入 GPU；应同时关注主机内存。新数据更复杂，不能直接沿用 Truck 的单步时间估算全程。`metrics/loss.jsonl` 的 `step_seconds` 和 `peak_allocated_mib` 可用于实际测量。

## 中断恢复

使用**同一组参数、同一源模型和原 OUT**，在确认旧任务已退出后执行：

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/train_successive_mipnerf360.sh \
  "$RUN" "$OUT" >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

- 剪枝阶段：复用已经完成导出和质量统计的场景；未完成场景从原 PLY 重跑，进入新的 `preprocessing/pass_001` 等目录，保留原始失败记录。此阶段不宣称恢复剪枝 Adam。
- 编解码器训练阶段：从 `training/checkpoints/latest/codec.pt` 恢复权重、Adam、随机状态、场景轮换与更新计数；最多回退到最近一次保存。切换场景不会清空 Adam。
- 如果在训练初始化尚未保存第一个 checkpoint 时就退出，会明确报错；可直接调用 `gaussian_jscc.successive_transfer`，使用已生成的 `dataset.json` 和一个新的训练 OUT 启动，无需重做剪枝。
- 禁止向同一个 OUT 同时启动两份任务。参数、源模型或数据路径变化应另建输出目录。

## 本机验证范围

定向 CPU 测试使用小型真实编解码器与可微合成渲染器，检查九场景更新、统计量固定、旧划分保留、留出目标变化不影响训练权重、精确续训，以及剪枝中断复用。未在本机进行九场景 CUDA 全量训练；实际显存峰值与收敛质量由服务器日志确认。
