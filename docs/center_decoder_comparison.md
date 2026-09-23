# 中心解码器小对照：当前 Transformer 与历史轻量结构

入口：`scripts/test_center_decoder_comparison.sh`，配对程序：
`scripts/compare_center_decoders.py`。正常 `train-center-attributes` 仍默认
使用原来的 Transformer，不恢复已撤回的 point-residual 试验。

## 两组究竟比较什么

| 组别 | 中心解码器 |
| --- | --- |
| transformer | 当前四层块内 Transformer、浅/中/深三层特征读出 XYZ |
| historical_light | 历史逐点 MLP 特征＋线性 XYZ 头，叠加门控的轻量窗口 Context 输出 |

历史来源是提交 `97ef4cc` 的 `MultiScaleSelfCore.decode` 中 XYZ 分支，位于
主干 Transformer 提交 `aaf75dd` 之前。保留旧窗口偏移编码、32 槽窗口、
×4/×16 多尺度池化和 `tanh(0.1)` 初始门控。**旧结构并非纯 MLP**，
Context 内仍有小窗口注意力；不要把结果描述成“Transformer 对完全无注意力网络”。
旧结构原本就含逐点项和 Context 项，不是重新加入本轮已撤回的
“latent 仿射直读＋零初始化 Transformer 残差”。

这不是完整旧 JSCC 系统复刻：旧接收包改为当前 32 维中心 latent，去掉档位与
SNR 条件，不训练属性，也不模拟通信。只移植旧的中心解码结构，其他部分保持一致。

## 控制变量

- 两组均随机初始化，不使用 5000 步旧模型，也不使用最小二乘拟合初始化。
- 相同中心编码器、初始权重、中心 latent 宽度、PLY、Morton 顺序、块划分。
  编码器在各组中分别更新，因解码器不同而逐步分化；比较的是两种解码结构
  下的端到端学习效果，不是固定 latent 的纯解码拟合。
- 相同采样块序列、Adam、固定 `2e-4`、原中心距离损失、无梯度裁剪。
- 相同 hidden96、block256、默认 batch32；两种解码器参数量不强行匹配，
  实际参数量写入报告。这是实用结构对照，不是等参数量消融。
- 默认每组 **2000 步**，只有 A 阶段；不因中心 PSNR 门槛提前结束，也不进入 B/C。
- 两组在同一张卡串行运行，每组独立子进程，避免同时占用两份模型显存。

初始化时保持 Transformer 基线构造的随机数序列不变，再在隔离的随机数
环境中生成轻量解码器，并复制形状相同的输入 MLP 初始权重。结束后程序
检查共同初始权重、每一步实际采样块、场景指纹、验证块与视角均相同；
任何不一致都会明确报错，不静默宣称公平对照。

## 服务器后台启动

确认空闲 GPU 后修改下例中的 `2`：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

OUT="$PWD/output/truck_center_decoder_pair_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 STEPS=2000 BLOCKS_PER_BATCH=32 \
  nohup bash scripts/test_center_decoder_comparison.sh "$OUT" \
  > "$OUT/console.log" 2>&1 &
echo $! > "$OUT/run.pid"
tail -f "$OUT/console.log"
```

`Ctrl+C` 退出 tail 不停止 nohup 任务。脚本先跑 transformer，再跑
historical_light，两组默认共 4000 个优化步骤。可用 `STEPS=1000` 缩短，
不要只改一组步数。`PLY`、`SCENE`、`LR`、`SEED` 可覆盖。

## 观察什么

所有结果在同一个根目录内：

- `comparison.json`：公平性检查、参数量、训练步耗时、最大梯度、最后一步和
  验证最优步的指标。必须同时看相同步数结果与选优结果。
- `comparison.png`：固定拟合块/留出块 RMSE、训练中心损失、梯度、中心诊断 PSNR。
- `transformer/` 与 `historical_light/`：各组完整 loss、validation、checkpoint、
  图片与日志。每 500 步和预算末尾保存图像，并在第 0 步保存初始图。
- 各组 `images/000500/view_00/center_only.png`：**预测中心＋原始属性**，
  是本试验应该比较的图。`full.png` 含未训练属性，不能用来评价中心路径。
- `validation.jsonl/centers` 是留出块；`fitted_centers` 是固定 32 个拟合块的抽样，
  不是整个拟合集。每块 RMSE 和最大两个块的 SSE 占比帮助识别异常区域影响。

有 `--source` 时，PSNR 相对于输入 PLY 的渲染图计算，也保留相对照片指标；
最优检查点按中心诊断 source PSNR 选择。无相机本机检查则按留出 world RMSE
选优，不声称验证画质。单次小测试只用于筛选，不能据此宣布某类架构普遍更优。

一组失败时停止后续执行。可以按该组的标准恢复状态续训：

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m gaussian_jscc train-center-attributes \
  --resume "$OUT/transformer/training_state.pt"
```

配对脚本不会覆盖已有目录，也不会自动接着运行另一组。只有两组均完成后，
可重新汇总已有结果：

```bash
python scripts/compare_center_decoders.py --out "$OUT" --summarize-only
```

## 本机可复现检查

```bash
python scripts/compare_center_decoders.py \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --out output/center_decoder_comparison_cpu_trial \
  --device cpu --steps 600 --blocks-per-batch 4 --validate-every 100
```

此命令使用完整 PLY，但不加载相机；与服务器 batch32 的训练轨迹和速度不同。

### 2026-09-23 本机实测

同一 Truck PLY，seed42，每组 600 步、batch4，其余使用上述默认值：

| 中心解码器 | 参数量 | 第 600 步拟合块抽样 RMSE | 第 600 步留出块 RMSE |
| --- | ---: | ---: | ---: |
| 当前 Transformer | 488451 | 2.77703 | 4.59229 |
| 历史轻量结构 | 349831 | 1.78044 | 2.70583 |

共同初始化、实际采样序列、数据划分审计通过。拟合指标只覆盖固定的 32 个
训练块，留出指标覆盖固定的验证块；均为世界坐标逐轴 RMSE。轻量结构在
本次短跑末尾更好，是值得上服务器验证的信号，不代表已解决中心精度问题。
这是单种子、不同参数量的对照，本机未执行 CUDA 渲染，因此没有本机 PSNR
或画质改善的结论。完整产物保存在本机忽略目录
`output/center_decoder_comparison_cpu_20260923/`，不提交大模型与训练产物。
