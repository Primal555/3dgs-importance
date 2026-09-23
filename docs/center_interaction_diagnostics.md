# 中心位置的跨点交互诊断

目的：区分逐点信息可读性、邻居依赖和真正有害的邻域交互。不改已有检查点，
不以单个注意力图、特征相似度或大梯度直接宣布“过度融合”。
默认训练模型不变；新增行为全部显式开启。

## 三项检查及边界

### 1. 固定目标点，微扰邻居

`python -m gaussian_jscc.center_interaction_diagnostics` 读取已有中心 Transformer
检查点（LayerNorm/affine 均可），冻结参数，沿用检查点的场景边界、Morton 顺序、
分块和训练/留出划分。默认取 8 个留出块，每块 4 个固定目标，3 个随机方向试验。
每个非目标点移动的世界距离是该块非零最近邻距离中位数的 0、1%、5%、10%。
方向独立随机；同一目标/试验在各扰动幅度间复用方向。目标坐标不动、不重排、
不重拟合归一化边界。编码器内部 kNN 可因坐标扰动变化，这属于编码端响应。

分别记录三种接收输入：

| 路径 | 目标 latent | 邻居 latent |
| --- | --- | --- |
| end_to_end | 扰动邻居后重新编码 | 扰动邻居后重新编码 |
| decoder_only | 恢复为原始目标 latent | 扰动邻居后重新编码 |
| encoder_only | 扰动邻居后重新编码的目标 latent | 恢复为原始邻居 latent |

输出每个样本的目标位移、目标误差变化、输入扰动尺度与 latent 位移，及均值/
中位数/P95/最大值。`response_per_neighbor_displacement` 的分母是单个邻居移动
距离，不是全部输入的总 L2 范数，不能称为严格 Jacobian 或 Lipschitz 常数。
零幅度是空操作校验。两个隔离路径是人为的混合 latent 干预，可能离开训练分布；
三条路径不要求线性相加。邻居依赖也可能有益，必须结合第 3 项判断。

### 2. 冻结网络，训练同规格逐层读出器

提取解码输入 MLP 特征、每个 Transformer 块输出、实际坐标读出前的三个 tap。
默认使用 64 个拟合块、8 个留出块；两组不交叠，块内所有真实点均参与。
每层使用相同的 `Linear(hidden,64) -> GELU -> Linear(64,3)` 探针，初始张量相同，
每步采样点相同；固定 Adam 1e-3、batch512、1000 步。只训练探针，不训练 codec。

为减少特征尺度本身对探针优化速度的干扰，各通道按**拟合块**均值/标准差标准化；
目标坐标也只用拟合块统计。留出数据不参与统计、梯度、早停或选优。
记录每 100 步及末尾的拟合/留出世界坐标 RMSE 和距离中位数。
额外记录每块的中心化特征 RMS 与有效秩，不能把秩高低单独解释为好坏。

探针误差衡量的是有限容量、有限训练预算下的可读性，**不是该层所有位置
信息的上限，也不是实际 codec 输出质量**。浅层好、深层差是进一步调查的信号，
不能据此断言信息不可逆丢失。探针训练曲线用于检查其是否充分收敛。

### 3. 从头训练的解码端匹配对照

扩展 `scripts/compare_center_decoders.py --comparison interaction`：

- `block_attention`：原中心 Transformer，整块注意力。
- `self_only`：注意力仅允许每个点关注自身；保留 V/输出投影、内部 Pre-LN、
  FFN、残差、多层坐标读出。没有添加新的坐标旁路。

单个 key 的 softmax 恒为 1，故用等价 V/输出投影计算自身注意力，避免浪费完整
注意力矩阵。Q/K 不再影响输出，其对应梯度为零；存储参数量相同但有效容量不同，
不能宣称严格等有效参数量。两版整个模型初始参数逐项相同，实际采样序列、数据
划分与验证视角一致，结束时自动审计；不一致会报错。

**编码器 Context 在两组中仍然保留，并分别随训练更新。** 这是解码端交互消融，
不是全链路纯逐点网络，也不是固定 latent 的解码器拟合。
默认两组各 2000 步、固定 2e-4、batch32、仅中心训练、无裁剪、无噪声。
可设置 `STEPS=5000`；每 500 步保存渲染诊断图和 PSNR。

## 一次运行全部检查（后台）

确认空闲 GPU 后修改下例中的 `2`。检查点用于前两项诊断，**不会初始化第三项训练**。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main || exit 1

CKPT="$PWD/output/truck_center_affine_20260923_140901/codec_5000.pt"
OUT="$PWD/output/truck_center_interaction_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 STEPS=2000 READOUT_NORM=affine \
  nohup bash scripts/test_center_interaction.sh "$CKPT" "$OUT" \
  > "$OUT/console.log" 2>&1 &
echo $! > "$OUT/run.pid"
tail -f "$OUT/console.log"
```

`Ctrl+C` 仅退出 tail，不停止后台。任务串行运行，各训练组是独立进程。
脚本默认配对组均使用 affine；若诊断旧 LayerNorm 模型并希望配套比较，显式设置
`READOUT_NORM=layernorm`。`PLY`、`SCENE`、`PROBE_STEPS`、`LR`、`SEED` 可覆盖。

结果全部在一个根目录下：

```text
OUT/
  console.log
  diagnostics/
    sensitivity.json       # 完整扰动样本及分组统计
    diagnostics.json       # 配置/指纹/划分/codec 不变审计/逐层探针/有效秩
    diagnostics.png        # 邻居敏感性、逐层误差、探针收敛曲线
  training/
    experiment.json
    comparison.json        # 配对审计、最后一步与选优指标
    comparison.png
    block_attention/       # 日志、检查点、图像、验证指标
    self_only/
```

训练组看 `images/000500/view_00/center_only.png`（预测中心＋原属性）与
`render.center_only.source_psnr`。属性未训练，不用 `full.png` 判断中心效果。
前两项不做渲染，也不生成所谓“探针 PSNR”。

## 单独运行诊断或配对训练

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m gaussian_jscc.center_interaction_diagnostics \
  --checkpoint "$CKPT" \
  --ply "$PWD/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply" \
  --out "$OUT/diagnostics_only" --device cuda --probe-steps 1000

CUDA_VISIBLE_DEVICES=2 python -u scripts/compare_center_decoders.py \
  --comparison interaction --readout-norm affine \
  --ply "$PWD/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply" \
  --source "$PWD/data/tandt_db/tandt/truck" \
  --out "$OUT/training_only" --device cuda --steps 2000
```

输出目录必须是新目录；不会覆盖旧结果。一组失败时停止后续执行。
单组可用 `train-center-attributes --resume 路径/training_state.pt` 精确恢复；
配对入口不自动续跑/跳过完成组。两组完成后可用
`python scripts/compare_center_decoders.py --out "$OUT/training" --summarize-only` 重新汇总。

PLY 指纹不匹配会拒绝运行。仅在明确接受**非严格复现**时，诊断模块才提供
`--allow-ply-mismatch`；即使使用它也不重拟合坐标边界，报告标记指纹不一致。
服务器原 PLY 不应需要此选项。不要通过此开关掩盖数据迁移问题。

## 本机验证范围

单元测试覆盖对角注意力数值/梯度等价、解码器邻居不变性、全 padding、零扰动、
探针配对初始化、模型不被修改、检查点/精确恢复、初始张量/采样审计和错误 PLY 拒绝。
本机真实 Truck 流程检查使用与本地 PLY 指纹匹配的 600 步检查点，而非服务器
5000 步检查点；因此本机结果不能作为那次服务器训练的过度融合结论。

已完成的本机流程检查：16 个拟合块、4 个留出块、每块 2 个目标、2 次扰动试验、
300 步探针；另做每组 100 步、batch4 的完整配对训练并通过初始化/采样审计。
原模型 hash 保持不变，JSON 与曲线均成功生成。产物位于本机忽略目录
`output/center_interaction_cpu_check_20260923/` 和
`output/center_interaction_pair_smoke_20260923/`。这些短跑不作为选定新架构的证据。
