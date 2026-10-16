# Truck 一体式渐进属性通信实验

独立入口 `gaussian_jscc.integrated_experiment`，启动脚本 `scripts/test_integrated_progressive_color_truck.sh`。
保留当前模块化双编解码器路线，不修改旧一体式脚本或 checkpoint；默认只跑一组，不自动增加对照组。

## 配置与训练

Gaussian 属性 → 一体式编码器 → 完整码字／独立增量功率归一化 → 前缀与 10 dB AWGN → 一体式属性解码器。
可靠 16bit XYZ 仍直接交付，量化整数 delta-zlib 压缩；不学习坐标，不将原浮点 XYZ 声称为无损恢复。
累计 q1/q2/q3 为 **12/24/48 个复符号**，对应增量 12/12/24。
q2、q3 使用完整已收前缀和前层属性进行累加修正，所有阶段允许联合反向。
旧属性残差版本本来就有完整前缀重读，本次将其与独立颜色监督和新预算组合，不把它虚构成全新能力。

- 从随机权重开始；RUN 只读取路径、相机划分及主干尺寸，不读取历史 codec 权重。
- 默认沿用 96 维、两层栅格编码上下文和两层局部注意力解码；不叠加 PointNet++、新邻居去噪或门控分支。
- 5000 步属性恢复：3750 步局部 Gaussian 响应 MSE，1250 步独立、未截断方向 RGB MSE。
- 颜色步骤也经过当前有限前缀和噪声链路。八个方向为四个训练相机方向、四个随机球面方向。
- 颜色步骤训练共享编码／解码网络和各层 DC/SH 头；各层非颜色输出头梯度设为 None，防止 Adam 动量更新。
- 两种属性目标各自循环档位，避免每第四步永远训练 mixed；5000 步后各 layout 总计 1250 次更新。
- 随后 1000 步原 PLY 场景渲染 RGB MSE，q1/q2/q3/mixed 各 250 步，replay 反向，无 latent 或属性辅助损失。
- 两阶段均固定 LR=1e-4，切换阶段新建 Adam；阶段内续训保留 Adam、RNG 和精确进度。
- 不接分配器，不学习删点、不加通信惩罚；mixed 是随机正档位混合，不是已学习部署。

## 服务器启动

GPU 3 为示例，换为实际可用卡。SOURCE 可以是原多场景目录、旧一体式实验，或含完整 scene 元数据的模块化 arm 目录。
以下选择已存在的原剪枝场景运行目录，不加载其权重：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

GPU=3
RUN="$PWD/output/multiscene_pruned_aligned_gpu0_20261010_194134"
OUT="$PWD/output/truck_integrated_color_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/output"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_integrated_progressive_color_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"${OUT}.pid"
printf '输出：%s\n日志：%s\n' "$OUT" "${OUT}.log"
tail -n 40 -f "${OUT}.log"
```

关闭终端不终止 nohup 训练；Ctrl+C 只停止 tail。
迁移数据路径时追加 `--manifest /absolute/path/manifest.json`，场景和相机身份必须一致。

断点续训使用原 RUN、原 OUT 和相同参数，不重新生成时间戳：

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/test_integrated_progressive_color_truck.sh \
  "$RUN" "$OUT" >>"${OUT}.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

精确续训会核验原 PLY、相机名字及实际内外参／相机中心／图像内容指纹、批次设置和成本换算条件。
已经存在 `complete.json` 的目录拒绝重跑，也拒绝在现有目录中回退到较旧 checkpoint，避免留下误导性的完成标记和未来端点。
最终修正量诊断直接使用已核验的内存场景，不重新读取未使用的存在概率文件。

可显式覆盖 `--bootstrap-steps`、`--render-steps`、`--rates`、`--lr`、`--color-every`、`--blocks-per-batch`。
`--color-every 0` 可另启同预算颜色关闭对照；修改这些设置不是精确续训，需使用新的 OUT。
每步日志记录 objective、layout 与梯度；阶段预算指真实 optimizer 更新次数，不是 epoch。

## 结果

所有结果集中在一个 OUT：

- `training.json`、`progress.json`、`complete.json`：设置、真实更新计数和成功完成标记。
- `metrics/loss.jsonl`：三种目标分开记录，含模块梯度／更新范数、耗时与显存。
- `metrics/validation.jsonl`：每 500 步及边界，无噪声有限链路和 10 dB 的 q1/q2/q3/mixed，属性、颜色和画质指标。
- `metrics/test.jsonl`：最终独立测试相机，两条链路；不自动运行多 SNR 扫描。
- `metrics/prefix_gains.jsonl`：配对视角／噪声下的 q1→q2、q2→q3 增量收益和改善比例。
- `images/initial`、`images/final`：相同两个代表验证视角、四种布局的 10 dB 对照图；不保存中间轮次图片。
- `stage_diagnostics.json`：最终三个阶段的属性误差和修正量，仅诊断，不是新损失。
- `charts`：目标分组、画质与色差、增强层收益与时间、模块梯度／参数更新曲线。
- `checkpoints`：initial、latest、end_bootstrap、end_render、final，均保存完整训练状态。

无噪声 `finite_noiseless` 仍经过前缀与功率约束，不是模块化实验的 clean latent 直通路径。
指标同时对原 PLY 渲染和照片计算，训练教师仅为原 PLY。
数字侧成本用实测坐标与元数据字节，模拟符号以复信道使用计；`net_bits_per_use` 是明确假设而非实际 FEC 测试。
本次预算大于旧 4/12/24，跨版本收益不能全部归因为结构或颜色监督。

本机 CPU 测试验证真实网络、信道、梯度、包接收和续训；仅 CUDA 光栅化边界由可微合成图像替代，
不据此宣称实际 Truck 画质或训练速度已经提升。
