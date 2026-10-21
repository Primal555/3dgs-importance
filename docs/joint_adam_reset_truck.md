# 阶段切换时清空 Adam：30步联合诊断

本实验与“冻结信道、仅训练语义解码器”版本独立，后者不作修改。
目的不是验证收敛，而是检查此前联合第1步过冲和后续大梯度是否与继承旧Adam有关。

## 唯一干预

从同一个 `end_channel_bootstrap/codec.pt` 初始化。
传入 `truck_power_stability_20261021_100748` 根目录时，自动选中
`block_continuation/block/checkpoints/end_channel_bootstrap/codec.pt`，即块组8000步端点。
不接着训练信道，不使用失败的联合 `latest`，也不接上解码器适应后的权重。

- 语义编码器冻结；信道编码器、信道解码器、语义解码器联合更新。
- 信道两端LR仍为 `1e-4`，语义解码器仍为 `1e-5`。
- 损失仍为 `RGB MSE + (1-SSIM)/3 + 原始y的MSE`。
- 16bit XYZ可靠交付、8/24/48复符号、10dB AWGN、q1/q2/q3/mixed轮换、
  两个训练视角及完整场景replay均保持不变，以源场景元数据为准。
- 相机、档位与噪声沿用之前短审计的累计步数采样公式。
- 不增加裁剪、步长拒绝、学习率热身或衰减。
- **切换时重建Adam**：移除信道的一阶／二阶动量及计步器。
  原有参数组betas、eps、weight_decay等超参数照旧；语义解码器依旧新建Adam状态。
  网络权重和属性统计不重置。

`optimizer_initial.json`记录三组状态的已初始化参数数目与计步范围。
reset组初始均应为0；`checkpoints/initial/codec.pt`保留完整证据。
每一步之后也记录优化器计步，避免只清空动量而遗留旧计步器。

## 观测与输出

- 默认30个真实优化器更新，每步记录同输入更新前后总损失、图像MSE、SSIM项及latent MSE。
- 默认保留原短审计的分项参数梯度、尺度／旋转／颜色梯度、前16个高梯度椭球、
  相机深度和梯度分解一致性检查。与旧 `instrumented` 组比较，不混同旧 `plain` 轨迹。
- `steps.jsonl` 是完整逐步证据，`summary.json`汇总首步变化、最大梯度和恶化步数。
- 仅保存 initial/latest/final 三份检查点；不扫描完整测试集、不额外保存大量图像。
- 输出目录必须新建。发生非有限错误会保存诊断状态，并以失败状态停止，不宣称完成30步。
- 本短探针不提供续训；中断后用新目录重新从相同端点跑，避免恢复受损的联合权重。

30步足以检查早期故障是否重现，**不能证明长期稳定性或最终画质**。
CUDA轨迹可能因数值差异分叉；若重跑继承组，使用同一入口和相同诊断设置配对。

## 服务器命令

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

GPU=3  # 示例，先确认空闲
RUN="$PWD/output/truck_power_stability_20261021_100748"
OUT="$PWD/output/truck_joint_adam_reset_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_joint_adam_reset_truck.sh "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

如需新的配对继承组，使用另一个新输出目录，并在同一脚本末尾添加 `--adam-mode inherit`。
脚本默认不会自动重跑旧对照。`--dry-run`只检查源端点并显示计划，不创建输出。
数据路径迁移时可添加 `--manifest`，仍会检查PLY身份和相机划分。

本机验证仅使用小型真实网络和合成可微渲染，检查优化器重置、采样对齐、反向路径及源文件不变。
真实CUDA梯度爆炸是否缓解，需要服务器这30步的结果判断。
