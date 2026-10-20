# 信道编解码器只读审计

入口：`python -m gaussian_jscc.channel_audit`；脚本：`scripts/audit_channel_codecs_truck.sh`。

这是现有诊断的扩展，不改变网络、损失、训练阶段或优化器。不执行训练，不写回源目录，不复制 checkpoint/PLY，不需要 CUDA 光栅化器。CPU 可用 `--device cpu`，正式配对审计建议服务器 GPU。

## 本轮比较对象

- 当前：`output/truck_channel_continue6000_gpu2_20261020_153617`，累计信道恢复 6000 步，尚未联合微调。
- 历史：`output/truck_error_gates_gpu3_20261015_120326/neighbor_plain10`，历史分步信道恢复后的模型，尚未联合微调。历史累计预算及各模块的更新范围不同，不能据此宣称纯粹的结构因果优势。
- 默认读取各自 `checkpoints/final/codec.pt`。直接 Python 入口支持 `--checkpoint` 和 `--reference-checkpoint`。
- 两份模型使用**各自原有的语义编解码器和 latent**。不互换信道权重，不把新模型的 y 交给旧模型。各维 latent 没有跨模型一一语义对应关系。

预检查核对 checkpoint/config、场景记录、指纹、点数、几何范围、分块大小和符号预算。实际运行再次核对 PLY，不能拿本地未剪枝的大 PLY 代替服务器 711691 点版本。路径迁移可用 `--manifest`，但数据必须相同。

## 记录什么

1. **功率归一化**：每个符号段的原始功率、归一化功率的均值/分位数、接近功率上限的比例，以及径向对数增益 `epsilon / (E + epsilon)`。正 power floor 下映射并非严格丢失幅度，但该增益很小时可能难以精确求逆。这项统计本身不证明它是最终瓶颈。
2. **实际解码过程**：观察每级主干、逐点 denoiser、最终邻域修正前后的误差。通过临时 hooks 记录正式 forward，不重新实现解码，不关掉模块改变输入分布。逐层记录与正式最终输出必须完全一致。
3. **误差分解**：原始完整 Morton 块抽样，多次独立噪声，记录逐点平均误差、随机方差、逐维误差、联合二阶矩、协方差/相关矩阵，以及最差 1% 点贡献的误差比例及原始点 ID。
4. **属性响应**：在固定语义解码器下，分别记录无裁剪方向 RGB、R−G/G−B、opacity、log-scale、log-covariance 和相对 covariance 误差。颜色和形状都检查，不以平均 latent MSE 代表画质。每个模块都有修正前/后的属性误差及差值。
5. **等强度敏感性**：把真实信道误差方向与各向同性随机方向都缩放到相同的块内相对 NMSE（默认 0.02），分别通过各自冻结的语义解码器，比较属性响应。没有训练或把特征维度手工标成“红色维”。
6. **逐点注意力的冗余检查**：pointwise denoiser 实际只重复读取自身。数值比较含 Q/K 的正式输出与不含 Q/K 的等价自身映射。等价不代表删掉整个 denoiser 后重新训练一定更好。

默认 32 个原始块（通常 8192 点）、32 次噪声、8 个固定球面方向；物理属性响应及等强度扰动在其中 4 个原始块上计算，节省时间。所有节点、档位和模型复用同一批点；旧、新相同符号槽使用相同噪声。原始块内部上下文完整，部分尾块保持原样，不把选出的点重新分组。

物理噪声 SNR 与网络条件 SNR 分别记录。条件处理沿用正式推理的 `resolve_condition_snr`：例如只在 10 dB 训练的模型，指定物理 `--snr 0` 时仍使用 10 dB 网络条件，而真实注入 0 dB 噪声，不人为改变正式部署规则。

q1/q2/q3 及固定随机 `mixed` 都检查。`mixed` 是均匀正档位诊断，不是学习后的分配器，也不测试 q0 剪枝。

## 数值解释

经验恒等式（方差分母为试验次数 T）：

`有噪 MSE = 各点跨试验平均误差的 MSE + 点内噪声方差`

有限次噪声的平均误差仍包含采样噪声，不能直接当作已证实的总体偏差。

若进一步分解成无噪误差与噪声引起的平均位移，还必须包含交叉项：

`有噪 MSE = 无噪 MSE + 平均位移 MSE + 点内方差 + 2 mean(无噪误差 × 平均位移)`

两条恒等式都保存残差用于检查。**不能直接把无噪 MSE 与噪声位移 MSE、方差相加而遗漏交叉项。**

`stage_metrics.csv` 中 `delta_mse < 0` 表示该模块在当前输入上的即时修正有效；`response_delta_* < 0` 表示相应属性的误差减小。只在该级实际更新的点上计量（q≥当前级），但解码仍保留整块上下文。这不是“删除该模块重新训练”的因果消融。原始观测点数分别列在 `latent_point_observations` 与 `response_point_observations`，两者覆盖范围不同。

`*_vs_clean` 隔离信道造成的增量损伤；`*_vs_source` 直接对照原 PLY 属性。方向颜色、物理参数误差不是渲染图像 PSNR/SSIM/LPIPS，不用它们替代整场景评价。本轮不另跑昂贵的整场景渲染。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

CURRENT="$PWD/output/truck_channel_continue6000_gpu2_20261020_153617"
REFERENCE="$PWD/output/truck_error_gates_gpu3_20261015_120326/neighbor_plain10"
OUT="$PWD/output/truck_channel_audit_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/audit_channel_codecs_truck.sh \
  "$CURRENT" "$REFERENCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -f "${OUT}.log"
```

2 号卡只是命令示例，启动前确认空闲并修改。外层 log 与输出目录同名，不另找内部训练日志。`tail` 的 Ctrl+C 不会停止 nohup 审计。

仅验证元数据、不读 PLY、不创建输出：在脚本末尾加 `--dry-run`。加大统计规模可覆盖 `--blocks 64 --trials 64 --response-blocks 8`；不要先增加全场景渲染负担。

## 下载哪些文件

整个审计目录只含 JSON/CSV 和一张对比图，无模型、PLY 或大量渲染图：

- `plan.json`：checkpoint 哈希、来源训练进度、配置及配对协议。
- `summary.json`、`comparison.csv`、`comparison.png`：旧/新各档核心指标。
- `current/`、`reference/`：各自 `audit.json`、`stage_metrics.csv`、`latent_dimensions.csv`。
- 根目录 `complete.json`：两份都完成后才生成。源 run 不发生任何修改。

把该目录和同名 `.log` 下载回来，就可以定位误差集中在哪个符号段、恢复级或属性响应，之后再决定结构改动，不预先认定必须加深网络或增加训练。
