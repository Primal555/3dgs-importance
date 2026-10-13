# Modular JSCC：高 SNR 退化定位与条件输入修正

本测试使用已训练的 Truck 检查点，固定权重、相机、档位、符号槽和噪声随机数，比较实际信道 SNR 与网络条件输入之间的关系。没有优化器更新。

当前检查点仅在10 dB训练。此前评估20 dB时同时改变实际噪声、编码器条件和解码器条件，导致无法分辨退化来自哪里。

## 默认推理保护

模型从检查点读取 `training.progress.train_snrs`。两个网络条件默认限制在该范围内，实际噪声照常按实际 SNR 生成。单一10 dB训练时，默认条件固定为10 dB。多SNR训练时只限制范围外的外推，不改变范围内的条件。

这项修正避免向单SNR模型输入未训练的条件值，不保证 PSNR 随信道改善严格单调。若固定条件后仍退化，需要再检查对特定噪声强度的适应，不把结果强行归为条件问题。

发送包新增 `tx_condition_snr_db`、`rx_condition_snr_db`，接收端使用包中的条件。老包没有这两个字段时，继续使用原来的 `snr_db`，保证既有包的解读不变。网络参数、符号预算与坐标路径不变。

## 四个受控组

| 组 | 编码器条件 | 解码器条件 | 实际噪声 |
|---|---|---|---|
| matched | 跟随实际 SNR | 跟随实际 SNR | 扫描0/5/10/15/20 dB |
| fixed_both | 固定10 dB | 固定10 dB | 同一组噪声 |
| fixed_tx | 固定10 dB | 跟随实际 SNR | 同一组噪声 |
| fixed_rx | 跟随实际 SNR | 固定10 dB | 同一组噪声 |

默认测试全部32个test视角，两次噪声试验，覆盖q1/q2/q3/mixed。mixed是同一随机正档位图，不重新训练分配器。finite_noiseless仍经过两个信道模块；clean绕过两者。

先看 `fixed_both` 对比 `matched`，再用 `fixed_tx` 和 `fixed_rx` 定位条件输入的编码/解码端贡献。实际10 dB处四组应相同，作为配对检查。两次噪声试验不等同于两个训练种子。

## 服务器启动

确认示例GPU 2可用后执行：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/truck_modular_latent_gpu2_20261013_113847"
OUT="$RUN/snr_controls_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_modular_snr_controls.sh "$RUN" "$OUT" \
  >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

默认读取 `checkpoints/final/codec.pt`。用 `--checkpoint`可指定其他检查点，场景指纹、相机划分和网络配置必须一致。
`--test-views 4`可先做较快的均匀相机子集测试；最终对比应使用默认全部相机。
源文件路径迁移后用 `--manifest /absolute/path/manifest.json`覆盖，仍验证场景身份。
运行所需的场景指纹、相机划分和渲染配置直接读取本轮元数据，不要求原来用于初始化的结果目录仍存在。
新检查点的训练SNR记录优先于run目录记录；只有检查点未记录时才从run补充。
旧run若未保存数字侧信息的净比特效率，原目录又已迁移，则通信成本换算沿用2 bit/use假设，并在 `experiment.json`标明来源；画质对照不依赖这个换算。
默认参考条件取训练SNR中位数，当前为10 dB；`--reference-snr 10`可显式指定。

## 保存结果

- `experiment.json`：源检查点、训练SNR、实际扫描范围、参考条件和测试视角。
- `metrics/test.jsonl`：每个组、档位、SNR的PSNR、SSIM、特征MSE/NMSE及逐视角结果。
- `summary.json`：同SNR下四组结果和对旧方式的dB增益。
- `charts/snr_controls.png`：q1/q2/q3的PSNR与特征NMSE曲线。
- `complete.json`：全部组评估完成；更新次数为0、权重未变化。

日志分别标记 `actual_snr`、`tx_condition_snr`、`rx_condition_snr`、`condition_policy`。训练仍使用原有两项损失，不增加训练阶段或自动重训。

## 后续讨论议题

高SNR问题量化后，再讨论q2→q3增强层的独立增量，以及将PointNet++的分层邻域聚合用于语义编码器、配合已交付XYZ辅助解码的结构实验。本次不提前实施这两个结构改动。
