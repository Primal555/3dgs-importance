# 冻结信道的语义解码器适应实验

这是一条独立实验入口，不替换原有信道训练或联合训练入口。

## 训练内容

- 初始化：只读取已完成的 `checkpoints/end_channel_bootstrap/codec.pt`。
  不使用短程探针或失败联合训练的 `latest`。传入 stability 根目录时，
  自动选择 `block_continuation/block`。
- 保留既有网络、8/24/48 复符号档位及可靠 16bit XYZ 交付。
- 冻结语义编码器、信道编码器、信道解码器，包括缓冲区和功率统计。
  冻结模块保持 eval 模式，信道噪声仍按每次更新重新采样。
- 仅语义解码器使用新的 Adam，固定学习率 `1e-5`，默认 1000 次更新。
  不继承或运行原信道 Adam。
- 固定 10 dB AWGN，依次轮换 q1/q2/q3/mixed；mixed 是正档位随机分布，
  不是学习分配器。本实验不训练删点或档位概率。
- 损失保持 `RGB MSE + (1-SSIM)/3`，目标为同相机的原 PLY 渲染。
  latent MSE/NMSE 只作为监测，不加入训练损失。
- 使用完整场景、多视角 replay 反向，不新增梯度裁剪或拒绝更新。

## 输出与验证

- `metrics/loss.jsonl`：真实图像损失、分模块梯度和更新范数、latent 误差、
  显存峰值及耗时。前三个冻结模块的参数更新应始终为零。
- 前16步以及之后每50步，使用相同相机／档位／噪声重测更新后的损失，
  记录为 `same_input`，用于观察是否仍存在首次过冲。
- 中途每250步在原留出验证视角检查 clean、有限无噪声、10 dB q1/q2/q3/mixed；
  第1000步直接做下述完整测试集评估，不再重复跑留出验证集。
- 初始及最终使用相同的32个实际测试视角、4次配对AWGN采样，记录相对于
  原 PLY 与真实照片的 PSNR、SSIM、LPIPS，以及颜色误差和 latent 指标。
  仅初始／最终保存图像，每种路径只保存首尾两个测试视角，包括 mixed。
- `comparison/endpoints.csv`、`comparison/summary.json`：初始／最终同口径对比。
  在配对评估下，latent MSE 必须保持相同；发生变化会报错，不以近似相等掩盖。
- `checkpoints/initial`、`latest`、`final`：保存完整模型及仅语义解码器的 Adam。
  每100步及验证前保存；`--resume` 恢复原配置、冻结范围、Adam、RNG及评估游标。
  异常中断后从最近保存点继续，不保证保留最后一次保存之后尚未持久化的更新。

冻结信道避免渲染梯度破坏信道映射，但并不保证语义解码器的形状反向不会爆炸。
也不要求适应后的解码器在 clean 上仍达到原上限；部署噪声路径是主要评估对象。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_power_stability_20261021_100748"
OUT="$PWD/output/truck_decoder_adapt_gpu3_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_decoder_adaptation_truck.sh "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

GPU 3 仅为示例，启动前确认空闲。不将任何其他用户任务作为可清理资源。
所有本轮结果位于单一 `OUT` 下，后台标准输出为同名 `${OUT}.log`。

中断后，使用原来的 `OUT` 和其他参数，在脚本最后添加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_decoder_adaptation_truck.sh "$RUN" "$OUT" --resume >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
```

续训前确保原进程已退出；不要同时向同一个输出目录启动两个写入进程。
`--resume` 是精确恢复原定1000步预算，不是额外追加1000步。
先加 `--dry-run` 可只验证源端点并打印计划，不创建输出目录。
若场景数据迁移，使用 `--manifest` 指定新路径，PLY身份与相机划分仍须一致。

## 本地实现验证

```powershell
python -m unittest discover -s tests -p test_decoder_adaptation.py -v
```

CPU测试使用真实编解码器、噪声、自动微分、优化器和检查点。
仅把CUDA渲染替换成可微合成渲染，不代表验证了实际场景画质或CUDA数值稳定性。
