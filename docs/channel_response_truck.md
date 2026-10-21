# 信道恢复：y 保真＋局部 Gaussian 视觉响应保真

独立入口 `gaussian_jscc.channel_response_training`，脚本
`scripts/test_channel_response_truck.sh`。不覆盖任何已有实验。

本轮只训练信道编码器、信道解码器。语义编码器、语义解码器及属性标准化参数
全程冻结，固定10 dB AWGN，可靠16bit XYZ，8／24／48复符号，逐块功率归一化。
没有联合微调、档位分配、q0删点、梯度裁剪或学习率调度。
整场景渲染仅用于评估，不反向训练。

## 初始化与训练预算

从良好的**信道恢复结束**检查点出发，例如：

`output/truck_power_stability_20261021_100748/block_continuation/block/checkpoints/end_channel_bootstrap/codec.pt`

向脚本传入稳定性实验根目录或其 `block_continuation/block` 子目录即可。
拒绝语义解码器适应后的检查点、联合探针末端或逐点归一化组。
沿用全部网络权重、符号预算及语义教师，不重新训练语义表示，也不随机重建信道。
目标改变，所以建立**新的信道 Adam**，默认LR=1e-4，新增2000次真实优化器更新。
日志全局步数继承来源8000，因而默认结束于10000；本轮更新数看 `phase_step`。
q1→q2→q3→mixed均衡轮换；mixed是随机正档位，不是学习后的分配。
可用 `--steps 4000`、`--lr 1e-5` 等显式覆盖。无联合训练参数或自动后续阶段。

### 信道从头8000步（新增独立启动脚本）

`scripts/test_channel_response_scratch_truck.sh` 显式传入 `--channel-init random --steps 8000`。
默认旧脚本仍使用 `--channel-init retain`，原2000步续训及其断点恢复不变。

从同一个源检查点**只复制语义编码器、语义解码器和attr_mean/attr_std**；
源文件仍要求是已验证的逐块信道结束端点，以保持同一配置、教师和数据划分，
但其信道编码器、信道解码器（含所有补偿支路）、功率状态、Adam和已完成步数
均不复制。信道完整结构用原配置按 `--seed` 重新初始化，建立新Adam。
随机初始化后的原始训练步数、channel_bootstrap计数均从0开始，最终为8000，
不是8000＋8000=16000。日志 `source_step=8000` 只表示教师来源记录，不表示继承信道训练。
`plan.json`明确记录 `training_start_step=0`、`source_channel_weights_loaded=false`；
`initialization.json`保存初始化种子、两类权重摘要和空Adam状态计数。

在本轮随机初值上重新执行一次梯度尺度校准，不读取上次续训的lambda或校准结果。
语义两端全程冻结，不加热身阶段，不接联合渲染微调，损失从第1步就是y MSE＋局部响应。
无噪声/10dB渲染仅评估；仍每500步保存验证数值，只保存初始/最终图像。
训练批次、噪声及方向按本轮step1开始采样，精确续训只恢复本轮latest，
不会再次随机化信道或重新校准。恢复时必须保留 `--channel-init random`；专用脚本自动保留。

启动方式（GPU0仅为示例）：

```bash
RUN="$PWD/output/truck_power_stability_20261021_100748"
OUT="$PWD/output/truck_channel_response_scratch_gpu0_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_channel_response_scratch_truck.sh "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

这验证新损失从随机信道权重学习的能力，不是新增一组旧MSE控制。
历史8000步组可作实用参照，但因采样/优化路径不同，不应宣称严格单变量损失消融。

## 损失及教师

`L = MSE(y_hat, stop_gradient(y)) + lambda * L_response`。

固定语义解码器D0，分别得到 `D0(y)` 和 `D0(y_hat)`。它们收到完全相同的
可靠量化XYZ，且始终在完整原始Morton块内解码，保留PointNet式上下文；
不能把有效点抽出后重组邻域。只有完成属性解码后，才排除padding并展开逐点响应。

响应复用 `local_response_loss`：多观察方向、局部采样位置及黑／白两背景下的RGB MSE，
共同反映投影形状、旋转、尺度、不透明度和DC／SH颜色损伤。
默认每步8个由独立种子生成的各向同性方向，非测试相机。
保留旧局部响应实现中参考/预测双覆盖采样、停止采样网格梯度、协方差数值floor等定义。
它是局部、近似正交投影响应，不是完整透视光栅化、遮挡合成或面积加权的整场景误差。
不承诺新损失必然改善PSNR，或自动根治所有大梯度。

教师来自当前干净语义表示，不是原PLY属性。
这隔离信道新增损伤，但也意味着不能靠这一项修正语义教师原有误差。
冻结参数不等于关闭预测分支的autograd：`D0(y)`分支no_grad，
`D0(y_hat)`必须保留到信道两端的梯度。没有额外信息交付接收端，
冻结教师和参考响应仅用于发送端离线训练。

## lambda 的一次性校准

默认训练开始前只做一次，使用初始模型和4个正常训练块批次，覆盖q1/q2/q3/mixed；
来源是训练场景的教师y，不读取验证/测试图像。每批单独向完整块的y_hat求两项梯度，
池化其平方和，设置：

`lambda = sqrt(sum ||dL_y/dy_hat||² / sum ||dL_response/dy_hat||²)`。

同形状梯度的元素数相同，因此等价于池化RMS之比。
不更新参数、Adam状态或功率统计，不生成逐维手工权重；不动态重平衡。
校准数据和噪声使用独立可重复序列，随后lambda固定到训练结束。
校准失败／零梯度时明确报错，不静默替换为1。
可用 `--response-weight VALUE` 固定覆盖（0可作纯MSE控制，但仍记录响应）。

这是**初始梯度尺度匹配的工程选择**，不是最优超参数的文献结论。
lambda数值很大也不意味着响应项支配训练，必须结合原始响应、加权响应和梯度看。
完整记录保存在 `calibration.json`，并进入每份可续训checkpoint。

## 评估与输出

- `metrics/loss.jsonl`：逐步y MSE/NMSE、原始/加权响应、黑白分项、固定lambda、
  各模块梯度/实际更新量、时间和CUDA峰值显存；只有两信道模块应有更新。
- `evaluation/initial`、`final`：相同32个实际test视角及4次配对噪声；
  clean、有限无噪声、10dB q1/q2/q3/mixed，原PLY/照片PSNR、SSIM、LPIPS/VGG及颜色误差。
- 每500步使用原有4个留出验证相机，只保存数值；不要与32视角端点混成同一曲线排名。
- 仅保存初始和最终代表相机的效果图，包括mixed；不保存中途图、PLY和噪声大张量。
- `comparison/summary.json`、`endpoints.csv`：配对端点对比。
- `checkpoints/initial`、`latest`、`end_channel_bootstrap`、`final`：训练状态；
  latest每100步覆盖保存。`complete.json`只有完整训练及最终评估结束才生成。

检查源checkpoint哈希、场景指纹及相机划分；保存时验证冻结语义状态完全一致。
同一输出和全部参数加 `--resume` 恢复Adam、固定lambda、RNG、更新游标及未完成评估。
不会因重启再次校准lambda，不修改源目录。
初始/最终clean应该一致；但不能用“loss变小”替代实际三档画质改善。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_power_stability_20261021_100748"
OUT="$PWD/output/truck_channel_response_gpu0_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_channel_response_truck.sh "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

GPU0只是示例，启动前选择空闲卡。断开电脑不会停止nohup任务。
断点恢复使用原RUN、原OUT及原全部参数，命令末尾加 `--resume`，日志建议 `>>` 追加。
可先加 `--dry-run` 只读核对来源、预算与计划，不加载场景或创建输出。

本机验证仅覆盖真实PyTorch网络/局部响应的梯度、padding、一次校准、冻结及精确续训，
整场景集成测试用合成渲染替代CUDA光栅化；实际Truck画质和显存须由服务器实验确认。
