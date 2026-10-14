# Truck 纯语义方向颜色保真实验

目的：在不经过任何信道模块、没有噪声和前缀截取的条件下，检查原场景的颜色与属性能否准确恢复。采用已选定的两端 PointNet++ 式空间聚合；网络宽度、latent 48 维、处理块大小等沿用来源配置。来源目录仅提供场景身份、路径、相机划分和宽度，不读取其中的网络权重。随机初始化，两个语义模块学习率固定 1e-4。

## 更新次数与目标

- 第 1–4000 步：语义初始化；每四步中前三步为原局部响应 RGB MSE，第四步为独立方向颜色 MSE。即 3000 次局部响应、1000 次方向颜色更新，没有额外优化器更新，也没有两项人为加权相加。
- 第 4001–5000 步：1000 次原 PLY 全场景渲染 RGB MSE，replay 反向传播。本次不插入额外颜色更新，便于观察渲染训练是否再次造成偏色。
- 两个信道阶段均为零步；分配器不训练，真实点不删除。语义 latent 直接交给语义解码器。clean 不是有限符号无噪声通信结果，也没有可宣称的 q1/q2/q3 码率收益。
- XYZ 继续沿用每轴 16bit 的可靠量化路径，位置不预测。

## 独立方向颜色监督

使用反标准化后的原始 DC、SH，按现有渲染器的相机到点方向计算 `0.5 + eval_sh(DC,SH,d)`；预测与原场景的 RGB 均方误差在非负/上界截断之前计算。目标不乘透明度或 footprint，不依赖位置、旋转或尺度。原场景属性只作训练教师，不注入接收端。

每次颜色更新使用八个方向：四个来自训练相机中心（按点计算真实方向，可重复采样），四个为均匀球面随机方向。测试与验证相机不用于训练方向。共享语义编解码网络与 DC/SH 头更新；透明度、尺度、旋转头的梯度设为 None，避免已有 Adam 动量在颜色更新中继续改变这些头。共享特征仍会变化，因此不声称其他属性的预测值完全不动。

4000/1000、1:3 更新比例、八个方向、学习率均为明确的实验设置，不宣称最优。标准实验未启用这些新开关时行为保持原样。可用同一启动脚本加 `--semantic-color-every 0` 做同更新预算的旧目标对照。

## 记录与图片

每 500 步保存 checkpoint 并验证固定的留出训练集相机：原 PLY / 照片 PSNR、SSIM、RGB 有符号偏差、红绿及绿蓝通道差异 RMSE、各属性标准化 MSE。后两项不是感知 Delta-E；全图偏差可能相互抵消，应结合保存图像观察。

图片只保留三个时刻，每次两个代表验证视角（第一及最后视角）：

- `images/initial`：随机初始化；
- `images/end_semantic_bootstrap`：4000 步属性恢复结束；
- `images/final`：5000 步渲染训练结束。

`metrics/loss.jsonl` 按 `objective` 区分两种初始化目标，不直接混为一条可排名 loss；颜色步骤还记录被抽取的训练相机索引。`charts/objectives_by_phase.png` 分目标绘制，`charts/validation_paths.png` 为 clean PSNR，`charts/clean_quality.png` 同时展示原 PLY / 照片的 PSNR、SSIM，`charts/color_errors.png` 为颜色指标。最终在真正 test 相机集上只评估 clean，记录 `metrics/test.jsonl`。所有结果在同一 OUT 目录，无随机信道评估与 SNR 扫描。

## 后台启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_modular_latent_gpu2_20261013_113847"
OUT="$PWD/output/truck_clean_color_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_clean_color_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

GPU 2 只是示例，启动前自行确认空闲卡。Ctrl+C 只停止 tail，不会停止后台训练；关闭 SSH 连接也不会因终端挂起中断此 nohup 任务。

精确续训：在同一 OUT 下重新启动脚本，追加 `--resume "$OUT/checkpoints/latest/codec.pt"`。更新预算、颜色调度、配置与来源身份必须保持一致。不能把该实验误解为恢复旧模型或噪声鲁棒性已经改善。
