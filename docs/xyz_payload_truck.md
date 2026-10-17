# XYZ 与属性共用信道的 Truck 独立实验

本实验架构为 `xyz_payload_jscc_v1`，不替换现有可靠 16bit XYZ 实验，不训练档位分配器。

## 实际链路

发送端将每个 Gaussian 的三个归一化中心坐标，与语义编码器产生的 48 维属性表示拼接：

`y = [u_x, u_y, u_z, z_attr(48)]`，共 51 个实数。

三个坐标不通过额外的位置自编码器，也不单独数字交付。它们和属性表示一起进入渐进信道编码器。
q1/q2/q3 分别交付累计 **8/24/48 个复符号**，包含 XYZ 和属性的全部开销，不是各层新增数量。
信道解码器先从接收前缀恢复完整 51 维表示，再由前三维还原坐标、其余 48 维恢复属性。
增强层重新读取完整接收前缀并累加表示修正；并不保证每一张图的 PSNR 严格单调递增。

全局归一化：`u = 2*(XYZ-c)/s`，`c` 是包围盒中心，`s` 是最大轴跨度。
解码端使用包头中的全局包围盒恢复世界坐标，不读取原始逐点 XYZ，不钳制出界预测。
语义两端保留 PointNet++ 式空间上下文；接收端上下文只使用恢复出的 XYZ。
信道空间去噪和邻居协作模块关闭，避免依赖尚未可靠恢复的坐标。

**clean 坐标准确不代表网络学会了定位**：clean 直接交付未受损的 51 维表示，前三维天然包含坐标。
真正的位置恢复能力必须看有限符号、尤其 AWGN 下的误差和渲染质量。

## 训练流程与默认配置

| 阶段 | 步数 | 更新模块 | 目标 |
| --- | ---: | --- | --- |
| semantic_bootstrap | 4000 | 语义编码器、语义解码器 | 单椭球局部响应恢复；每四步替换为一次方向颜色恢复 |
| semantic_render | 1000 | 语义编码器、语义解码器 | clean 路径整场景渲染 MSE |
| channel_bootstrap | 4000 | 信道编码器、信道解码器 | 固定语义表示，恢复 XYZ 与属性 latent |
| channel_render | 2000 | 信道编码器、信道解码器、语义解码器 | 渲染 MSE + 表示恢复约束；语义编码器冻结 |

所有模块学习率固定 `1e-4`，信道固定 10 dB。两个通信阶段各自独立轮换 q1/q2/q3/mixed，
mixed 是随机正档位，不是已学习分配，也不删点。颜色步骤与档位轮换不存在共享计数器。
颜色步骤使用 8 个方向，一半来自训练相机、一半来自球面随机方向，仅颜色头与共享语义路径更新；
非颜色头不接受 Adam 动量更新，但共享特征改变仍可能间接改变它们的预测。
全场景反传使用 replay；没有新增混合显存机制，也不启用固定梯度裁剪。

信道恢复损失分组取均值，防止三维坐标被 48 维属性平均稀释：

`L_payload = xyz_weight * mean((u_hat-u)^2) + mean((z_hat-z)^2)`

联合阶段：`L = MSE(rendered, source_PLY_render) + channel_latent_weight * L_payload`。
两个系数默认均为 1，可通过 `--xyz-latent-weight` / `--channel-latent-weight` 显式修改，
属于公开工程超参数，不宣称最优。日志同时记录分组 MSE、NMSE、普通 51 维平均 MSE、各档误差及梯度/更新量。
`--joint-semantic-encoder` 是明确的可选实验开关，默认不启用。

## 从随机权重启动（服务器 Bash）

`--run` 仅提供 Truck PLY、相机划分、网络宽度等元数据，**不加载该目录的旧模型权重**。
输入目录应是已存在的多场景或模块化实验，且 `training.json` 及所引用的场景元数据可用。
不要使用旧架构 checkpoint 初始化本实验。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
RUN="$PWD/output/multiscene_pruned_aligned_gpu0_20261010_194134"
OUT="$PWD/output/truck_xyz_payload_gpu2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/output"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_xyz_payload_truck.sh "$RUN" "$OUT" \
  >"${OUT}.log" 2>&1 </dev/null &
echo $! >"${OUT}.pid"
tail -F "${OUT}.log"
```

GPU 2 只是示例，请先确认空闲。`nohup` 不依赖本地终端持续连接，`Ctrl+C` 退出 tail 不会中断训练。
脚本启用 PSNR、SSIM、LPIPS；LPIPS 需要安装 `lpips` 并能加载预训练 VGG 权重，缺失时明确报错，不伪造分数。
临时仅检查管线可追加 `--no-lpips`；此时不能宣称完成 LPIPS 评估。
路径变化可提供 `--manifest /path/to/manifest.json`，但 PLY 内容及相机划分必须与源记录一致。

精确续训使用**同一 OUT**与原参数，恢复 Adam、各阶段步数和 RNG：

```bash
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_xyz_payload_truck.sh "$RUN" "$OUT" \
  --resume "$OUT/checkpoints/latest/codec.pt" \
  >>"${OUT}.log" 2>&1 </dev/null &
echo $! >"${OUT}.pid"
```

若上次显式修改过步数、学习率、损失等参数，续训时必须重复相同参数；不兼容配置会被拒绝。
checkpoint 还必须与 OUT 的实验身份一致，不能把另一轮训练的 Adam/学习率静默接入。
从较早 checkpoint 回退续训时，过期的完成标记和后期产物会归档，阶段端点重新生成。
`--init-semantic` 只允许显式加载本架构已完成语义渲染阶段的权重，跳过语义阶段，信道随机初始化；
`--init-codec` 显式加载同架构已训练信道的全模型，两者均为新 OUT、重置 Adam 与阶段进度，不是精确续训。

## 输出与判读

所有内容位于一个 OUT：

- `training.json`、`progress.json`、成功后的 `complete.json`；失败留 `failure.json`，不伪装完成。
- `metrics/loss.jsonl`：逐步目标、分组误差、模块梯度/更新量、耗时、峰值显存。
- `metrics/validation.jsonl`：默认每 500 步以及阶段端点评估。
- `metrics/test.jsonl`：最终真实 test 相机评估，不用训练视角冒充测试集。
- `checkpoints/{initial,latest,end_*,final}/codec.pt`：模型、优化器、随机状态、场景指纹。
- `images/{initial,semantic_render_end,channel_bootstrap_end,final}/`：代表性首尾两个验证视角，
  保存 clean 与 AWGN q1/q2/q3/mixed 图，不保存每次验证图，也不保存有限无噪声的重复图。
- `charts/`：目标/画质/位置误差/颜色误差/分组恢复误差与训练耗时曲线。
- `packet/`：最终 q3 模拟接收包，包含 `metadata.bin`、`received.npy`、`cost.json`，**没有 `xyz.bin`**。

评估区分三种情况：clean（绕过信道，无有限符号约束）；finite_noiseless（有限符号，关闭 AWGN）；
awgn（有限符号、10 dB）。同时对原 PLY 渲染和真实照片统计 PSNR/SSIM/LPIPS，并记录颜色偏置。
位置指标包括世界坐标 RMSE、逐轴误差、P50/P95、整体偏移、去全局/块平移后的残差。
`teacher_xyz_source_psnr`、`center_only_source_psnr` 仅用于误差归因，明确是替换真实属性/真实位置的诊断，
不能当作部署画质。mixed 也不代表档位分配器性能。

接收包可在**没有源 PLY、相机数据**的环境中恢复场景，需预先共享相同模型权重：

```python
from gaussian_jscc.xyz_payload_checkpoint import load_checkpoint
from gaussian_jscc.xyz_payload_transport import receive_packet
from gaussian_jscc.data import write_ply
model = load_checkpoint("OUT/checkpoints/final/codec.pt", "cuda")
raw, header = receive_packet(model, "OUT/packet")
write_ply("received.ply", raw.cpu(), model.cfg.sh_degree)
```

通信成本以实际累计复符号数计量；全局包头和档位表统计真实压缩字节，换算信道使用次数时显式给出
`net_bits_per_use_assumption`（继承源实验，默认 2），这不是通用物理常数。
独立坐标字节为零不等于坐标免费：它已占用上述总符号预算。
`received.npy` 的 float32 存储体积不是实际无线带宽；共享模型成本不计入本次包成本，元数据假设可靠且不模拟 FEC/丢包。
元数据另包含固定 40 字节的 CRC/SHA256 完整性封装，已经计入成本；用于检测模拟接收文件损坏，
不等同于为真实无线信道增加了纠错能力，也不会将 AWGN 扰动检测成文件错误。

本机验收使用 CPU 单元测试及可微合成渲染覆盖训练/续训/传输流程；不据此宣称 CUDA 光栅化或实际画质达标。
