# 显式坐标交付对照实验

本实验检验：在当前 render-first 随机初始化流程中，绕开学习式 XYZ 恢复，
能否改善属性分支的有效梯度和渲染恢复。它**不能单独证明**历史多项参数损失存在
梯度竞争，也不是相同总通信预算下的优劣比较。

## 三组设置

| 设置 | 接收端位置 | 其余属性 | 坐标费用 |
|---|---|---|---|
| learned | 原来的 JSCC XYZ 输出 | 原有 JSCC | 无额外逐点坐标 |
| float32 | 已交付的 float32 归一化坐标 | 原有 JSCC | 每非零 Gaussian 96 bit + framing |
| quantized | 已交付的 b-bit/轴归一化坐标 | 原有 JSCC | 每非零 Gaussian 3b bit + framing |

默认 b=12，仅作为精度测试设置，不声称已经足够。float32 控制组也计费；
归一化/反归一化存在浮点舍入，不等于原始世界坐标逐位相同。
位置只替换输出，不新增位置 residual、不改变属性 decoder context，避免同时改变多个结构。
发送端仍可使用原始 XYZ 提取上下文；属性接收端不能读取干净属性。
旁路 XYZ head 保留相同随机初始化但不训练；它的零梯度是设计结果，关注 attribute_heads。
日志中的 scene_gradient_norms.xyz 是渲染器对场景坐标的导数，不是旁路 XYZ head 的参数梯度。

三组均从相同随机种子、相同参数初始化开始；固定 AWGN 10 dB、0 bootstrap、
0 joint mask steps、相同 payload 0/8/16/32、相同训练/验证相机和噪声调度。
量化不消耗随机数。训练依旧是完整场景，缩小的是相机集合和步数。
暂禁显式坐标模式的 joint mask 与 bootstrap，避免沿用不含坐标开销的 rate penalty。

## 运行（服务器）

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_position_delivery_$(date +%Y%m%d_%H%M%S)"
# 仅示例使用 GPU 0；请先确认空闲。
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_position_delivery.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

默认依次执行三组，每组 300 个渲染步骤，12 个训练视角中每步采样 2 个，
4 个不重叠验证视角 × 2 次固定噪声试验；每 25 步验证，blocks-per-batch=32。
可用 RENDER_STEPS、POSITION_BITS、PLY、SCENE 等环境变量覆盖设置。
本脚本强制随机初始化，无视继承的 INIT；不修改普通训练的默认架构。
不要同时运行三组，脚本串行执行，便于控制显存。

## 查看什么

- `summary.json`：各组初始/最后/最佳渲染 MSE、零梯度比例。
- `validation_comparison.csv`：每个验证步骤、每个档位的质量和费用。
- `{learned,float32,quantized}/loss.jsonl`：全部优化日志和分支梯度。
- 各组 `charts/validation_quality.png`、`charts/optimization.png`。
- 各组 `validation_images/000000/` 和 `validation_images/000300/`：初始与最后对比。
- 各组 `codec_best_render.pt` 和 `codec.pt`：验证选择与最后一步，二者未必相同。

float32 明显改善而 learned 不改善：支持位置恢复路径是瓶颈，不能确定瓶颈就是损失权重。
float32 改善而 quantized 不改善：说明当前全局量化精度可能不足。
两种显式坐标都不改善：位置不是唯一问题，还应考虑属性初始化、光栅化和梯度链。
短测试用于定位，未收敛不能据此否定方案。

## 开销和协议边界

`xyz.bin` 是真正的独立 side stream：只包含 q>0 的 XYZ，顺序与包内行相同。
量化整数紧凑按位打包，无熵编码；float32 为 little-endian。18 字节头部含模式、
精度、点数和 CRC。q0 不发送坐标，接收端依据已有档位序列填回槽位。
接收端只需 checkpoint 和 packet，不读取输入 PLY。训练张量量化与二进制解码一致。

假设数字 side stream 可靠交付，**没有实现 FEC 或模拟坐标包错误**。CRC 只是文件校验。
训练日志默认以 2 个净信息 bit / 复信道使用估算费用，可用
POSITION_NET_BITS_PER_USE 覆盖，不表示某个实际编码在 10 dB 已验证可靠。
`payload_plus_position_uses_per_source_gaussian` 包含 XYZ framing，但不含原有全局 bbox、
档位图与模型标识元数据；`symbols_per_source_gaussian` 始终仅为 JSCC payload。

既有 `benchmark_codec.py` 可直接评估这些新 checkpoint。真实包导出时会计量
xyz.bin 的字节数，并在 total_channel_uses 中加入坐标和原有元数据开销。
若要与训练日志的 2 bit/use 对齐，评估指定
`--metadata-code-rate 0.5 --metadata-modulation-bits 4`；这仍是费用模型，不是 FEC 仿真。
共享模型及归一化统计的交付方式沿用现有框架，不应把本实验视为无先验跨场景结果。

第一阶段旧参数目标是对输入 PLY 对应行的参数重建，不是渲染损失；当前可选 bootstrap
是归一化特征的 SmoothL1，而当前 render 阶段是多视图 RGB MSE，对应 source PLY 渲染。
本测试不重新引入历史加权参数目标。
