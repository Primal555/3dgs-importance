# Successive 信道接回逐点档位分配

独立入口：`gaussian_jscc.successive_allocation`。旧版模块化实验与已完成的 Joint A 目录均保留。

## 输入与训练范围

输入采用已完成的 Joint A：`output/truck_successive_joint_gpu0_20261022_105837`。
读取 `checkpoints/final/codec.pt`，保留全部已训练权重、属性统计量、上下文、后置邻居分支和归一化设置。
入口核对完成标记、冻结编码器指纹、配置与场景身份；不接受尚未完成的训练或放开语义编码器的 Joint B。

| 阶段 | 更新次数 | 档位表 | 语义编码器 | 信道编码器／解码器 | 语义解码器 |
| --- | ---: | --- | --- | --- | --- |
| allocation | 2000 | 更新 | 冻结 | 冻结 | 冻结 |
| joint | 1000 | 更新 | 冻结 | 更新 | 更新 |

固定 SNR=10 dB，q0/q1/q2/q3 对应 0/8/24/48 个累计复符号。
XYZ 继续走可靠的 16bit 量化压缩路径；信道的 binary16 块尺度继续实际交付。
分配表使用高保留率初始化，保留后三个正档位等概率；不加载历史分配表。
新建 Adam：keep LR=0.01、条件档位 LR=0.001、eps=1e-15；联合阶段三个可训练编解码模块 LR=1e-5。

## 目标与反向传播

分配阶段使用原 PLY 渲染 RGB MSE，加 `beta=0.005` 的通信代价项。
代价按同场景全 q3 的开销归一化，包含 payload、坐标、档位表代理和实际尺度帧。
联合阶段另加权重为 1 的实际信道 latent MSE：保留点的误差求和，除以全部有效源点数与 latent 维数。
latent 目标只更新编解码器，避免分配器通过删除难恢复点来减小这一辅助损失。
每 4 次联合更新额外做一次固定前缀渲染锚定，独立循环 q1/q2/q3；该步与 mixed 的编解码器梯度取平均。
训练采用 replay，按场景分块重建反向图。

真实前向使用离散档位、实际收到的前缀及其尺度。训练单次采样；部署抽 10 次，全为 q0 才删除，否则取正档位众数。
所有本次关闭的点均保留 MaskGaussian 式零门控反事实渲染反馈，无每块 4 点上限。

Successive 发送端会随档位重新生成增强符号及块尺度，因此分配梯度需要适配这一边界。
反向估计对同一保留集合计算三个正档位候选，用相同噪声起点取得逐点输出差异；候选无编解码器梯度，实际前向不变。
候选在块内将所有保留点统一改成某一正档位，属于带上下文的有偏局部代理，包含块尺度与邻居响应，不能解读为逐点独立升档的精确有限差分。
块尺度的实际字节开销计入记录与目标；其块存在性项保持 detached。payload 概率期望与保留坐标的局部代理提供速率梯度。
不在真实解码端引入源属性或未传输符号。

## 观测与输出

- `metrics/loss.jsonl`：每次更新的渲染、速率、latent、逐模块梯度／更新、采样档位数量、步耗时、CUDA 峰值已分配显存。
- `metrics/validation.jsonl`：每 250 步验证；initial、end_allocation、final 在相同的最多 32 个真正测试视角、4 组配对噪声上评估。
- `metrics/test.jsonl`：最终测试。每处均检查固定 q1/q2/q3 与学习部署，分别对原 PLY 和真实照片计算 PSNR、SSIM、LPIPS。
- `deployment/{initial,end_allocation,final}/tiers.npy`：原始输入 PLY 行序的真实十次采样部署档位；`allocation.json` 含实际占比与概率统计。
- `images/{initial,end_allocation,final}/`：每档保留 2 个代表测试视角；deployment 同时输出接收图、档位贡献着色图及逐像素标签。
- `comparison/summary.json`、`endpoints.csv`：三阶段端点的 12 行精简对比表。
- `charts/tier_shares.png`：实际 q0–q3 部署占比；`quality_rate.png`：实际总信道使用与画质。
- `checkpoints/latest/state.pt`：模型、档位表、两个 Adam、阶段进度、随机状态，可用相同参数加 `--resume` 精确续训。
- `checkpoints/final/codec.pt`、`allocation.pt`：推理权重与绑定模型指纹的分配表。

档位图采用沿光线可见贡献最大的档位标签：q1 蓝、q2 橙、q3 紫。未覆盖背景为黑色；已删除 q0 点没有可见 splat，不将背景直接解释为 q0 分布。
总成本包含模拟 payload 与实测坐标／完整元数据／尺度流；数字侧流按源实验的净 bit/use 假设折算，共享模型权重不计。
不得把复符号的 float32 存储字节当作无线传输字节。原始字节与假设均保留在指标中。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/truck_successive_joint_gpu0_20261022_105837"
GPU=0  # 按实际空闲卡修改
OUT="$PWD/output/truck_successive_allocation_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_successive_allocation_truck.sh "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

SSH 断开后 nohup 训练继续；Ctrl+C 退出 tail 不会停止后台训练。
若数据目录迁移，可加 `--manifest /path/to/multiscene_pruned.json`，场景内容与相机拆分仍须一致。
配置预检可加 `--dry-run`；不生成输出、不训练、不渲染。
续训时使用原 RUN 与 OUT，并在脚本最后加 `--resume`；仍指定总预算 2000+1000。

## 同一步复用加速

此入口默认启用 `--allocation-reuse`，原有实验入口保持原样。采样、噪声、目标、梯度估计、
q0 反馈覆盖范围、精确通信成本与十次部署规则均保持不变；无需重训输入编解码器。

- 同一更新内，实际档位与三个正档位候选的保留点集合相同，共用冻结且处于 eval 模式的语义编码结果。
- q0 反事实的保留点集合不同，独立编码并计算一次，缓存其关闭点输出；不借用实际集合的上下文。
- 冻结分配阶段缓存实际恢复结果，replay 计算三档候选与概率梯度时复用该结果。
- 联合阶段重新计算实际信道与语义解码图，保留真实编解码器梯度；仅复用冻结语义表示及 detached q0 反馈。
- 候选仍逐批计算，不将三份全场景候选输出常驻显存。每批 replay 后立即释放对应缓存，下一次更新重新建立。

含 q0 的一个批次，局部分配步骤中的语义编码调用由 7 次减少到 2 次；信道编解码／属性恢复
在冻结阶段由 7 次减少到 5 次、联合阶段由 7 次减少到 6 次。无 q0 时相应减少更多。
额外固定前缀锚定仍独立执行，不包含在这些调用数内。

缓存增加显存但不保存网络激活图。FP32 缓存约为
`4 × [填充后点数 × (latent维数 + 冻结阶段特征维数) + 本次q0数 × 原始属性维数]` 字节。
全场景填充、渲染与单批网络工作区另计；实际峰值见日志，不按调用次数推算加速倍数。

`--no-allocation-reuse` 可切回原计算路径做速度对照；这两个路径都可从同一旧版 latest checkpoint
加 `--resume` 续训，保留概率表、模型、两个 Adam、进度和随机状态。缓存不写入 checkpoint。
执行开关独立记录在 `allocation_runtime.json` 和每步日志中，不改变历史训练参数匹配规则。

### 耗时记录

默认 `--profile-every 10`，另记录每阶段第一步与最后一步；`--profile-every 0` 关闭详细计时。
CUDA 内部小段采用 event 汇总，避免每个批次都同步。日志新增：

- `semantic_encode_seconds`：复用路径的实际保留集合语义编码，不含 q0 独立编码。
- `candidate_decode_seconds`、`actual_decode_seconds`：三候选／实际恢复路径的累计时间；关闭复用时包含各自的语义编码。
- `q0_shadow_seconds`：q0 独立编码和恢复。
- `codec_forward_seconds`、`codec_replay_backward_seconds`：全场景首次构建与 replay 反向。
- `render_forward_seconds`、`render_backward_seconds`：渲染；流式多视角的前者包含逐视角反向，后者主要为交接开销。
- `rate_accounting_seconds`、`deployment_rate_gradient_seconds`：原有实测通信成本与概率速率梯度计算。
- `prefix_anchor_seconds`、`optimizer_step_seconds`：联合前缀锚定与 Adam 更新。
- `allocation_cache_peak_mib`、`allocation_reuse_enabled`、`allocation_profiled`：缓存张量容量峰值与执行状态。

细分路径与全场景阶段计时存在包含关系，不能将全部字段相加。比较速度时应同时检查
`step_seconds`、采样 q0 数、是否执行了前缀锚定以及显卡是否共享。
本地使用真实小型 PointNet++ 双端模型及可微合成渲染验证数值与更新一致性；服务器 CUDA 光栅化速度另行实测。
