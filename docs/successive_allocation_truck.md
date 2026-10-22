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
