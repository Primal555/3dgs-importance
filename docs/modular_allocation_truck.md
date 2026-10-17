# 已训练模块化三档模型接回逐点分配

独立入口，不覆盖之前的信道、语义重建或八档预算实验。
默认来源是 `truck_channel_joint2000_gpu3_20261016_110009/checkpoints/final/codec.pt`。
必须为完整训练过的 progressive modular 8/24/48 checkpoint；拒绝静默重置网络或修改预算。

## 两个阶段

| 阶段 | 实际更新次数 | 更新对象 | 目标 |
| --- | ---: | --- | --- |
| allocation | 2000 | 新建逐点 keep/tier 概率表 | 原 PLY 渲染 RGB MSE + β 归一化通信代价 |
| joint | 1000 | 同一概率表、语义编码器/解码器、信道编码器/解码器 | 同上；额外保留固定前缀锚定 |

固定 10 dB AWGN，16bit XYZ 可靠交付。原始数据和来源权重只读。
初始化 keep=0.99、保留后的 q1/q2/q3 等概率；不把历史 MaskGaussian prior 当作当前分配收益。
单次硬采样训练、十次采样部署：全部 q0 才删除，否则按正档位众数选档；平票按概率、再按固定随机种子。
q0 的每个采样点都保留零存在掩码的反事实反馈，不设每块四个点的上限。

模块化信道新增无参数的 ST 前缀训练入口：前向仍执行实际硬档位，反向通过前缀符号门控和渐进层门控传递局部档位梯度。
低档点的假想高层修正只用于局部梯度，不作为真实接收信息、额外载荷或接收端输入。此梯度是有偏局部代理，并非逐点升降档的精确有限差分。
q0 的假想点不加入实时语义上下文；通过额外零掩码渲染获得保留梯度。

沿用 β=0.01；归一化分母是“全 q3 payload + 实测 XYZ + 档位表代理”的每源点开销，并在日志中明示。
训练中 payload 的单次概率期望可导；压缩 XYZ 的逐点梯度使用实测每保留点平均代价代理，压缩档位表不可精确求导。
部署评价重新测量保留点 XYZ 和完整 metadata，不能把概率期望冒充实际部署成本。
可靠数字侧流的净 2 bit/use 是沿用的显式假设，不是模拟符号的文件大小，也不等于实际 FEC 实现。

所有编解码模块 joint LR=1e-5，keep LR=.01，tier LR=.001，mask Adam eps=1e-15，无梯度裁剪。
每四个 joint 更新额外锚定一个固定前缀 q1→q2→q3，独立轮换，使用同一训练相机。
编解码器梯度为 learned-map 与 uniform-anchor 图像梯度的等权平均；分配表的图像/速率梯度不变。仍只执行一次优化器更新。
joint 不重新添加属性预训练或颜色专训；不添加 LPIPS/SSIM 损失，也不重新训练一轮 latent-only 信道。

## 输出与可视化

- `metrics/loss.jsonl`：采样 q0–q3 数量、图像/速率梯度、逐模块更新量、时间、前缀锚定。
- `metrics/validation.jsonl`：每250步、阶段边界，固定 q1/q2/q3 与学习后的十次部署；原 PLY 和真实照片的 PSNR/SSIM/LPIPS。
- `metrics/test.jsonl`：最终全体测试相机，同四种部署布局。`--test-views N` 才缩小测试集。
- `deployment/allocation_history.jsonl`：单次概率期望、十次真实部署计数与变化。
- `deployment/initial/tiers.npy`、`deployment/final/tiers.npy`：**原始输入 PLY 行序**；模型内部 Morton 排序不会写回错误索引。
- `images/initial`、`images/final`：相同代表验证视角的效果图；部署另外保存 `deployment_viewXXXX_tiers.png` 和像素标签 `*_tier_labels.png`。
- `checkpoints/initial`、`end_allocation`、`end_joint`、`latest`、`final`：完整 codec、绑定模型/场景身份的 allocation 表和精确续训状态。

档位图不是把相机空间强行切成互斥点区域。使用恢复后的同一组椭球和同一相机，
把 q1/q2/q3 分别以 RGB 三通道 one-hot 光栅化一次，保留真实跨档遮挡和透明度，
得到各档对每像素累计 alpha/transmittance 贡献；取最大贡献档位着色。
q1 蓝、q2 橙、q3 紫。黑色代表无覆盖，不是 q0；删除点不在接收端画面中，数量单独标注。
它显示哪类档位主导了可见画面，不等价于物体语义分割或逐点重要性真值。
独立保存的逐点 tiers.npy 才是每个高斯的完整真实档位。

SSIM 使用显示域 RGB；LPIPS 使用冻结的预训练 VGG v0.1，仅评价。
请求 LPIPS 但缺权重会报错，不用零分替代。首次运行先缓存：

```bash
python -m gaussian_jscc.perceptual_metrics --device cpu
```

## 服务器后台启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/truck_channel_joint2000_gpu3_20261016_110009"
GPU=3  # 启动前确认可用；不是强制使用3号卡
OUT="$PWD/output/truck_modular_allocation_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_modular_allocation_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

先不写输出/不渲染，只核对 checkpoint/config 和数据路径：

```bash
CUDA_VISIBLE_DEVICES="$GPU" python -m gaussian_jscc.modular_allocation \
  --run "$RUN" --out "$OUT" --dry-run
```

中断后保持原 RUN/OUT 和全部设置不变：

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/test_modular_allocation_truck.sh \
  "$RUN" "$OUT" >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
```

精确续训使用 `latest/state.pt`，同时恢复 codec、分配表、两个 Adam、随机状态和阶段进度；
独立 `codec.pt` 只供推理/新实验初始化，不包含完整的分配训练续训信息。
本机 CPU 测试用可微替代光栅化验证逻辑，不代表真实 CUDA 画质或收敛；服务器首次运行会检查 MaskGaussian 零掩码前向一致性。
