# 独立对抗渲染训练实验设计

日期：2026-10-19。状态：用户已明确要求直接实现；运行入口与实际说明见 `docs/adversarial_training.md`，不再要求重复设计审批。

## 1. 目标与边界

在当前双编解码器实验上新增 MSE + LPIPS + 对抗损失实验，判断局部外观约束是否能改善模糊、颜色和伪影，并检查其是否损害场景忠实度。

- 不修改、不停止正在运行的 MSE+SSIM、MSE+LPIPS 两组。
- 使用独立启动脚本、独立输出目录；不自动启动服务器任务。
- 16bit 可靠 XYZ、语义/信道网络、8/24/48 个复符号预算保持不变。
- 不训练档位分配器；不引入新的发送内容，不增加判别器通信开销。
- 不承诺 PSNR、LPIPS 或观感必然改善；二维对抗约束不保证多视角一致性。

## 2. 方案选择与依据

推荐无条件局部图像判别器：输入一张 RGB 渲染图，输出局部真假 logits。训练相机跨步骤变化，各视角共用同一个判别器。

备选方案：

1. latent 条件判别器：更接近 NTSCC，但当前逐高斯 latent 与二维像素不直接对齐，需要额外投影和条件设计，本轮不采用。
2. 只在最终阶段进行 GAN 精修：更便宜，但无法与正在运行的两组比较从首次语义渲染训练起更换目标的影响，本轮不采用。

文献依据：[NTSCC，IV-A3，式(26)](https://arxiv.org/html/2112.10961)将 MSE、LPIPS 和条件对抗项结合。其判别器以 latent 为条件；本实验只是借鉴混合目标，并非复现其条件判别器或数值配置。

判别器不会比较一对图的逐像素差异，也不接收相机、原图配对条件或原始高斯属性。MSE、LPIPS 才负责同视角配对恢复；判别器负责局部外观分布。

## 3. 初始化、训练阶段与调度

沿用现有两组的配置来源、初始化校验、场景指纹、相机划分和随机种子：

- 配置来源：`output/truck_beta_parallel_gpu2_20261017_213438/beta0p005`；仅取配置和场景记录，不取后期权重或档位表。
- 公共属性初始化：`output/truck_clean_color_gpu1_20261014_103802/checkpoints/end_semantic_bootstrap/codec.pt`；复用已完成的 4000 步，不重新训练。
- 新建信道模块，避免混入历史信道训练结果。

| 阶段 | 新训练步数 | 可训练模块 | 目标 |
|---|---:|---|---|
| semantic_render | 1000 | 语义编码器、语义解码器；另更新判别器 | clean 路径混合图像目标 |
| channel_bootstrap | 2000 | 信道编码器、信道解码器 | 10 dB latent MSE；判别器完全不更新 |
| channel_render | 2000 | 信道编码器、信道解码器、语义解码器；另更新判别器 | 10 dB 混合图像目标 + latent MSE |

模块学习率保持 `1e-4`。语义编码器在后两个阶段冻结。信道阶段继续轮换 q1/q2/q3/mixed，与现有实验一致。

两个渲染阶段从第一步即使用混合目标，不另设隐含 warmup。判别器及其 Adam 状态跨阶段保留，中间暂停；生成端保持现有阶段边界重建优化器的规则。

## 4. 损失定义与默认值

设 `z(I)` 为判别器局部 logits，`S(a)=softplus(a)`；每张图先对局部 logits 求平均，再对本步相机等权平均。

生成端：

`L_image = MSE(I_hat, I_source) + lambda_p * LPIPS(I_hat, I_source) + lambda_a * mean(S(-z(I_hat)))`

判别端：

`L_D = 0.5 * [mean(S(-z(I_source))) + mean(S(z(stop_gradient(I_hat))))]`

- MSE：未裁剪 renderer RGB 的逐通道像素均值，与当前实验一致。
- LPIPS：冻结的预训练 VGG LPIPS，RGB 裁剪至 [0,1] 后映射到 [-1,1]；保留对输入图像的梯度。
- 判别器：相同的裁剪及 [-1,1] 映射用于真图、假图；直接使用 logits 与 softplus，避免显式 log(sigmoid)。
- `lambda_p=1`：沿用当前 MSE+LPIPS 组。
- `lambda_a=0.001`：公开可调的保守工程起点，不声称来自 NTSCC 的公开数值或是最优值。不能跨 RGB 尺度直接搬用其他论文的 GAN 系数。
- channel_render 保留当前 `1 * latent_MSE`，不把 GAN 加入信道初始化阶段。
- 对抗权重为 0 应可用于测试接线等价性，但默认独立实验为正权重。

## 5. 判别器结构

采用轻量 70×70 感受野的 PatchGAN 风格全卷积网络：

- 5 个 `4x4` 卷积，通道 `3 -> 64 -> 128 -> 256 -> 512 -> 1`。
- 步幅依次 `2,2,2,1,1`，padding=1；前四层 LeakyReLU(0.2)。
- 卷积使用 spectral normalization；不使用 BatchNorm、Dropout。
- 输出未 sigmoid 的局部 logits；输入为原分辨率 RGB，不额外随机裁块或缩放。
- 输入高宽至少 32；不满足时明确报错。
- 判别器 Adam：`lr=1e-4, betas=(0.5,0.999)`，均为可记录、可配置的工程参数。

不新增条件网络、多尺度判别器、感知特征匹配、R1 等额外机制。本轮先检验最小可解释的对抗增量。

## 6. 更新顺序与 replay 一致性

每个真实渲染训练步严格执行：

1. 生成当前 noisy/clean 高斯场景，沿用现有 replay 解码与 RNG 恢复机制。
2. 固定判别器参数，置 eval 模式，避免 spectral normalization 的 power-iteration buffers 在生成端梯度计算时变化。
3. 逐视角渲染，计算混合生成损失，累加对场景的梯度，再 replay 回传编解码器。生成端必须保留穿过判别器到假图的梯度，不能使用 no_grad 包住该计算。
4. 缓存每个视角裁剪并脱离梯度的假 RGB 到 CPU；不保留 renderer 或 LPIPS 计算图，也不重新渲染一套假图。真实图使用既有 reference 缓存。
5. 执行一次生成端优化器更新。
6. 开启判别器参数梯度、置 train 模式；逐视角将缓存真/假图送入判别器，按视角数归一化累计 D 梯度，执行一次 D 优化器更新。
7. 记录该步结束状态后才允许保存 checkpoint。

这是明确的 G 后 D 交替更新：D 使用本步生成端更新前的假图，而不是更新后重新生成的图。两者在一轮中各更新一次，没有额外更新比例。

判别器初始化应保存并恢复全局 RNG，避免其随机权重初始化改变现有相机/信道采样序列；判别器不使用随机裁块或 Dropout。缓存仅在当前步有效，不跨步或跨阶段复用。

若任一损失、梯度非有限，立即报错，禁止把本步标记为完成或保存为成功状态；从最近完整 checkpoint 恢复。错误时不宣称当前内存中的模型仍处于更新前状态。

## 7. 接入边界与兼容性

预期新增独立 `adversarial_objective.py`、`adversarial_training_controls.py` 和 `scripts/test_adversarial_training.sh`，复用现有模块化实验循环，不复制整个训练器。

现有模块化入口增加显式 opt-in 的 GAN 目标、判别器状态及训练钩子。默认 MSE、MSE+SSIM、MSE+LPIPS 的目标、命令和保存路径不变；没有 GAN 状态时不创建判别器。

checkpoint 必须在同一次原子保存中包含：

- 原有 codec、生成端 Adam、阶段进度、RNG、scene record；
- 判别器 state_dict（包含 spectral normalization buffers）；
- D Adam、D 已更新步数、完整对抗协议参数。

旧 checkpoint 不强制携带该字段；对抗 run 的 exact resume 缺失或不匹配对抗状态必须报错，不能悄悄重建 D。新开实验可以使用公共 bootstrap，但不能把普通中途 checkpoint 冒充对抗精确续训。

resume 参数必须校验权重、D 结构/优化参数和阶段预算；相同断点的恢复要能重现后续 CPU 小样本更新。

## 8. 输出与判断标准

单一实验文件夹保存日志、协议、checkpoint、指标、曲线与图片。保留初始和最终验证图片；中间只保存数值，最后额外输出测试集指标。

沿用当前 clean / finite_noiseless / 10 dB q1/q2/q3/mixed 的评估，记录对 source PLY 和真实照片的 PSNR、SSIM、LPIPS，以及颜色和 latent 指标。判别器不参与评估指标计算和部署推理。

新增逐步日志：MSE、LPIPS、原始及加权 GAN 项、GAN/其他图像项比例、D 真/假 logits 均值、D loss、D 梯度/参数更新范数、G/D 各自耗时与显存峰值；显示各项曲线，不能只比较总 loss。

D 的分类准确率或 loss 不是画质结论。优先检查：

1. 在相同相机、同档位下，LPIPS 与局部观感是否改善。
2. PSNR、SSIM、颜色是否出现不可接受退化。
3. 文字、边缘和纹理是否出现伪造、多视角不一致。
4. q1/q2/q3 是否同时受益，而非只优化某一档。

## 9. 验证范围

不在本机运行完整场景训练；本机 CPU 检查：

- logits 形状、有限损失、参数边界、生成端输入梯度可达。
- G 步不改变 D 参数和 spectral buffers；D 步不反传到 codec 或场景。
- 各视角梯度按相机数平均；replay 与直接图上的生成端场景梯度一致。
- 一次 G 更新和一次 D 更新的步数及缓存释放行为。
- 非有限错误、协议不匹配、D 状态缺失明确失败。
- D/Adam/RNG 随 checkpoint 原子保存并精确续训。
- 原有非 GAN 入口及旧 checkpoint 兼容性的针对性测试。
- shell 语法、CLI plan-only、启动/续训命令与独立输出保护。

真实 CUDA 渲染、峰值显存、实际训练效果由服务器实验确认；交付时不能把 CPU 接线检查描述为 CUDA 场景验证。

## 10. 下一步

按用户明确要求直接实现、验证、提交及推送；运行与续训方式以 `docs/adversarial_training.md` 为准。
