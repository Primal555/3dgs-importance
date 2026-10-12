# 独立实验：Gaussian 属性表示与信道编解码分离

日期：2026-10-12

状态：待用户审核的架构设计；尚未实现训练代码。用户已同意另建模块化实验版本，本文件明确实现边界和检验标准。审核本文件后再编写实施计划。

## 1. 目标与不变项

目标是分开验证两个问题：不经过通信链路时，属性表示能否恢复场景；相同表示经过有限符号预算和信道后，会损失多少质量。拆分本身不作为性能提升的证据。

首轮只使用 Truck；保留现有主流程及其 checkpoint、模型标识和包格式兼容性。实验默认随机初始化，不加载此前实验权重；已有运行目录只提供路径、统计配置和相机划分。其他实验版本不叠加进本实验。

保持以下设计：

- 每轴 16bit 量化 XYZ，量化整数采用 delta-zlib 无损压缩；不宣称对原浮点 XYZ 无损。
- XYZ 可靠交付假设、坐标和档位元数据成本继续明确记录。
- Morton 排序、256 点块、64 块训练批次。
- 编码特征宽度 96、2 个空间网格上下文块、网格分辨率 4/8。
- 解码特征宽度 96、2 个局部注意力块、32 点窗口、4 个注意力头、交替半窗口错移。
- 通信预算为 q1/q2/q3 = 4/12/24 个复符号；完整码字编码后按前缀交付。
- 现有局部响应属性目标和原 PLY 渲染 RGB MSE；照片仅作独立评价参考。
- 全场景反向传播采用现有 replay，不引入整场景巨大激活图。
- 默认学习率固定 1e-4、默认不裁剪梯度；记录非有限梯度并在更新前报错。

首轮不学习删点、不训练档位分配器、不加入通信惩罚。q0 仅用于 padding 和显式验证用的关闭点，不能把这种支持表述为已实现可学习分配。

## 2. 四模块接口

输入为标准化 Gaussian 特征 f、发送端 XYZ、有效点掩码；接收端还具有可靠交付的量化 XYZ。SH degree=3 时 f 包含 3 维 XYZ 和 56 维属性。

完整通信路径：

```text
f + sender XYZ + valid
  -> SemanticEncoder -> y
  -> ChannelEncoder(y, SNR) -> full codeword
  -> incremental power normalization -> prefix selection(q)
  -> channel simulation -> received symbols + prefix mask
  -> ChannelDecoder(received, mask, q, SNR) -> y_hat
  -> SemanticDecoder(y_hat, active, sequence offset) -> attributes
  -> fixed delivered XYZ + attributes -> scene renderer
```

### 2.1 SemanticEncoder

保留五组逐点线性嵌入、融合 MLP、多尺度 GridContext 和残差路径。移除对 SNR 和正档位的依赖；上下文门控改由可学习的、与信道无关的常量条件驱动。使用显式 valid/active 掩码处理 padding。

输出逐点实值 y，默认 `latent_dim=48`，可配置。48 是首轮工程设置，与 q3 的 48 个实数信道槽位匹配，便于控制表示规模，并非论文给出的最优维度。内部 96 维特征经线性读出成为 y；y 不执行发送功率归一化，不附带向接收端传递的逐点尺度因子。

y 不等于通信符号，不是压缩比或比特率指标，也不作为额外可靠侧信息发送。干净表示能够恢复良好，不等于它一定能通过 24 个复符号恢复良好。

### 2.2 ChannelEncoder

采用轻量逐点 MLP：`latent_dim -> 96 -> 48`，隐藏层 GELU。SNR 通过独立的小型条件网络调制隐藏特征。首轮不在这个模块加入跨点注意力，以免同时改变表示骨干和通信骨干。

在固定保留集合及 SNR 下生成一次完整码字；编码器不读取正档位 q1/q2/q3，正档位只决定交付前缀。4、8、12 个复符号的增量段分别归一化，保留原有正的 power floor，不能因增加后缀改变前缀。

### 2.3 ChannelDecoder

输入为补零后的 48 个实数接收槽位、48 个有效槽位标记、档位和接收端 SNR 条件。采用轻量逐点 MLP：`96 -> 96 -> latent_dim`，条件注入隐藏层，最后使用无界线性读出。

必须明确区分符号值为零和符号未收到。恢复 y_hat 不经过最终 LayerNorm 或单位功率归一化，以免强制抹去表示的幅值信息。

### 2.4 SemanticDecoder

`latent_dim -> 96` 线性输入映射、块内序号编码、原有 2 层局部窗口注意力、独立属性输出头。

不输入原始属性、未交付表示 y、SNR 或正档位嵌入。active 掩码只是控制哪些点存在；通信状态由 ChannelDecoder 整理为 y_hat。

首轮不加入 XYZ 几何注意力实验，保持主基线的语义解码骨干。XYZ 只在生成场景时使用可靠交付值，不预测、不微调。代码应避免创建可训练但无实际贡献的 XYZ 输出头。

## 3. 干净表示路径与训练阶段

干净表示路径：`SemanticEncoder -> y -> SemanticDecoder`，完全绕过两个信道网络、前缀裁切、发送功率归一化和噪声。它不是 `channel=none`：后者仍需经过完整有限预算信道编解码路径。

默认四个子阶段，均为真实 optimizer update 数：

| 阶段 | 默认步数 | 目标 | 可更新模块 |
| --- | ---: | --- | --- |
| semantic_bootstrap | 2000 | 现有局部响应属性目标 | SemanticEncoder、SemanticDecoder |
| semantic_render | 1000 | 原 PLY 渲染 RGB MSE | SemanticEncoder、SemanticDecoder |
| channel_bootstrap | 1000 | 相同局部响应属性目标，经过有限预算 AWGN | ChannelEncoder、ChannelDecoder |
| channel_render | 1000 | 原 PLY 渲染 RGB MSE，经过有限预算 AWGN | ChannelEncoder、ChannelDecoder、SemanticDecoder |

所有步数和模块学习率可调；默认 1e-4，无隐含学习率下降。阶段切换使用新 optimizer，记录状态重置。semantic 阶段只使用有效点，保留全部真实点；channel 阶段循环 q1、q2、q3、mixed，同样不随机删点。

channel_bootstrap 固定两个语义模块，但冻结 SemanticDecoder 不能用 `no_grad()` 断开 y_hat 的梯度。可以对冻结的 SemanticEncoder 使用 `no_grad()`。进入 channel_render 后仅开放语义解码器，语义编码器仍固定。

不将 y_hat 的普通 MSE 设为额外加权主目标；只记录为诊断，避免又引入未经验证的特征权重。两个通信阶段仍从真实属性或渲染目标获得梯度。

多 SNR 接口一并设计：默认训练 10 dB；可显式提供训练 SNR 列表，每个真实更新均匀抽取一次、该更新的所有块和 replay 使用同一个值。首轮不默认启动多 SNR 对照。固定 SNR 与多 SNR 可从同一个 semantic_render checkpoint 分叉，并用相同通信阶段预算对比。

发送端条件 SNR 暂按已知估计值模拟，与真实 AWGN SNR 区分记录。首版默认二者相等，不能据此宣称已验证估计误差鲁棒性。可靠 XYZ 链路没有随 SNR 改变的实际 FEC 仿真，成本换算假设必须保持可见。

## 4. 验证与归因

每 500 个真实更新验证一次，阶段起止额外验证。固定相机、固定 mixed 档位图及配对噪声；验证不改变后续训练 RNG，也不更新任何参数或统计。周期验证默认只用 4 个固定验证视角，测 clean、有限预算无噪声和 10 dB AWGN；完整 SNR 扫描留到最终评价，避免验证开销反过来主导训练时间。

同时记录：

1. clean representation：干净表示路径的属性误差、对原 PLY/照片的 PSNR、SSIM；属于诊断，不声称有 4/12/24 符号通信成本。
2. q1/q2/q3/mixed、无噪声：仍走完整有限符号预算路径，测量映射、前缀和功率约束造成的损失。
3. q1/q2/q3/mixed、AWGN：最终评价默认扫描 SNR 0/5/10/15/20 dB，噪声重复至少 2 次，给出均值和离散程度；周期评价只使用 10 dB。
4. y/ y_hat 的均值、标准差、诊断 MSE，以及各属性误差；不把不同 latent 尺度的 MSE 当跨模型排名依据。
5. 四个模块各自的梯度范数、参数更新范数、可训练参数数和冻结参数检查；同时记录阶段时间和 GPU 峰值显存。
6. payload 复符号数量、坐标实际压缩字节数、档位和必要元数据成本；计算机 tensor 文件大小不是通信码率。

画图分别呈现各阶段目标、clean/finite-noiseless/AWGN 恢复指标、各前缀 PSNR-SNR 曲线，不把不同阶段的总 loss 直接拼为一个连续目标。

干净表示强但有限预算无噪声弱，表明信道映射、维度预算或功率约束值得检查；有限预算无噪声强而 AWGN 弱，才主要指向抗噪路径。channel_render 会更新 SemanticDecoder，所以必须重测同一时间点的 clean 路径，不能把阶段 A 的历史上界作为阶段 B 的当前上界。

semantic 阶段的两个信道模块尚未训练，其有限预算结果必须标记 `channel_trained=false`，不能将随机信道模块的表现当作学习瓶颈证据。性能对照默认提供可选的当前 direct 基线，从头运行 3000 步属性启动和 2000 步渲染训练，保持同一 Truck、预算、统计、相机划分、不删点和总计 5000 次更新；模块化分阶段方案同样总计 5000 次更新。此控制不声称参数量、梯度路径或计算量严格相同，也不把与历史更长训练结果的差异直接归因为模块化。

## 5. 输出、保存与继续训练

所有产物位于一个新输出根目录，子目录为 checkpoints、metrics、charts、images。第一轮所有图像均使用相同的 2 个代表相机；只保存实验初始和最终的 clean、q1/q2/q3/mixed（10 dB）图像，阶段中间保留指标而非图像。最终所有测试相机上评价，不用两张保存图替代全测试集指标。

每 500 步保存 latest；每个阶段结束保存阶段 checkpoint。checkpoint 必须包含独立架构配置、latent 维度、四个模块、属性统计、训练阶段、实际更新计数、当前 optimizer、相关 RNG 和明确的数据/相机配置。

明确区分精确 resume 和从 semantic checkpoint 开始新的通信实验：resume 恢复 optimizer/RNG/阶段计数；阶段初始化只载入相应权重，新建 optimizer。checkpoint 不得被旧 architecture 静默加载。

共享模型出现在包元数据中时必须走显式模型工厂分发；接收端只依赖 checkpoint、收到的符号、量化 XYZ、档位和必要元数据。不能依赖 source PLY 或训练目录的干净表示。模型权重交付是否计入成本仍沿用现有共享模型假设，并在记录中注明。

## 6. 实现范围与隔离

计划新增独立模块化核心、实验训练入口、Truck 启动脚本和专用测试。复用现有场景输入、局部响应、渲染、replay、评价及坐标交付代码。

只在模型工厂、checkpoint/packet 加载和参数观测中增加 opt-in 分支；不把旧 learned_joint 改成模块化，不改变旧 checkpoint 的配置序列化或 model_id，不覆盖旧训练脚本。必要的 shared helper 调整必须以回归测试证明不改变基线行为。

不在同一实验中叠加 PointNet++、残差增强解码、XYZ 几何注意力、可学习分配或模型专属新损失。现有分配器以后可接入，但概率表必须在新编解码器下重新拟合，不能默认沿用旧最优分配。

## 7. 验收标准与风险

实施时须覆盖以下自动检查：

- y 在固定输入/valid 下对 SNR、正档位不敏感；receiver 不访问 y 或源属性。
- 固定保留集合/SNR 下，前缀数值及增量功率不因正档位改变。
- clean 路径完全绕过信道模块；finite-noiseless 路径实际执行信道模块。
- padding、q0、空注意力窗口及最后一个不满块处理正确。
- 坐标始终等于可靠交付的量化值；XYZ 没有训练更新。
- 每阶段冻结参数不变；冻结的语义解码器仍向 ChannelDecoder 传梯度。
- replay 与直接反向的小样本损失/梯度一致，冻结/解冻后仍一致。
- checkpoint 往返、阶段初始化与 exact resume 语义正确，旧版本加载及 model_id 不受影响。
- RNG 配对、多 SNR 抽样复现、receiver-only 包解码和元数据成本准确。

本机 CPU 可进行接口、反向、冻结、resume、传输及小型拟合测试；CUDA 光栅化和真实 Truck 性能须在服务器验证，不宣称 CPU 单元测试证实了最终画质。

性能提升不是代码验收条件。模块化可能引入额外参数与计算，冻结语义编码器也可能限制最优通信表示。48 维干净表示可能仍不够，或对低前缀过难；这必须用 clean 与有限预算无噪声的差距判断，不能预先认定是噪声问题。与旧模型比较时报告参数量和真实训练成本，不宣称容量或 FLOP 严格匹配。

下一步：用户审核此设计，确认后编写并审核实施计划，再实现实验代码。
