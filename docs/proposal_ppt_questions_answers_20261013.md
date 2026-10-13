# 3DGS 语义通信开题 PPT 补充内容与逐项回答

核对日期：2026 年 10 月 13 日。代码依据：`3a0772e`。用途：补充开题汇报内容，并为答辩提问准备解释口径。

本文逐项回应所提供的 17 项检查意见，并非对尚未提供的完整 PPT 逐页审阅。建议围绕一条主线组织汇报：

> 面向已有 3DGS 场景的无线交付，以接收端渲染质量为目标，研究高斯属性的渐进式联合信源信道编码，以及高斯级通信资源的差异化分配。

最值得补充的是“研究问题—研究内容—系统模型—技术路线—验证实验”的闭环，不需要把清单机械地扩成 17 张独立页面。

## 使用前先统一当前版本的边界

| 内容 | 当前实际状态 | 开题时的表述 |
|---|---|---|
| 可靠坐标交付＋属性 JSCC＋q0–q3 分配＋联合微调 | 已有多场景完整实现和运行记录 | 已完成系统原型与初步验证 |
| 语义编解码器与信道编解码器模块化 | 已有 Truck 实验和独立训练流程 | 已完成模块化实验，正在优化表示与通信的协同 |
| PointNet++ 式空间上下文 | 已完成四组实验代码及本地小块测试 | 正在验证的结构方案，尚不能宣称整场景画质提升 |
| 模块化＋新空间上下文＋完整分配器 | 尚未合为同一套已验证完整流程 | 后续集成目标，不应画成全部已验证 |
| 多 SNR 自适应 | 有条件输入、噪声扫描和相关接口；主要实验固定 10 dB 训练 | 已进行信道条件测试，多 SNR 联合训练属于后续工作 |
| 未见场景直接分配档位 | 当前主要是逐场景可学习概率表 | 共享编解码器可以迁移；新场景分配表仍需拟合 |
| 实际无线链路 | 当前是信道仿真，坐标和元数据假设可靠交付 | 不宣称已实现真实空口、纠错编码或重传协议 |

最新模块化实验不训练档位分配器。下面分别标明“完整分配流程”和“模块化实验”，避免把两个分支的能力拼接成一个已经完成的系统。

## 1. 研究空白应该怎样提炼

**判断：必须补充，但不要把“没人做什么”当作必须满足的句式。**

不能写“现有方法只压缩数据、不考虑质量”，也不能写“尚无人将 3DGS 引入语义通信”。已有 3DGS 渐进表示和流式交付研究，例如 LapisGS、PCGS；也已有 3DGS 语义通信研究，例如 HiGS-3DSC，其公开摘要涉及分层语义特征和异构链路保护。[LapisGS](https://arxiv.org/abs/2408.14823)、[PCGS](https://arxiv.org/abs/2503.08511)、[HiGS-3DSC](https://ieeexplore.ieee.org/abstract/document/11664491)。

### 建议放到 PPT 的文字

> 现有研究通过表示压缩、分层表示与按需交付，提高了 3DGS 场景的存储和传输效率。本研究进一步面向带宽受限、存在信道噪声的无线交付，关注一个具体问题：**如何根据不同高斯基元对最终画面的影响，联合决定“是否发送”和“发送多少”，并使所分配的通信资源转化为有效的渲染质量收益。**
>
> 为此，需要把高斯级资源分配、渐进属性恢复与接收端渲染反馈联系起来，而不仅分别优化场景体积或固定码率下的重建质量。

这里的空白是需要解决的具体机制问题，不是宣称整个领域无人涉及。相较于从少量输入图像推断新高斯的前馈重建，本项目的输入是**已经存在的、可渲染的 3DGS 场景**；发送端已知坐标和属性，研究重点是交付和恢复，而非重新估计场景几何。

建议本页下方提炼三个问题框：

1. 如何把有限符号分配到真正需要它们的高斯上？
2. 如何让增加的符号带来稳定、可测的属性恢复收益？
3. 如何在含噪传输下保持场景结构，并准确核算全部交付成本？

## 2. 研究目标如何表述

**判断：必须补充，适合一页。**

### 一句话总目标

> 构建以接收端渲染质量为导向的 3DGS 渐进式语义通信方法，通过属性编解码与高斯级资源分配的协同优化，在有限通信资源下提高场景恢复质量。

### 三个分解目标

1. **可恢复：**建立可靠几何辅助的属性通信路径，使含噪接收结果能够稳定还原可渲染场景。
2. **可伸缩：**学习可按前缀交付的属性表示，使同一套编解码器支持不同通信资源投入。
3. **可分配：**根据渲染反馈学习各高斯的保留和档位选择，在实际离散部署中取得可验证的率失真收益。

目标不宜写成“保证最优”“无损还原全部属性”或“任意信道下实时传输”。目前最明确的优化目标是源场景渲染 MSE；语义在这里指面向场景恢复任务的表示，并不是已经学习了“车辆比树木更重要”等类别语义。

## 3. 研究内容应该分成哪些部分

**判断：开题核心，建议总览一页，再展开三页。**

### 内容一：几何辅助的高斯属性表示与通信编码

将位置坐标与其余属性区别处理。坐标采用每轴 16bit 量化和差分压缩，提供稳定几何；不透明度、尺度、旋转和 SH 颜色属性通过学习式通信路径恢复。进一步拆分语义表示模块与信道适配模块，并利用可靠坐标建立局部空间上下文。

当前对应：`position_delivery.py`、`modular_core.py`、`modular_codec.py`、`pointnet_context.py`。

### 内容二：可变前缀与渐进恢复

一次产生完整符号序列，按照 q1/q2/q3 交付前 4/12/24 个复符号。研究基础前缀与增强部分的协同，避免只提高高档位而破坏低档位，以及增加资源但画质收益很小的问题。

当前已有前缀共享机制；显式属性残差累加、分层冻结属于独立消融，不是当前模块化基线的默认结构。

### 内容三：渲染驱动的高斯级资源分配

对每个高斯分别学习“保留概率”和“保留条件下的档位概率”，结合逐点渲染反馈与通信惩罚，优化实际离散档位分布。验证删点收益与正档位分配收益，避免把两者混为一谈。

当前对应：`allocation.py`、`mask_gradient.py`、`allocation_diagnostics.py`。

### 可作为第四项，但不必另立一条技术主线

**多场景迁移与系统评估：**共享编解码器，分别保存场景几何和分配表，研究跨场景恢复、场景适配开销、通信成本及效率。它也可以并入实验方案页，避免研究内容过散。

## 4. 拟解决的关键问题是什么

**判断：必须补充；不要只列“可导”“联合建模”这样的名词。**

| 关键问题 | 为什么难 | 当前切入点 |
|---|---|---|
| 离散资源决策如何获得有效反馈 | q0 会关闭点，前缀长度是离散值；整图损失难以直接解释每个点的升降档收益 | 保留/档位分层概率，硬采样前向、连续替代梯度反向，关闭点的反事实渲染反馈 |
| 增强符号如何转化为有效画质 | 多档位共享网络存在优化折衷；更多符号并不自动意味着网络会利用它们 | 前缀训练、固定前缀保护、层间协同消融；模块化信道恢复的 latent 约束 |
| 大规模不规则高斯集合如何高效编码 | 点分布不规则，属性具有空间相关性，全场景注意力成本高；过强混合也可能损害逐点属性 | 局部分块、逐点自身路径、可靠 XYZ 辅助、多尺度空间聚合 |

需要特别说明：当前逐点反馈是**局部替代梯度**，不是穷举每个高斯所有档位后精确测得的边际率失真收益。“重要性—信道联合建模”可以作为后续方向，但主要结果仍在固定 10 dB 下获得，不能把任意 SNR 下的联合自适应写成已解决问题。

## 5. 系统模型与问题建模怎么写

**判断：必须补充。主讲页只留场景、信道和目标三个公式，其余可放备份页。**

### 5.1 输入、接收结果与档位

输入是固定源场景：

\[
\mathcal G=\{(\mathbf x_i,\mathbf a_i)\}_{i=1}^{N},
\quad \mathbf a_i=(\alpha_i,\mathbf s_i,\mathbf r_i,\mathbf c_i).
\]

其中 \(\mathbf x_i\) 是中心坐标，\(\alpha_i\) 是不透明度，\(\mathbf s_i\) 是尺度，\(\mathbf r_i\) 是旋转，\(\mathbf c_i\) 是 SH 颜色系数。代码内部使用适合学习的参数化和标准化，不是直接对这些物理量统一做裸 MSE。

\[
q_i\in\{0,1,2,3\},\qquad k(q_i)\in\{0,4,12,24\}.
\]

q0 不发送该高斯的属性载荷和坐标，但它在档位描述中仍需要被表达。因此“q0 的属性载荷为零”不等于整个系统没有它的任何控制开销。

当前概率表采用两层决策。令 \(p_i\) 为保留概率，\(\rho_{ij}\) 为保留后选择正档位的概率：

\[
\pi_{i0}=1-p_i,\qquad \pi_{ij}=p_i\rho_{ij},\quad j=1,2,3.
\]

它们是逐场景、逐高斯学习的参数，不是一个已经能对任意新点直接输出重要性的通用策略网络。训练时从概率中硬采样，反向使用连续替代梯度；部署时独立抽取十次，全部为 q0 才删除，否则选择出现次数最多的正档位，平票再按概率和固定随机种子处理。因此单点部署删除概率为 \(\pi_{i0}^{10}\)，单次采样概率不能直接当作最终部署占比。

### 5.2 可靠坐标与属性链路

坐标在场景包围盒内按轴量化为 16bit 整数，随后使用 `delta_zlib` 无损压缩这些整数。接收端恢复的是量化坐标 \(\widetilde{\mathbf x}_i\)。

**准确说法是“16bit 量化坐标的无损压缩与可靠交付”，不是“原始浮点坐标无损交付”。**坐标和控制信息在当前仿真中假设可靠，但其字节数计入成本。

模块化实验的属性链路为：

\[
\mathbf y=E_s(\mathcal G),\quad
\mathbf z=E_c(\mathbf y,\gamma_c),\quad
\widehat{\mathbf y}=D_c(\widetilde{\mathbf z},\mathbf q,\gamma_c),\quad
\widehat{\mathbf a}=D_s(\widehat{\mathbf y},\widetilde{\mathbf x}).
\]

这里 \(\mathbf y\) 是语义潜变量，当前默认每点 48 维；\(\mathbf z\) 是可发送的复符号，不应把二者都叫“码流”。信道编码器产生完整 24 符号序列，档位选择决定发送的前缀，而不是语义编码器分别生成三套不同表示。

### 5.3 信道模型

主要实验采用复 AWGN：

\[
\widetilde{\mathbf z}_i=\mathbf z_{i,1:k(q_i)}+\mathbf n_i,
\quad \mathbf n_i\sim\mathcal{CN}(0,\sigma^2\mathbf I),
\quad \sigma^2=10^{-\gamma/10}.
\]

\(\gamma\) 为以单位复符号功率为基准的标称 SNR，主要训练设为 10 dB。代码按复符号实、虚部各分配 \(\sigma^2/2\) 的噪声方差；分段功率归一化带有稳定项，平均符号功率不超过单位预算，不宜描述成每个样本都精确等于 1。

物理噪声 SNR 与网络条件输入 \(\gamma_c\) 是两个概念。最新模块化代码会把条件输入限制到已训练范围；只用 10 dB 训练时，测试 20 dB 的实际噪声仍按 20 dB 加入，但条件输入仍为 10 dB。这不是已经完成多 SNR 自适应训练。

### 5.4 最终渲染失真

设 \(\mathcal R\) 为 3DGS 渲染器，\(v\) 为相机视角：

\[
D_{\rm render}=\mathbb E_{v,\mathbf n,\mathbf q}
\left[\operatorname{MSE}\left(
\mathcal R(\widehat{\mathcal G};v),
\mathcal R(\mathcal G;v)\right)\right].
\]

这里的教师是输入 PLY 的渲染图，**不是 12bit 坐标版本，也不是所有阶段都直接监督真实照片**。当前剪枝实验中，\(\mathcal G\) 就是作为通信输入的剪枝后场景。真实照片另作评价参考，用于反映从实拍到最终接收画面的总误差。

### 5.5 通信成本与优化目标

部署属性载荷使用次数：

\[
C_{\rm payload}=\sum_{i=1}^{N} k(q_i).
\]

若可靠数字侧信息的净效率为 \(\eta\) bit/复信道使用，则成本写成：

\[
C_{\rm total}=C_{\rm payload}
+\left\lceil\frac{8B_{\rm XYZ}}{\eta}\right\rceil
+\left\lceil\frac{8B_{\rm meta}}{\eta}\right\rceil.
\]

\(B_{\rm XYZ}\) 和 \(B_{\rm meta}\) 单位为字节；后者包括档位表和包头等实际元数据。共享模型默认预先可用；首次交付模型的成本应单独列出。

当前完整分配流程采用软惩罚：

\[
\min_{\theta,\phi}\quad
D_{\rm render}+\beta\,\widetilde R(\phi),
\]

其中 \(\theta\) 为编解码器参数，\(\phi\) 为场景分配表参数。分配阶段固定 \(\theta\)，联合阶段再更新二者。\(\widetilde R\) 是归一化通信成本代理，不是直接对压缩器求导。

以单次采样训练为例，令 \(\pi_{ij}=P(q_i=j)\)、\(c_x\) 为实测每保留点坐标成本、\(c_t\) 为实测档位表每源点成本，可概括为：

\[
\widetilde R=
\frac{N^{-1}\sum_i\left[\sum_{j=1}^{3}\pi_{ij}k_j+(1-\pi_{i0})c_x\right]+c_t}
{C_{\rm full,q3}/N}.
\]

此式对应当前训练代理的含义：坐标单价与档位表实测值在本次更新中作为常量。**档位表的压缩字节数被记录，但没有其精确可导模型；归一化分母是全 q3 的载荷＋坐标＋档位表代理，不只是 24。**完整元数据包头在最终评测时核算。

如果 PPT 想先写理想问题 \(\min D\;\text{s.t.}\;C\leq B\)，可以，但必须说明当前实现用软惩罚探索率失真折衷，**没有保证每次部署严格满足指定硬预算的投影算法**。

当前常用 \(\eta=2\) 是数字侧链路效率假设，不是“一个 JSCC 复符号等于 2bit”的物理定律。复符号保存为两个 float32 得到的是磁盘存储量，不是空口带宽。[DeepJSCC](https://arxiv.org/abs/1809.01733)、[NTSCC](https://arxiv.org/abs/2112.10961)提供了以信道使用次数比较传输资源的相关基础。

## 6. 技术路线与总体框架怎样讲

**判断：必须补充，建议一张交付框架图＋一页训练步骤。**

### 交付框架图的文字规划

```text
输入 3DGS 场景
   ├─ 中心坐标 → 16bit 量化 → 差分压缩 → 可靠侧信息 → 恢复坐标 ───────┐
   ├─ 属性及局部几何 → 属性表示/JSCC 编码 → 完整符号序列               │
   │                                       ↓                         │
   └─ 场景分配表 → q0/q1/q2/q3 → 选择 0/4/12/24 符号前缀             │
                                           ↓                         │
                                        含噪信道                      │
                                           ↓                         │
                           属性解码 ← 档位与控制信息、恢复坐标         │
                                           ↓                         │
                                  接收端 Gaussian 场景 ←─────────────┘
                                           ↓
                                       多视角渲染
```

说明：图中“属性表示/JSCC”在完整分配基线中是一体化网络；模块化实验把它拆为语义编解码器与信道编解码器。画最终研究框架时可展示四个模块，但应将“接回分配器”标为待集成。无需把数据下载或离线剪枝画作核心通信模块，输入注明“给定 3DGS 场景”即可。

### 已跑通的完整分配流程

| 阶段 | 更新什么 | 监督/目标 | 已完成多场景实验的步数 |
|---|---|---|---|
| 属性初始化 | 编解码器 | 单高斯局部响应 MSE | 每场景 5000 |
| 场景渲染训练 | 编解码器，轮换固定档位与 mixed | 源 PLY 渲染 RGB MSE | 每场景 5000 |
| 档位分配训练 | 冻结编解码器，只更新场景概率表 | 渲染 MSE＋通信惩罚 | 每场景 1000 |
| 联合微调 | 编解码器与概率表 | 同上，可加固定前缀锚定 | 每场景 1000 |

“局部响应 MSE”不是旧版 XYZ、旋转、尺度等损失手工加权相加。代码把源高斯和恢复高斯放到相同中心，比较不同方向、局部采样位置以及黑白背景下的 RGB 响应，联合约束形状、透明度和颜色。它不包含整场景遮挡、透视和像素低通，因此只是便宜的初始化目标，最终仍需全场景渲染监督。

当前认可的分配研究设置是单次采样学习、十次规则部署；旧 `multiscene_pruned_aligned` 结果实际采用十次部署式训练。两者都在代码中保留用于对照，不能把旧结果重新标注为新设置。固定前缀锚定也是配置项，不是所有入口都默认启用。

### 最新模块化编解码器实验的流程

| 阶段 | 步数 | 可训练模块 | 目标 |
|---|---:|---|---|
| 语义属性初始化 | 2000 | 语义编码器＋语义解码器 | 局部响应 MSE |
| 无信道语义渲染 | 1000 | 语义编码器＋语义解码器 | 源 PLY 渲染 MSE |
| 信道初始化 | 2000 | 信道编码器＋信道解码器 | \(L_y=\operatorname{MSE}(\widehat y,\operatorname{stopgrad}(y))\) |
| 通信渲染优化 | 2000 | 信道编码器＋信道解码器＋语义解码器 | \(D_{\rm render}+\lambda_y L_y\) |

后两个阶段语义编码器固定，使信道模块有稳定的恢复目标。当前 \(\lambda_y=1\) 为可调工程设置，不是理论最优值。此流程共 7000 步，只验证编解码器，不包含学习分配器。

## 7. 是否需要单独讲两条方案

**判断：取决于“两条方案”指什么，不能为了凑结构创造第二项课题。**

仓库早期流程图中的“路线一/路线二”是重要性先验驱动和联合学习驱动的两种资源策略构想，不等同于现在的一体化/模块化编解码器。旧图中的预算投影、策略网络和部分损失，也不都是当前实现。

### 建议汇报的对比

| 维度 | 方案 A：先验重要性分配 | 方案 B：渲染驱动分配 |
|---|---|---|
| 决策依据 | 原场景存在概率等已有重要性排序 | 本任务中的画质损失与通信代价 |
| 分配方法 | 按排序给定不同资源 | 学习每个点的保留概率与条件档位概率 |
| 优点 | 简单、便于解释和复现 | 能适应具体编解码器的恢复能力和代价 |
| 局限 | 存在必要性不等于升档收益 | 需要场景级优化，替代梯度有偏 |
| 当前定位 | 诊断基线；有真实先验时才可用 | 已实现的主研究路线 |

当前三场景剪枝输入的记录显示 PLY 没有历史 mask logits，所以没有实际使用那些存在概率初始化分配表。没有先验时默认保留概率为 0.99，三个正档位均匀初始化；**不能宣称这轮结果由原版 MaskGaussian 的重要性分数直接引导**，也不能用不透明度冒充存在概率。

模块化与一体化的取舍可以在方法页另用一个小框说明：一体化是已跑通完整流程的基线，模块化用于分清表示误差和信道恢复误差，便于替换空间骨干及研究信道适配。建议一条主路线深入，另一条作为必要对照。

## 8. 题目中的渐进式具体落在哪里

**判断：必须解释，适合一个前缀示意图。**

| 档位 | 总复符号数/高斯 | 在前一档之上追加 | 对应意义 |
|---|---:|---:|---|
| q0 | 0 | — | 不交付该高斯 |
| q1 | 4 | 4 | 基础属性恢复 |
| q2 | 12 | 8 | 使用更多信息改善恢复 |
| q3 | 24 | 12 | 更完整的属性描述 |

> 一次编码产生可复用的嵌套前缀；资源不足时仅交付基础前缀，资源允许时再交付增强部分。

要区分三个层次：

1. **代码已经具备：**固定保留集合时，q1/q2/q3 复用相同编码序列的不同前缀，不是三套独立码字。
2. **仍须实验保证效果：**前缀更长不意味着 PSNR 在每个视角、每次噪声下必然更高。共享解码网络和优化折衷可能造成退化，必须报告增强层增益。
3. **当前未实现完整网络服务：**全场景基础层优先发包、网络到包驱动增量渲染、重传和时延调度。可以作为部署设计展望，不能仅凭前缀机制宣称已完成实时渐进流式系统。

此外，改变 q0 保留集合可能改变上下文，不能把“固定保留集合下正档位可截取”扩展为“任意删点后旧码字都完全不变”。当前模块化默认是直接属性解码；q2/q3 显式累加属性修正属于另一条已做过的结构消融，不要与“符号前缀嵌套”混用。[DeepJSCC-l](https://arxiv.org/abs/2009.12480)可作为分层交付的依据，但它不能替代本项目的增益验证。

## 9. 创新点怎样写才可辩护

**判断：必须补充，建议标题用“拟创新点”或“预期方法贡献”。**

1. **面向渲染任务的高斯级通信分配：**将是否交付和属性恢复精细度统一为离散资源决策，通过逐点渲染反馈与通信代价联合学习，实现内容贡献与通信投入的匹配。
2. **可靠几何辅助的渐进属性通信：**利用单独可靠交付的坐标支撑属性恢复，以共享符号前缀提供多档位交付，并研究基础层与增强层的协同优化。
3. **可选、待进一步验证的贡献：**利用接收端可靠空间信息构建多尺度属性上下文，在语义—信道模块化框架下改善有限符号的表达效率。

第一项的验证重点是：同样总成本下，学到的分配优于固定档位、随机分配或先验排序。第二项要证明可复用前缀的收益和代价，而不只展示 q3 比 q1 好。第三项需要完整 GPU 实验、计算开销比较与消融，不能因为加入 PointNet++ 就称为创新。

不建议独立列为创新的内容：使用 Transformer、使用 PointNet++、使用 Gumbel-Softmax、16bit 量化、zlib 压缩、MSE 监督、直接引用 MaskGaussian 的 masked-rasterization。这些是借鉴或工程组成，不是本项目首次提出的方法。

## 10. 可行性分析怎么写

**判断：有价值，适合与前期基础合并一页。**

- **数据基础：**已有 Tanks and Temples 的 Train/Truck、Deep Blending 的 Playroom/DrJohnson，以及相机和图像；前三个场景已进行共享编解码器训练，DrJohnson 配置为留出场景。[Tanks and Temples](https://www.tanksandtemples.org/)、[Deep Blending](https://repo-sam.inria.fr/fungraph/deep-blending/)。
- **代码基础：**依托公开 3DGS 渲染与 MaskGaussian 概率掩码方法，已实现通信仿真、实际包成本统计、分配对照和训练日志；使用与传播时仍需遵守各项目许可证。[3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)、[MaskGaussian](https://arxiv.org/abs/2412.20522)。
- **算力基础：**服务器有 4 张 24GB RTX 4090，可分卡运行对照；它是共享资源，不写成四卡始终独占。全场景反向采用 replay 降低显存，小块 CPU 测试用于检查实现，不能代替 CUDA 渲染验证。
- **方法基础：**DeepJSCC/NTSCC 提供面向重建质量的学习式通信思路，分层 JSCC 提供前缀传输参考，MaskGaussian 提供概率存在与渲染反馈参考。方法有可追溯来源，但这些文献并不直接保证本项目组合后的性能。

可直接使用的总结：

> 已具备数据、可微渲染、信道仿真和多场景训练基础，后续工作以受控消融、方法协同和公平通信评估为主，具备持续迭代条件。

## 11. 实验方案应该如何设计

**判断：必须补充，建议一页总览＋一页关键消融。**

### 11.1 数据与划分

| 场景 | 数据来源 | 当前剪枝输入高斯数 | 研究用途 |
|---|---|---:|---|
| Train | Tanks and Temples | 397,430 | 共享训练、较快的初步消融 |
| Truck | Tanks and Temples | 711,691 | 共享训练、结构与颜色详细消融 |
| Playroom | Deep Blending | 562,510 | 室内场景共享训练与测试 |
| DrJohnson | Deep Blending | 按实际输入统计 | 留出场景测试与分配表适配 |

前三个数字来自已完成实验的 `scene.json`，不是某个数据集固定不变的点数。该实验的相机测试集分别有 38、32、29 个视角；需延续保存的训练/验证/测试划分，不能选择看起来最好的视角报告总体结果。

“已见场景的新视角”与“未见场景”是两种泛化，不可互相替代。DrJohnson 上应分别报告直接使用共享编解码器，以及只拟合分配表后的结果，并记录适配步数和时间。三个训练场景支持初步验证，不能证明大规模通用泛化。

### 11.2 必要基线

1. **固定 q1/q2/q3：**检查基础编解码能力和每一级的资源—质量折衷。
2. **学得的档位分配：**检查实际部署 q0–q3 占比、总成本及画质。
3. **保持档位数量的随机打乱：**只打乱保留点 q1–q3，隔离正档位分配收益；全部打乱则同时改变保留位置，两者分开报告。
4. **先验排序：**仅在存在可对齐的真实 mask 先验时使用；当前剪枝 PLY 缺失该字段的实验不能虚构此基线。
5. **传统数字交付：**使用同一源场景的压缩表示，明确源编码、调制和纠错配置后与 JSCC 比较。当前主要完成的是净效率假设下的资源换算，不能写成已完成 FCGS/HAC＋LDPC 的真实链路实验。

随机打乱保持相同档位数量和属性载荷，不一定保持相同坐标压缩成本，因此仍需测量总成本。传统基线也应使用相同的剪枝输入；若比较未压缩 PLY，只能作为存储参照，不能把离线剪枝和已有压缩方法的收益全部归到本方法名下。

### 11.3 优先消融与每项回答的问题

| 消融 | 回答的问题 | 控制条件 |
|---|---|---|
| 单次/十次式训练，统一十次部署 | q0 冷启动学习与部署规则的作用是否被混淆 | 同一冻结 codec、初始化、相机序列与训练预算 |
| 部署抽样 10/5/3 次 | 部署规则对删点和画质的影响 | 同一最终概率表；注意也会改变正档位投票结果 |
| 多个 β | 是否存在更好的率失真工作点 | 先冻结 codec；新一轮应在最终选择的采样方式下扫描 |
| 正前缀 4/12/24 与增强层机制 | 追加符号是否真正有效 | 同样噪声、参考图、训练预算与坐标成本 |
| 语义—信道一体化/模块化 | 模块拆分是否改善诊断和恢复 | 不同时改骨干、训练长度和目标后直接归因 |
| 空间上下文 baseline/encoder/decoder/both | 上下文在哪一端更有帮助 | 匹配数据、种子和流程，同时报告参数与耗时 |
| 无 XYZ/有 XYZ 解码辅助 | 可靠坐标是否减少属性恢复歧义 | 同样坐标交付开销，不额外偷偷提供教师属性 |
| 固定前缀锚定开/关 | 联合分配优化是否伤害均匀前缀恢复 | 同一分配初始化，分别监测 q1/q2/q3 |

过去的分配控制组中，β 对照是在 `deployment` 训练方式下进行，不能直接等价为已完成 `single` 模式的全套 β 扫描。

### 11.4 指标与展示

- **画质：**对 Source PLY 和真实照片分别计算 PSNR、SSIM；LPIPS 可作为补充，但不能声称每个现有脚本都已输出。另展示颜色误差和局部细节，避免只有平均分。
- **资源：**属性复符号总数、每源高斯平均符号数、坐标/元数据实际字节数、总复信道使用次数、首次模型交付成本。总成本曲线标明数字侧信息的 \(\eta\) 假设。
- **策略：**实际 q0–q3 数量与占比、期望与部署成本差、只打乱正档位的质量变化。不要只看概率期望，也不要跨 β 排名总 loss。
- **效率：**训练时间、编码/解码时间、显存、空间聚合开销。共享 GPU 上的速度要注明并发情况。
- **可靠性：**固定相机集，重复信道噪声和部署种子；正式关键结果增加独立训练种子，报告均值与波动。现有单训练种子结果不能当作统计显著性结论。

推荐最终图表：①画质—总信道使用曲线；②画质—SNR 曲线；③真实档位占比；④同视角局部细节与误差图；⑤关键消融表。PSNR 应统一使用现有逐视角计算再平均的口径，不能与“对平均 MSE 再取对数”混用。

## 12. 预期成果怎么写

**判断：需要，但不要承诺与当前工作无关的成果。**

1. 一套可复现的 3DGS 渐进语义通信原型，包括训练、部署、资源统计和评测工具。
2. 一套围绕渐进属性恢复与高斯级资源分配的方法，以及多场景对照、消融和失效分析。
3. 完成学位论文，形成可投稿的研究论文与实验材料。

不建议默认写“新建数据集”“申请若干专利”“实现实际 5G 系统”。目前主要使用公开数据，尚无自采数据集或真实空口系统依据；论文录用也不能列为已经可保证的结果。

## 13. 进度安排怎么写

**判断：需要，具体时间必须按真实答辩日期填写。**

以下是可调整的工作顺序，不是已经确认的毕业日程：

| 阶段 | 主要工作 | 完成标志 |
|---|---|---|
| 开题前，已有基础 | 系统原型、多场景流程、模块化实验、关键问题定位 | 代码与结果可追溯 |
| 后续第 1–4 周 | 确定 q0 学习方式，完成空间上下文和增强层受控对照 | 选定主方法，不再无控制叠加结构 |
| 第 5–8 周 | 将有效模块接回分配流程，完成多场景和留出场景评测 | 主结果与适配成本完整 |
| 第 9–12 周 | 补齐数字基线、成本敏感性和多种子实验 | 公平的率失真与效率比较 |
| 第 13–16 周 | 整理消融、图表、方法论证和论文初稿 | 论文初稿及复现实验包 |
| 实际答辩前预留阶段 | 修改论文、核验引用、补实验与答辩演练 | 定稿及正式答辩材料 |

若学校要求覆盖到具体年月，将上述工作包映射到真实开题、中期、送审和答辩日期。不要为了填时间轴自行编造这些日期。

## 14. 参考文献需要多少，应该引用哪些

**判断：需要。20–40 篇适合作为开题报告的参考文献规模，不等于 PPT 正文必须逐篇展示。**

下面给出 25 项可追溯文献入口。正文优先讲与方法最相关的 8–10 项，其余放备份页或书面报告。条目采用简写，正式提交时从论文/出版页面导出作者、卷期页码并统一为学校格式；预印本明确标注，不补猜会议归属。

### 场景表示、压缩和渐进交付

1. Kerbl 等. **3D Gaussian Splatting for Real-Time Radiance Field Rendering**. ACM TOG / SIGGRAPH, 2023. [作者项目页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)。基础表示与可微渲染。
2. Liu 等. **MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks**. CVPR, 2025. [论文](https://arxiv.org/abs/2412.20522)。概率存在和关闭点反馈的主要来源。
3. Fan 等. **LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS**. NeurIPS, 2024. [会议页面](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fd881d3b625437354d4421818f81058f-Abstract-Conference.html)。重要性剪枝与表示压缩。
4. Lee 等. **Compact 3D Gaussian Representation for Radiance Field**. CVPR, 2024. [论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Compact_3D_Gaussian_Representation_for_Radiance_Field_CVPR_2024_paper.pdf)。紧凑高斯表示。
5. Chen 等. **HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression**. 2024. [论文](https://arxiv.org/abs/2403.14530)。压缩中的空间上下文建模。
6. Chen 等. **Fast Feedforward 3D Gaussian Splatting Compression**. ICLR, 2025. [论文](https://arxiv.org/abs/2410.08017)。前馈压缩与跨场景编码参考。
7. Lu 等. **Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering**. CVPR, 2024. [会议页面](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Scaffold-GS_Structured_3D_Gaussians_for_View-Adaptive_Rendering_CVPR_2024_paper.html)。结构化表示参考，不是本项目已采用的锚点结构。
8. Shi 等. **LapisGS: Layered Progressive 3D Gaussian Splatting for Adaptive Streaming**. 3DV, 2025. [论文](https://arxiv.org/abs/2408.14823)。渐进表示与自适应流式交付。
9. Chen 等. **PCGS: Progressive Compression of 3D Gaussian Splatting**. arXiv, 2025. [论文](https://arxiv.org/abs/2503.08511)。数量与属性精度的渐进压缩，属于必须区分的相关工作。

### 学习式通信与直接相关的三维通信

10. Bourtsoulatze、Kurka、Gündüz. **Deep Joint Source-Channel Coding for Wireless Image Transmission**. IEEE TCCN, 2019. [论文](https://arxiv.org/abs/1809.01733)。复信道符号和端到端失真优化。
11. Kurka、Gündüz. **Bandwidth-Agile Image Transmission with Deep Joint Source-Channel Coding**. arXiv 2020，修订 2021. [论文](https://arxiv.org/abs/2009.12480)。DeepJSCC-l，分层渐进传输。
12. Dai 等. **Nonlinear Transform Source-Channel Coding for Semantic Communications**. IEEE JSAC, 2022. [论文](https://arxiv.org/abs/2112.10961)。非线性表示、可变率与信道编码。
13. Wang 等. **Improved Nonlinear Transform Source-Channel Coding to Catalyze Semantic Communications**. arXiv, 2023. [论文](https://arxiv.org/abs/2303.14637)。上下文、率和信道条件适配。
14. Yang 等. **SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding**. arXiv 2023，修订 2024. [论文](https://arxiv.org/abs/2308.09361)。注意力骨干与多率、多信道适配。
15. Xie 等. **Semantic Communication for Efficient Point Cloud Transmission**. IEEE GLOBECOM, 2024. [论文](https://arxiv.org/abs/2409.03319)。局部/全局语义表示与信道模块分工；点集重建不同于可渲染 Gaussian 属性交付。
16. **HiGS-3DSC: A Generalizable 3D Gaussian Splatting-based Semantic Communication System for Sparse-View 3D Reconstruction**. IEEE TCSVT，Early Access, 2026. [出版摘要与 DOI 页面](https://ieeexplore.ieee.org/abstract/document/11664491)。公开摘要已表明存在 3DGS 分层语义通信研究；正式创新对比仍应核对全文，不能只据摘要断言本方法独有。

### 空间上下文与前馈高斯恢复

17. Qi 等. **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**. CVPR, 2017. [论文](https://arxiv.org/abs/1612.00593)。逐点映射与对称聚合。
18. Qi 等. **PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space**. NeurIPS, 2017. [论文](https://arxiv.org/abs/1706.02413)。度量空间中的分层局部聚合。
19. Wu 等. **Point Transformer V3: Simpler Faster Stronger**. CVPR, 2024. [会议页面](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html)。大规模点特征处理的效率参考，不作为已使用骨干宣称。
20. Charatan 等. **pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction**. CVPR, 2024. [会议页面](https://openaccess.thecvf.com/content/CVPR2024/html/Charatan_pixelSplat_3D_Gaussian_Splats_from_Image_Pairs_for_Scalable_Generalizable_CVPR_2024_paper.html)。前馈高斯恢复参考，不是通信性能基线。
21. Chen 等. **MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images**. 2024. [论文](https://arxiv.org/abs/2403.14627)。几何辅助前馈恢复参考；输入任务与本项目不同。

### 数据与评价

22. Knapitsch 等. **Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction**. ACM TOG, 2017. [数据与引用](https://www.tanksandtemples.org/)。Train/Truck 数据来源。
23. Hedman 等. **Deep Blending for Free-Viewpoint Image-Based Rendering**. 2018. [作者项目页](https://repo-sam.inria.fr/fungraph/deep-blending/)。Playroom/DrJohnson 数据来源。
24. Wang 等. **Image Quality Assessment: From Error Visibility to Structural Similarity**. IEEE TIP, 2004. [作者提供的论文](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)。SSIM 指标来源。
25. Zhang 等. **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric**. CVPR, 2018. [论文](https://arxiv.org/abs/1801.03924)。LPIPS 指标来源，作为补充评价准备。

这些文献支持方法动机与相关工作定位，不代表它们全部已经在本仓库复现。特别是 NTSCC 的超先验熵模型、PointNet++ 原论文完整网络，以及上述压缩器，都不能仅凭借鉴思路就写成“已实现”。

## 15. 前期工作基础有哪些能明确写出来

**判断：很有价值，建议展示可核验的结果而不是“已初步复现”。**

### 已完成的工作

1. 在公开 3DGS/MaskGaussian 基础上搭建属性 JSCC、可靠坐标交付、资源统计和渲染评估流程。
2. 完成 Train、Truck、Playroom 的共享编解码器与逐场景分配实验，每场景 5000＋5000＋1000＋1000 次更新，共 36,000 次场景更新。
3. 完成前缀、概率部署、β、颜色、XYZ 解码辅助、层间协同等控制实验代码和若干实验记录。
4. 完成显式 latent 重建约束的模块化 Truck 实验，共 7000 步，提供表示误差和通信误差的分阶段诊断。
5. 完成 PointNet++ 式编码端、解码端和双端升级的实验入口及本地小块检查，完整渲染收益仍待验证。

### 一组适合展示的已完成多场景结果

以下是 `multiscene_pruned_aligned_gpu0_20261010_194134` 最终固定档位测试，10 dB AWGN、每视角 3 次噪声试验，对源 PLY 的 PSNR：

| 场景 | q1，4 符号 | q2，12 符号 | q3，24 符号 |
|---|---:|---:|---:|
| Train | 20.86 dB | 22.56 dB | 23.28 dB |
| Truck | 23.55 dB | 24.98 dB | 25.44 dB |
| Playroom | 26.35 dB | 27.63 dB | 28.09 dB |

这张表支持“已经跑通多场景、多前缀恢复，增加资源有质量收益”，**不单独证明学习分配优于相同成本的其他分配**。它也反映 q2→q3 的边际收益较小，正是后续研究的依据。

模块化 Truck 实验 `truck_modular_latent_gpu2_20261013_113847` 的 32 个测试视角、2 次试验中，10 dB q1/q2/q3 对源 PLY 的 PSNR 分别约为 22.31/23.73/24.08 dB。它与上表训练范围、流程和模型不同，**不能直接用两张表判断模块化优于或劣于一体化**。

最新 PointNet++ 式本地探针仅为 4 个拟合块＋4 个留出块、50 步局部响应训练，无 CUDA 渲染：baseline 留出损失 0.02213，编码端升级 0.02217，解码端升级 0.01764，双端升级 0.01887。只能作为“值得开展解码端上下文实验”的初步依据，不能换算成 PSNR 提升或跨场景泛化。

### 归属必须清楚

3DGS 表示、可微光栅化、MaskGaussian 概率存在与 masked-rasterization 是已有研究基础。本项目的工作是围绕场景通信扩展属性编码、前缀预算、多档位反馈、训练流程和评估体系，而不是重新发明这些基础方法。

## 16. 风险与应对应该写什么

**判断：建议保留，可并入进度或实验页。**

| 风险 | 目前可见的依据 | 应对与备选 |
|---|---|---|
| 增强层投入大、收益小 | 固定档位及模块化结果中高档位增益趋缓 | 优先验证表示/解码上下文和层间协同；若收益仍小，保留更简洁的档位，不强行堆层 |
| 分配收益由删点主导，或期望成本与部署成本不一致 | 已出现 q0 学习困难及十次规则放大保留的情况 | 单次学习/十次部署对照，真实压缩成本核验，独立打乱正档位；以实际率失真结果选设置 |
| 坐标/控制成本与场景适配抵消属性节省 | 低档位下可靠坐标占比可能较高，分配表逐场景学习 | 单列全部侧信息与适配成本；用固定档位作为保底，后续再研究更高效坐标压缩或通用分配策略 |

模块化分支的集成风险也应在内部计划中预留：不能把新骨干、信道条件、分配训练和部署规则同时替换后再将收益归因到某一个模块。

## 17. 总结与致谢页怎样收尾

**判断：建议保留一页，不需要增加新技术内容。**

### 可直接使用的总结

> 本课题面向受限无线资源下的 3DGS 场景交付，以接收端渲染质量为目标，研究渐进属性通信与高斯级资源分配。
>
> 前期已完成多场景系统原型和模块化编解码实验。后续重点验证增强层的有效收益、分配策略的实际率失真优势，以及多场景迁移能力。
>
> 请各位老师批评指正。

## 建议的 PPT 页面组织

在现有背景页之后，按以下顺序安排；不是要求把本文全部搬上页面。

1. 研究问题与现有工作的关系。
2. 总目标与三项研究内容。
3. 系统模型与交付框架。
4. 渐进属性编解码：前缀与可靠几何。
5. 渲染驱动分配：保留、档位、通信惩罚。
6. 关键问题与模块化/空间上下文的改进计划。
7. 训练路线：完整基线与实验分支的区别。
8. 实验方案与公平对比口径。
9. 前期基础：多场景结果和代表性图像。
10. 拟创新点、风险与后续验证。
11. 进度与预期成果。
12. 总结与致谢。

参考文献、详细公式、两条资源策略的对比、训练参数放备份页。每页优先说明“为什么做—怎么做—用什么证据判断有效”，不要把代码配置列表当研究内容。

## 代码与实验依据索引

| 本文内容 | 可追溯位置 |
|---|---|
| 坐标交付与成本 | [position_delivery.py](../gaussian_jscc/position_delivery.py)、[transport.py](../gaussian_jscc/transport.py) |
| 信道与复符号 | [codec.py](../gaussian_jscc/codec.py) 中 `channel`、`prefix_mask`、`pack` |
| 保留/档位概率与十次部署 | [allocation.py](../gaussian_jscc/allocation.py) 中 `GaussianTierMask`、`deployment_tiers` |
| q0 反馈与成本代理 | [mask_gradient.py](../gaussian_jscc/mask_gradient.py) 中 `local_mask_step` |
| 实测成本、存在先验来源 | [allocation_diagnostics.py](../gaussian_jscc/allocation_diagnostics.py) 中 `AllocationCostMeter`、`load_existence_prior` |
| 多场景阶段与固定前缀保护 | [multiscene.py](../gaussian_jscc/multiscene.py) 中 `train`、`preserve_prefix` |
| 局部响应初始化目标 | [local_response.py](../gaussian_jscc/local_response.py) 中 `local_response_loss` |
| 模块划分、维度与前缀 | [modular_config.py](../gaussian_jscc/modular_config.py)、[modular_core.py](../gaussian_jscc/modular_core.py) |
| 模块化可训练范围与损失 | [modular_training.py](../gaussian_jscc/modular_training.py) 中 `PHASE_MODULES`、`latent_reconstruction_loss`、`render_step` |
| 模块化 7000 步与 SNR 设置 | [modular_experiment.py](../gaussian_jscc/modular_experiment.py) |
| 空间上下文实验 | [pointnet_context.py](../gaussian_jscc/pointnet_context.py)、[pointnet_context_controls.py](../gaussian_jscc/pointnet_context_controls.py) |
| 三场景训练完成和固定档位结果 | `output/multiscene_pruned_aligned_gpu0_20261010_194134/complete.json`、`training.json`、`scenes/*/scene.json`、`evaluation/final/results.csv` |
| 模块化 Truck 完成和测试结果 | `output/truck_modular_latent_gpu2_20261013_113847/complete.json`、`analysis/summary.json` |
| 本地空间上下文探针 | `output/pointnet_context_cpu_probe/summary.json` |

输出目录为本地证据，不保证随 Git 仓库分发。引用时应带实验配置，不只截取某个 PSNR 数字。
