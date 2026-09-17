# v6：公共参考、位置目标与训练衔接

本次实现针对三个已确认的问题：公共参考主导位置误差、损失的外层块尺度与
实际几何子组不一致，以及几何目标在属性/渲染/mask 阶段被替换或减弱。
旧 v4/v5 模型保留原格式。新格式为 `--position-head reference_v6`。

## 1. 公共参考路径

原来公共参考与局部偏移共享含噪导频增益，增益误差会造成整组平移/伸缩。
现在分为两条路径：

- 公共参考：四个实数槽用于两个相位对，按接收端已知的保留点顺序，交错传输
  中心 xyz、log 半径及 1/4/16 三个相位频率。粗频率消除细频率的周期歧义。
- 局部偏移：其余几何槽包含局部偏移、冗余和局部导频。其幅度恢复不再决定
  公共中心或半径。

参考部分是恒能量相位信号，接收端不需要真实发送归一化因子。中心与半径仍在
原有 JSCC payload 内，不是附加粗坐标包头。几何/总复符号仍为 4/8/16 和
8/16/32，保持子组平均复符号功率为 1。参考与局部功率比例可学习，限制在
25%～75%，初始各一半；这没有引入额外平均功率，但不是峰值功率约束。

新增专门的 reference encoder/decoder。解码器输入包括参考估计、噪声精度
近似、档位分布、组长度和 SNR。它修正公共参考，不再只靠局部偏移网络补偿。

同一轮开发中发现固定绝对参考修正尺度会产生大梯度，因此最终使用
`reference_bounded=True`：参考修正幅度限制为估计精度的 0.5 倍，局部残差
限制为 `0.25*tanh(...)`。损失权重没有为压低梯度而缩小。
该精度是 AWGN 相位估计的近似尺度，不是严格的误差保证。

不足 24 个保留点的子组无法完整使用当前多频排布，双方按已知组长度回退至
v5。这是明确的短组限制，不能宣称稀疏极端档位布局已有相同性能。

`reference_v6_local_20260916` 是未做幅度预条件的中间开发结果；最终结果目录
是 `reference_v6_bounded_local_20260916`。旧中间 checkpoint 缺失 bounded 字段
时按旧数值路径加载，不会静默 reinterpret。

## 2. 同一个位置目标

`supervision_scale` 使用与收发完全一致的保留点压紧、分组和尾组规则。
不再拿 4096 点外层块半径替代 256 点子组半径；批次展平之前就计算监督尺度。

每点目标为：

```
r = max(当前几何子组半径, 0.001)       # bbox 归一化坐标
w = clamp(sqrt(r / Gaussian尺度), 1, 4)
Lpos = w * SmoothL1((xyz_hat - xyz) / r, beta=0.01)
       + SmoothL1((xyz_hat - xyz) * bbox_span / bbox_diagonal, beta=0.001)
Lgeometry = Lpos(noisy) + clean_weight * Lpos(noiseless)
```

Gaussian 尺度使用源椭球最大标准差除以 bbox 对角线，仅用于监督，不送给接收端。
它不是完整的协方差/可见性/投影敏感性建模。尺度下界与 [1,4] 权重上界控制
小椭球的梯度放大；这些常数是显式工程设置，不宣称最优。

clean_weight 默认 1，存入 v6 checkpoint，贯穿几何、属性、渲染和 mask 训练。
full-scene 目标对 v6 改为：

```
L = Lrender + geometry_weight * Lgeometry
    + attr_weight * Lother_attributes
    + beta * expected_rate             # mask 联合阶段
```

因此 attr_weight 不会再把几何约束一起缩小。v4/v5 旧训练权重逻辑不变。
日志中的 geometry / aux contribution 也同步更新。

## 3. 混合档位和 mask

- 几何短训练每步覆盖 q1/q2/q3 三种均匀档位，以及含 q0 的混合布局。
- 属性 all-tier 训练增加正档位混合布局；mask 联合阶段允许 q0。
- v6 支持 hard one-hot / straight-through Gumbel choices，硬前向与真实打包
  接收端对齐。保留了对普通连续 soft 布局的拒绝，避免误当作可发送格式。
- 梯度通过槽位门控、功率、属性 context、存在性与渲染/代理目标传回 mask。
- **保留点压紧、离散组成员和相位解缠分支使用 detached 的硬决定。**
  因而是有偏的条件直通估计，不是对重新分组的精确梯度。

属性网络不是冻结不动就自动适配；已有属性训练和联合训练路径现在可继续优化。
本机验证其可运行和梯度链路，未验证最终分配最优性。

## 4. 本机真实场景结果

CPU PyTorch，Truck 883,438 个 Gaussian，固定 AWGN 10 dB。300 步，Adam 1e-4，
属性权重冻结，四种几何布局等权，几何四分支分别裁剪至 1。
所有 RMSE 是未裁剪 XYZ、三个维度均方开根号，单位为场景坐标，不是米。

| 同一验证种子 100042 | q1 | q2 | q3 |
|---|---:|---:|---:|
| 输入 v5 checkpoint | 1.1927 | 0.9309 | 0.8215 |
| 新 v6 结构，残差零初始化 | 0.6290 | 0.3894 | 0.2515 |
| v6 bounded，300 步后 | 0.6278 | 0.3880 | 0.2506 |
| 另一验证种子 200042，300 步后 | 0.6267 | 0.3899 | 0.2505 |

主要收益来自参考路径重构，300 步附加收益很小，不能表述为长训练已经解决问题。
最终固定八块无噪声 RMSE 为 0.0114 / 0.0116 / 0.0113，仍非精确可逆。

| 裁剪前梯度，300 步 | 中位数 | 最大值 | 超过 1 的步数 |
|---|---:|---:|---:|
| 公共参考编码器/功率参数 | 0.290 | 2.932 | 107 |
| 公共参考解码器 | 0.212 | 2.953 | 74 |
| 局部编码器 | 0.432 | 1.526 | 16 |
| 局部解码器 | 0.414 | 1.582 | 12 |

未约束的开发版本公共编码器最大 113.3、公共解码器最大 51.6；预条件显著缓解
放大，但仍会裁剪，不能宣称消除了所有梯度风险，更不能外推到 CUDA 渲染。

另取真实场景两个外层块（8,192 点），执行 10 步属性 context 训练和 10 步
mask 联合反传。关闭 rate penalty 后，联合阶段 mask 梯度仍非零且有限。
使用可微属性代理损失，**不是实际渲染指标或部署质量实验**。

自动回归：88 项，85 通过，3 项 CUDA 测试跳过。包括独立接收端、混合/q0、
实际符号/功率、分组监督、参考分支梯度、三阶段 CLI checkpoint 衔接，以及
联合 mask 的 replay/checkpoint 梯度一致性。本机无 CUDA，真实渲染仍待验证。

## 5. 复现与结果位置

在仓库根目录使用 CPU PyTorch 环境，输出必须是新目录：

```powershell
python scripts/test_reference_v6_local.py --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply --checkpoint output/pilot_v5_clean_local_20260916/codec.pt --out output/reference_v6_recheck --steps 300
python scripts/test_reference_pipeline_local.py --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply --checkpoint output/reference_v6_recheck/codec.pt --out output/reference_pipeline_recheck --steps 10
python -m unittest discover -s tests -v
```

- `output/reference_v6_bounded_local_20260916/`：最终模型、训练/梯度记录、全场景两种子评估和来源哈希。
- `output/reference_v6_pipeline_local_20260916/results.json`：真实数据 context/mask 链路运行记录。
- `scripts/train_codec_fixed_geometry.sh` 已切换至 v6，仍是固定 10 dB 的几何训练，不自动启动长时间渲染训练。

后续风险：低 SNR 相位解缠可能跳周，当前只验证固定 10 dB；小组回退仍有 v5
限制；bbox/点数分组不是严格空间直径控制；位置误差相对小 Gaussian 仍偏大。
因此这次完成的是结构修正和可运行性验证，不是完整通信恢复或论文性能结论。
