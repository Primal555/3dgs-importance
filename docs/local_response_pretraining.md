# 局部响应预训练 → 全场景渲染训练

这是新增的训练目标，不是新的通信协议，也不是已被论文或实验验证的最优损失。
默认实验从随机权重开始，固定 10 dB AWGN，逐 Gaussian 的 0/8/16/32 复符号档位不变。
12 bit/轴 XYZ 仍由已有可靠数字侧流交付；其余属性通过 JSCC。
阶段一、二训练的是同一套编码器和解码器，不加载旧实验初始化。

## 阶段一：isolated local response

入口：`--bootstrap-objective local-response --bootstrap-steps 2000`。
只对非零档位和非 padding 行计损失，轮流使用 q1/q2/q3/混合档位。
编码、符号掩码、功率归一化和 AWGN 均走真实 codec 路径。

1. 将预测和源属性还原为物理属性；源属性仅作训练教师。
2. 每步采样 4 个球面均匀方向（可配置）；单个 Gaussian 正交投影：
   `Σ₂ = P R diag(exp(2s)) Rᵀ Pᵀ`，旋转和尺度通过同一个 footprint 起作用。
3. 源和接收 Gaussian 放在相同局部原点。此目标不计算 XYZ 误差，
   也不会训练 XYZ head，所以明确禁止搭配 learned XYZ 模式使用。
   量化位置对实际成像的影响仍由阶段二完整渲染体现。
4. 采用中心以及半径 0.5、1.5、3 标准差的三圈八方向探针。
   各从源 footprint 和 **detach 后的预测 footprint** 生成 25 个，合并为 50 个。
   两者在完全相同的探针位置比较，防止只观察源附近而遗漏过宽预测。
   探针生成不反传梯度，因此这是自适应采样下的局部代理目标，不是固定测度的精确面积积分。
5. `a(u)=sigmoid(opacity)*exp(-uᵀΣ₂⁻¹u/2)`；
   `c(d)=max(SH(d)+0.5,0)`，复用工程 SH 计算和 PLY 排列。
   比较 `F(u,d,b)=a(u)c(d)+(1-a(u))b`，分别取黑背景 b=0、白背景 b=1。
6. 黑、白背景 RGB MSE 等权平均；Gaussian、方向、探针等权。
   没有 XYZ/旋转/尺度/透明度/DC/SH 六项加权和。

黑白背景让“低透明度高颜色”和“高透明度低颜色”不再只凭相同 a*c 蒙混过关。
尺度单位在计算中消除；矩阵构造前提取最大 log-scale，使用 2×2 Cholesky solve，
归一化投影协方差加 `1e-5 I` 保证极薄 Gaussian 的数值稳定。
这个 floor 相对最大三维轴，不是物理像素滤波。

明确局限：方向分布、探针范围、等权平均、协方差 floor 都是设计选择，
不是“没有超参数”。这是孤立正交响应，不包含透视、邻居遮挡、透射率、
真实 rasterizer 的像素滤波/截断规则。低透明度或颜色截断仍可能产生弱梯度；
不能承诺彻底消除梯度爆炸或冷启动问题。

## 阶段二：原来的场景渲染目标

`--render-steps 300`，使用原始 PLY 渲染作为教师，完整场景多视角 RGB MSE。
不再加入局部响应或参数重建辅助项。阶段切换保留模型权重、重置 Adam moments；
两个阶段的学习率默认都固定为 2e-4，不自动下降，见下文。
300/2000 是便于迭代的预算，不是收敛保证。
默认不裁剪梯度。模型最终是否有效以固定验证视角/噪声的渲染表现判断。

## 分批重算反向传播与学习率

第二阶段默认 `--render-backward replay`：先无梯度编解码完整场景，计算渲染损失对
场景的梯度，再用相同信道随机数逐批重算编解码器并反传。只保留当前批次的 codec
激活；完整场景与渲染器仍占显存。第一阶段仍是普通局部批次前向/反向，不额外重算。
没有混合缓存或自动切换机制，损失、档位和信道配置不变。

`direct` 仍可显式选择，但 Truck 的883438点全场景运行已在第二阶段编码器前向 OOM。
减小 `BLOCKS_PER_BATCH` 不能消除 direct 的全场景激活总量。replay 可尝试64个block，
若峰值过高可降至32；旧版本约2秒/步不能作为新结构、多视角训练的速度保证。
`loss.jsonl` 会记录每步 `peak_allocated_mib` / `peak_reserved_mib`（CUDA运行时）。

第一阶段完成后若第二阶段 OOM，不必重跑预训练。使用 `scripts/train_codec_learned.sh`，
设置 `INITIALIZATION=checkpoint INIT=/path/to/codec_end_bootstrap.pt BOOTSTRAP_STEPS=0`，
并保持 `POSITION_DELIVERY=quantized POSITION_BITS=12`，显式设置 `RENDER_BACKWARD=replay`。
不要使用强制随机初始化的 `test_local_response.sh` 来续训。输出目录必须另建。
`--init` 加载模型及统计，不恢复Adam或随机数；正常 bootstrap→render 本来也清空Adam状态，
但重新启动并非逐步完全复现原运行。

学习率默认 `--lr 2e-4 --render-lr 2e-4 --lr-schedule constant`，
两个阶段均固定，不因验证波动自动降低。为避免旧shell变量影响本轮，启动时可显式设置
`LR=2e-4 RENDER_LR=2e-4 LR_SCHEDULE=constant`。

以下是保留的**显式可选** `--lr-schedule plateau` 行为，本轮默认不开启：

- 阶段初始值分别来自 `--lr` 和 `--render-lr`。
- 连续3次固定验证未超过0.5%的相对改善，乘0.5，最低1e-6。
- 阶段一监控四布局平均局部验证loss；阶段二监控原有固定视角/噪声的渲染验证score。
- 切换阶段重置调度状态与阶段初始LR，不比较不同目标的loss；不调节mask优化器LR。
- 每次实际降LR后重置早停计数，避免刚降速就停止；最佳模型仍按每次真实改善保存。
- 这些阈值是可配置工程默认值，不是理论最优值。

对应环境变量：`LR_SCHEDULE`、`LR_PATIENCE`、`LR_FACTOR`、`LR_THRESHOLD`、`MIN_LR`。
若显式开启调度，它只在验证时更新，`VALIDATE_EVERY` 决定两次更新之间训练多少步。
固定学习率模式仍记录验证及学习率日志，但不会降LR或因调度重置早停计数。

## 服务器小实验

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_local_response_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_local_response.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

GPU 2 只是示例，运行前确认空闲。可以通过 `PLY`、`SCENE` 指定路径；
`BOOTSTRAP_STEPS`、`RENDER_STEPS`、`LOCAL_RESPONSE_VIEWS`、`BLOCKS_PER_BATCH`
可以覆盖默认值。该脚本要求显式选择 GPU，强制随机初始化和 joint-steps=0。
不改变现有 render-only 对照脚本的阶段配置。

小实验仍使用整个场景，在阶段一抽局部块，阶段二抽 12 个训练视角中的 2 个。
不是截断场景为少量 Gaussian。每 100 步验证，验证会产生额外耗时。

## 观察内容

- `training.json`：目标、初始化、学习率、采样和通信配置。
- `loss.jsonl`：phase=bootstrap/render；局部黑/白 MSE、总梯度、分支梯度、更新量。
- `lr_schedule.jsonl`：各阶段基准指标、每次验证的学习率前后值、是否降低；`loss.jsonl.lr` 是该步实际使用值。
- `bootstrap_validation.jsonl`：固定随机种子下各档位局部响应误差；与渲染 MSE 不可直接比较。
- `validation.jsonl` 和 `validation_images/`：初始化、阶段一期间、阶段边界和阶段二的真实场景恢复。
- `codec_end_bootstrap.pt`：阶段一终点；`codec_best_render.pt`：阶段二按验证 MSE 选择；
  `codec.pt`：最终执行步，不保证最好。
- `charts/`：复用现有分阶段曲线；不要把两个阶段 loss 的跳变当作质量飞跃。

XYZ 侧流和源模型统计的交付假设、开销统计保持不变，不能把该方案和无坐标侧流方案
当作相同总码率直接比较。教师局部响应无需交付给接收端。

## 本地验证

```bash
python -m unittest discover -s tests -p test_local_response.py -v
python -m unittest discover -s tests -p test_training_performance.py -v
python -m unittest discover -s tests -p test_validation_lr.py -v
python -m unittest discover -s tests -v
```

测试涵盖同响应零误差、四元数符号等价、黑白背景消歧、各属性及 SH head 梯度、
极端长短轴数值稳定、真实 AWGN codec 短优化、随机初始化两阶段 CLI 及启动参数。
本地无 CUDA 时完整场景 renderer 以可微模拟函数验证训练接口；不等于真实图像质量验证。

本次额外以 Truck 的 883438 个 Gaussian 为数据池，用小网络（hidden=32、depth=1）、
每步两个 128 点块完成 CPU 100 步，无梯度裁剪、无 NaN/Inf 报错。
但仅取两块的局部验证误差未改善，因此这个检查只证明运行和反传可用，
不能证明方案已收敛或优于原始训练。服务器脚本使用完整默认网络和更长预算，
应同时查看阶段一局部验证与两阶段真实渲染图，不以训练 loss 单独判定成功。
