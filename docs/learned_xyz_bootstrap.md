# 无逐点坐标侧流的第一阶段测试

入口：`scripts/test_learned_xyz_bootstrap.sh`。它与已验证的量化坐标两阶段入口分开：
强制随机权重、`position-delivery=learned`、`spatial-response`，只执行5000步bootstrap；
不接收相机、不调用场景渲染、不执行render/joint，不读取INIT。
旧的 `test_local_response.sh` 和损失不变。

## 信息流与通信口径

发送端仍可使用原始XYZ做Morton排序、局部网格和特征编码。
接收端仍是既有 `learned_joint` decoder：只接收有噪声符号、档位、SNR和固定块内序号，
XYZ与其他属性都由网络输出。没有位置seed、手工位置波形、预测中心替换或逐点坐标侧流。
每Gaussian预算仍为0/8/16/32复符号；位置和属性共同使用这份预算。
全局bbox、属性统计/模型、档位和顺序协议仍沿用现有假设，不能称为“零辅助信息”。
teacher位置只用于训练目标，不进入decoder。全局bbox对坐标误差做等比例的对角线归一化。

## v2：位置与尺寸解耦，形状恢复不再被全局模糊掩盖

旧v1的预测位置与预测协方差共同进入响应重叠项。错误中心可以通过放大椭球补偿，
而全局平滑又弱化了真实小椭球的尺寸约束。现有权重曾出现严重膨胀与近乎纯色的渲染。
v2替换训练目标，不修改接收端、不引入坐标侧流，也不裁小输出椭球来伪造质量。

`L = (L_position + L_native_shape + L_centered_RGB) / 3`。

- **位置**：在随机正交方向投影中心差，使用同宽、固定各向同性核。
  `L_position = mean[2*(1-exp(-||delta||^2/(4*sigma^2)))]`。
  delta按bbox对角线归一化；sigma仍为`0.5, 0.125, 0.03125, 0.0078125`。
  预测尺寸、旋转、透明度和颜色完全不进入这个项，因此不能用放大或变透明降低位置惩罚。
- **形状**：中心对齐后，比较未加全局平滑的投影协方差A、B。
  `L_native_shape = mean[-log(2*(det(A)*det(B))^(1/4)/sqrt(det(A+B)))]`。
  用负对数而不是`1-K`，避免巨大椭球的惩罚与纠正梯度很快饱和。
  以teacher最大轴为共同数值单位，用float64执行2x2矩阵运算；不通过旋转参数MSE强迫等价四元数。
  独立于透明度，低alpha不能隐藏错误形状。沿用相对于最大轴的`1e-5`协方差数值下限。
- **外观**：复用已存在的中心对齐local-response计算，按teacher和detached预测自身的footprint采样，
  比较黑/白背景RGB响应，不使用bbox宽度。只在loss内部对齐中心，不替换真正接收或渲染的XYZ。

**三项等权、位置带宽、采样方向与相对数值下限仍是明确的实验设计，不是无超参数或已验证最优。**
三个标量相近不意味着梯度相近。宽尺度位置项只负责冷启动捕获，极远偏移仍可能饱和，
目前仍没有证明可达到逐像素位置精度。该代理不包含场景透视、遮挡及alpha合成。
不得只凭新loss下降或尺度正常就宣称初始化成功，仍需检查离线真实渲染。
旧v1的loss值不能直接与v2比较，日志中的objective已改为`spatial_response_v2`。

## v3：原始椭球尺度控制的细位置项

v2训练到20000步，参数损失继续改善但渲染没有改善，因此增加细位置监督。
宽尺度核、原生形状、中心对齐外观均保留。新的位置项为：

`L_position = L_coarse + w * L_fine`

`L_fine = mean[(sqrt(d^2 + r^2) - r) / h]`

- d：三维中心距离 / bbox对角线；不使用随机投影，因此每步都约束全部XYZ方向。
- r：原始Gaussian最大轴标准差 / bbox对角线（仅数值下限1e-12）。
- h：`min(spatial_bandwidths)`，默认0.0078125。它控制数值/梯度幅度，不是细位置的平滑宽度。
- w：`--spatial-fine-weight`，默认1；脚本可用`SPATIAL_FINE_WEIGHT`设置。0严格退回v2目标组合。

d较小时为平滑平方误差，偏差大于原始尺度后为近似线性惩罚，不会像Gaussian重叠核那样
在偏差较大时失去吸引梯度。r只控制转折尺度，**不直接以1/r放大梯度**。
相对bbox归一化位移的单点梯度范数不超过1/h；总loss中的贡献还乘w/3并平均。
预测尺寸、旋转、透明度均不能改变此项，teacher仅用于loss，接收端无新增输入或额外坐标传输。
采用等价有理式`d^2/(sqrt(d^2+r^2)+r)/h`避免小偏移的浮点抵消，float64计算后返回模型精度。

这不是“无权重/无超参数”：w和h仍是实验选择，也不意味着神经网络参数梯度被同样上界限制。
共享网络Jacobian仍可能放大梯度；不启用隐式梯度裁剪。它使用最大轴而非最小轴或真实相机
footprint，不对每个相对误差等权，仍不是像素损失，更不保证渲染已经恢复。
全流程保持bootstrap-only，没有暗中加入场景渲染训练或使用真实XYZ替换预测XYZ。

新增日志/验证CSV字段：`spatial_coarse_position_response`、`spatial_fine_position_response`、
`spatial_fine_weight`、`spatial_fine_contribution`（包含w/3的实际标量贡献）。
日志objective为`spatial_response_v3`，不要直接比较v2/v3总loss数值。
诊断脚本支持`--fine-weight 0/1`和`--lr`做相同起点、块采样、噪声与优化步数的对照。

### v3 本地验证边界

单元测试验证恒等点零loss/零梯度、极小teacher尺度下有限且有界的输出梯度、远处预测仍有
吸引梯度、teacher无梯度、预测尺度不影响细位置项，以及w=0复现v2组合。

真实Truck 20000步权重上，固定32个完整块（16训练、16验证）、相同噪声及120步优化：
- 新建Adam、LR2e-4：w=0/0.1/1均出现位置反弹，不作为支持新目标优越性的证据。
- 新建Adam、LR2e-5：w=0的q1/q2/q3验证XYZ RMSE约0.932/0.853/0.754；
  w=1约0.912/0.845/0.754。初始约0.957/0.836/0.764；q2仍比初始略差。
  这是单次固定噪声、小样本检查，改善很小，不能宣称优于v2或渲染已恢复。
- 从随机权重运行缩小网络120步，q1/q2/q3 XYZ NRMSE约
  0.10149/0.12711/0.15940 → 0.07568/0.07669/0.07422；与v2的短测试接近。
  新目标的梯度幅度比v2更大但有限，未启用裁剪。输出空间上界不等于模型参数梯度上界。

正式脚本仍保持随机初始化与2e-4，不默认加载旧权重；诊断中的小学习率不是偷偷修改正式参数。
所有检查均未使用CUDA渲染，本机不能提供新目标的PSNR提升证据。

## 服务器启动

当前默认使用v3，保留以上v2形状与外观约束，增加下述细位置项。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main || exit 1
OUT="$PWD/output/truck_learned_xyz_v3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/output"
CUDA_VISIBLE_DEVICES=2 PYTHON_BIN="$(command -v python)" \
BOOTSTRAP_STEPS=5000 BLOCKS_PER_BATCH=32 LR=2e-4 \
nohup bash scripts/test_learned_xyz_bootstrap.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

GPU需选择空闲卡。可设置CHANNEL=none对照，但默认仍是10 dB AWGN。
带宽可通过CLI `--spatial-bandwidths ...`显式修改，脚本保留默认便于复现。

## 观察什么

- `training.json`：损失定义、带宽、通信假设、固定验证块、是否与训练块重叠。
- `loss.jsonl`：总损失、各带宽位置响应、独立position/native_shape/appearance响应、XYZ世界单位RMSE、
  bbox对角线归一化RMSE、点距离中位数/P95，位置头梯度及参数更新。
- `bootstrap_validation.jsonl`：q1/q2/q3/mixed固定种子验证，每次默认16块、2次噪声。
  指标为各块/试验等权平均，RMSE不是全场景汇总；按点分位数也是块内分位数均值。
- `codec_best_bootstrap.pt`：按四布局平均空间响应损失选择，**不是最佳渲染模型**。
- `codec_end_bootstrap.pt` / `codec.pt`：最后一步；selection.json给出最佳步数。
- `charts/`：训练loss/梯度和位置验证曲线；没有场景恢复图是正常的，因为未加载相机。
- 尺度诊断同步写入训练日志、固定验证日志和`charts/bootstrap_position_validation.csv`：
  `max_axis_ratio_p05/p50/p95`、`max_axis_ratio_gt10_fraction`、`max_axis_ratio_lt0_1_fraction`、预测/teacher最大轴中位数、
  `xyz_distance_over_source_radius_p50`和`decoded_alpha_p50`。
  ratio是逐点最大轴标准差之比，1为尺寸相符；10倍统计仅为报警观测，不是裁剪阈值。
  console每10步打印三项loss和尺度ratio中位数。固定块验证仍是块内分位数的均值，不是全场景分位数。

判断时必须同时看held-out块的位置误差、局部外观响应和xyz_head梯度，不能用总loss
下降替代位置恢复证据，也不能由本测试宣称渲染质量、跨场景或多SNR能力已经解决。

如需查看每500步的真实场景恢复图和PSNR，可在训练结束后运行
[`evaluate_bootstrap_history.py`](bootstrap_render_history.md)补做只读渲染评估，
不需要重跑第一阶段，也不会进入第二阶段优化。

## 本机检查

CPU测试覆盖：恒等输入近零、teacher无梯度、位移吸引梯度、低alpha不屏蔽位置/形状约束、
固定中心误差时放大2/10/34/266倍不能改善测试样例的总损失、原生形状尺度不变性、
极端各向异性有限梯度、随机网络所有输出头反传与AWGN优化、无相机CLI及脚本参数隔离。

新增独立诊断程序，可加载已有问题权重，仅在内存中进行短修复检查，**不作为正式训练初始化**：

```bash
python diagnose_spatial_constraints.py \
  --checkpoint output/truck_learned_xyz_bootstrap_20260920_101516/codec_5000.pt \
  --ply output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply \
  --out output/spatial_constraints_v2_check --steps 60 --sample-blocks 16 --device cpu
```

固定完整块采样，奇偶块分为训练/验证；验证q1/q2/q3使用固定噪声。
报告原预测、替换正确尺度、替换正确XYZ、同时替换的代理损失与诊断，后者仅作teacher对照。
`--steps 0`只计算不优化；正步数在内存中用2e-4 Adam、10dB AWGN混合/均匀档位训练。
不保存或修改checkpoint，也不渲染；results.json记录输入SHA256及详细轨迹。
本机无CUDA：本地检查不能替代服务器渲染、显存与收敛验证。

### 2026-09-20 本地运行结果与限制

- 真实Truck旧5000步权重，16个分散完整块，半数训练/半数固定验证，CPU 60步修复：
  初始q3将尺度换回teacher后，v2损失从1.27727降至0.51377，位置项严格不变（0.047107）。
  与旧目标的“正确尺度反而更差”现象相反。
- 同一小实验q3验证最大轴ratio中位数25.50→0.199，XYZ RMSE 1.047→2.037。
  这说明纠正了膨胀，但出现了收缩过度与共享网络位置退化；不能称为恢复成功，
  也不建议用这个修复过的旧模型开始正式训练。程序只在内存中更新，源checkpoint未变。
- 另从随机权重在完整Truck数据池抽样训练120步（hidden32/depth1、block128、每步2块，
  4个固定验证块，10dB AWGN，LR2e-4、无裁剪）：q1/q2/q3 XYZ NRMSE从
  0.10149/0.12711/0.15940变为0.07561/0.07608/0.07429。
  全程未裁剪，梯度范数均有限，范围1.016–8.791；最终各布局尺度ratio中位数的块均值约0.20–0.22，
  仍偏小，后续需要同时观察收缩与膨胀，不能只检查超大椭球比例。
  这是有限梯度与冷启动通路检查，不是渲染质量、精度达标或长期稳定性证明。
