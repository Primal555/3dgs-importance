# Render-first Gaussian JSCC：目标、训练与短实验

## 优化什么

固定原场景 `G`，编码器、噪声信道、解码器输出 `G_hat`。主目标是：

```text
D = mean over sampled cameras, pixels, RGB channels:
      (Render(G_hat, camera) - Render(G, camera))²
```

使用未经输出截断的图像 MSE：超出显示范围的预测仍能获得梯度。
不要求 `G_hat` 的每个位置、四元数或 SH 必须逐项等于 `G`；只要各训练视角
和独立验证视角的场景渲染得以保留，就符合这个目标。相机只用于监督和验证，
不作为接收端输入。训练没有增加粗坐标、手工参考、重复位置观测或接收端原图。

这是明确选择的失真定义，而不是声称某篇论文证明它对 3DGS 最优。
MSE 对应 PSNR 的平方误差口径，利于首先验证端到端图像恢复是否成立；它并不
等价于感知最优。SSIM 作为独立观测指标，当前不再人为混入多个参数加权项。
图像语义通信的思想借鉴在于最终恢复图像的监督，**不是**把规则像素位置监督
直接搬成逐 Gaussian 的 XYZ 复制任务。

## 什么不变

- 全学习 `learned_joint` 编解码、逐 Gaussian 的四档预算、同块混合档位不变。
- XYZ 与属性仍共享原 JSCC payload；没有专门的位置保底通道或几何预算。
- 发送端空间上下文聚合、接收端局部符号特征注意力不变。
- 当前固定 AWGN 10 dB；并未在这次修改中恢复动态 SNR 训练。
- 包头的可靠接收假设、全局 bbox 元数据和收发模型预共享假设不变。
- 权重格式仍为 learned_joint v4。旧配置中的 `learned_v1` 和六项权重仅为
  格式/模型标识兼容保留，**新训练并不使用它们**；训练配置以
  `objective: render_mse_v1` 为准。旧损失工具可用于已有诊断，不是新主目标。

## 训练阶段

| 阶段 | 实际目标 | 数据与用途 |
|---|---|---|
| 可选 bootstrap | 规范化特征向量的统一 SmoothL1 | 局部块，提供可渲染初始化；不代表通信质量 |
| render | 多视角图像 MSE | 每步完整场景，采样多个训练视角；核心 codec 训练 |
| 可选 joint | 同一图像 MSE + beta × 期望符号数 / 最大档预算 | 联合四档 mask；短实验默认关闭 |

不再使用 `XYZ + .25 rotation + scale + opacity + DC + .25 SH` 的主目标，
不再添加逐点投影误差，也不再从低到高 ramp 渲染权重。
一旦进入 render，目标定义和权重就保持不变。不同阶段的曲线各画各的，不比较
bootstrap 数字和渲染 MSE 数字。bootstrap 仍是工程初始化，有标准化和维数偏好，
不是一个没有假设或经理论证明的损失。

脚本与 CLI 默认都从随机网络权重开始，bootstrap=0，直接使用渲染目标训练。
不依赖过去失败实验的检查点。随机初始化是网络的标准初始化，不是随意随机生成
输入场景；属性标准化统计量仍从待传 PLY 计算，全局 bbox 归一化约定不变。
仅显式设置 `BOOTSTRAP_STEPS` 为正数时才执行预热。这些步数是迭代预算，不是收敛保证。
`--steps` 仍作为 `--bootstrap-steps` 的别名，含义已更新。旧的辅助权重、投影
权重、render-ramp 参数不再接受，避免旧命令静默执行不同目标。

默认 bootstrap LR=1e-4，render/joint codec LR=1e-5，均为公开的工程超参数。
bootstrap→render 清空 Adam 历史，避免旧目标动量继续主导；render→joint
保留相同图像目标。`--init` 加载网络和统计量，不恢复旧 Adam/步数，不是精确续训。

图像损失不能保证从任意随机几何出发都能收敛，尤其是不可见点或极差初始化。
保留可选初始化是为了解决这个优化条件，不是继续以参数精确复制定义最终成功。

## 多视角与资源开销

每步均衡轮换 q1、q2、q3、逐点混合 q；默认不随机删除 Gaussian。
同一步先得到一个有噪恢复场景，再对多个视角平均图像 MSE，防止仅追逐单视角。
replay 模式逐视角累积 `dL/d(scene)`，释放各自渲染图，再对 codec 分批精确
链式回传；不是旧结果缓存，也没有抽稀场景。checkpoint 模式用于等价性核验，
其多视角图会同时保留，显存可能更高。

短实验保留完整 PLY，以12个分散训练视角、4个不重叠验证视角和300个渲染步骤
降低预算；每步2视角，32 blocks/batch。它不是“只训练小片区域”。默认每25步
在固定4个档位布局、2个噪声试验上验证，开销应计入总时间。验证视角从训练相机
列表中留出，不接触数据集正式 test split；验证好后再跑独立 benchmark。

梯度默认不裁剪。每步记录分支梯度、XYZ/属性场景梯度（replay）、实际 Adam
更新量和相对更新量；非有限 loss/gradient 会报错停止，而不是默默写 NaN。
大梯度依然可能出现，改成 MSE **不保证消除梯度爆炸**。

## 在服务器运行短实验

确认所选 GPU 空闲；下面的2只是可修改的物理卡编号。仅复制代码块内部。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=2
export INITIALIZATION=random
export BOOTSTRAP_STEPS=0
unset INIT STEPS
OUT="$PWD/output/truck_render_first_short_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/test_render_first.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

这条命令从随机权重直接进行300步渲染训练，没有旧权重，也没有隐式参数预热。
默认 random 模式会忽略终端残留的 `INIT` 并打印提示；旧 `STEPS` 环境变量也
不再触发预热。初始化模式、seed、预热步数均写入 `training.json`。
将来如需明确的检查点对照，必须同时设置 `INITIALIZATION=checkpoint` 和 `INIT`；
这是可选功能，不是本次训练的默认前提。直接调用 CLI 时，`--init` 仍是显式加载选项。
`RENDER_STEPS=600 RENDER_LR=0.00002` 等环境变量可修改短实验预算与学习率。
`tail -f` 的 Ctrl+C 只停止查看日志，不停止 nohup 训练。

## 应查看的结果

- `training.json`：完整目标说明、实用配置、输入点数、视角名称/索引。
- `loss.jsonl`：每步目标、图像 MSE、布局、梯度分组、更新量、采样视角、时间。
- `validation.jsonl`：固定噪声、视角、布局下的逐视角/逐试验 MSE、PSNR、SSIM、
  L1，以及符号数、档位组成；XYZ RMSE 只作为诊断，不参与 checkpoint 选择。
- `validation_images/000000/`：初始化的实际恢复效果；之后每次检查都保存同样的
  视角与布局。左到右是**照片、原 PLY 渲染、通信恢复渲染、绝对误差×4**。
- `charts/training_objectives.png`：按阶段分开。
- `charts/validation_quality.png`：四档布局的固定验证质量，分别以源渲染/照片为参照。
- `charts/optimization.png`：梯度、真实更新量、训练步耗时。
- `charts/last_layout_comparison.png`：最终检查各布局的预算与恢复质量。
- `charts/validation_metrics.csv`：图表对应的可读数据。
- `codec_best_render.pt`：按四种固定布局的平均源渲染 MSE 选择；每档仍单独报告，
  平均改善不代表所有档位改善。停止容差与最佳权重保存分别处理。
- `codec.pt`：最后执行步骤，不一定最好；`selection.json` 标明最佳步骤和分数。
- `codec_end_bootstrap.pt` / `codec_end_render.pt`：阶段末尾，便于定位变化。
- joint 可选阶段另存严格配对的 `codec_best_joint.pt` / `route2_best_joint.pt`。

SSIM 按显示范围截断后的图像计算；训练 MSE 和验证 PSNR 不截断。旧独立 benchmark
会对渲染输出先截断，不能把两种 PSNR 口径不加说明地混为一条曲线。
validation 的符号图仅含 payload，独立 packet benchmark 才包含现有包头开销。
原始照片指标用于观察源 PLY 本身的误差，**不作为这次训练的直接目标**。

训练途中可手动重新画图（新输出目录）：

```bash
python -m gaussian_jscc plot-stats --training "$OUT" --out "${OUT}_charts_$(date +%Y%m%d_%H%M%S)"
```

正式质量评估使用 best checkpoint 与数据集 test 相机，例如：

```bash
CUDA_VISIBLE_DEVICES=2 python benchmark_codec.py \
  --ply "$PWD/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply" \
  --checkpoint "$OUT/codec_best_render.pt" \
  --source "$PWD/data/tandt_db/tandt/truck" \
  --out "${OUT}_test" --device cuda --channels none awgn --snrs 10 \
  --tiers 1 2 3 --resolution 2 --save-images
```

短实验用于判断：固定图像指标是否持续改善、是否有档位退化、图像是否恢复实际
结构、梯度尖峰是否对应异常更新。它不是跨场景泛化或最终通信性能的证明。

## 本机验证范围

CPU 回归测试覆盖图像目标不截断预测/不回传教师、多视角 replay 与 checkpoint
梯度等价、渲染训练绕过旧辅助损失、固定验证保持随机数状态、分档指标和图像
保存、非有限图像损失中止以及完整阶段控制流。另运行了真实 Truck PLY 的20步
bootstrap 和模拟可微渲染器的三阶段短跑，检查日志、图表和序列化。
模拟渲染图不是 Truck 的恢复结果；本机无 CUDA，因此实际全场景画质与显存
占用仍须在服务器检验，不能根据这些单元测试宣称位置或通信恢复已经解决。
