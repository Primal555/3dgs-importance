# 逐点自身路径 + 多尺度 Context（logcov 一阶段实验）

## 不变的实验边界

- 输出仍是 XYZ、透明度、6 维对称 log-covariance、DC 和高阶 SH。
- `spatial_logcov_v1` 损失不变：位置粗/细尺度项、logcov Frobenius MSE、中心对齐的单 Gaussian 外观响应；并未重新引入相机投影/深度损失。
- 只训练一阶段；第二阶段场景渲染优化、mask 联合优化均为 0。
- 逐 Gaussian 的 0/8/16/32 复符号预算、档位条件、功率归一化和信道不变。**本次没有实现渐进档位**。
- XYZ 仍通过神经 JSCC 恢复；无原始 XYZ、局部中心、聚类对应关系或干净特征绕过信道。
- 默认随机初始化、固定 10 dB AWGN、固定学习率 2e-4、不做梯度裁剪。

## 新结构

通过 `--architecture learned_split_logcov --context-mode multiscale_self` 显式选择。
旧的 `window` 模式不改变；旧模型哈希/检查点保持可读取。切换模式不能加载旧模式权重冒充新模型初始化，CLI 会拒绝不匹配。

编码端：

```
每个 Gaussian 的几何/外观特征 -> 逐点 MLP -> 自身符号 ─────────────┐
                       └-> 多尺度几何 Context -> 修正符号 × gate ┤
                                                               ↓
                                                  相加、截取、功率归一化
                                                               ↓
                                                        同一个 JSCC 信道
```

接收端：

```
自己的接收符号 -> 逐点 MLP -> 自身 XYZ/logcov/外观预测 ────────────┐
邻域接收特征   -> 多尺度序列 Context -> 属性修正 × gate ──────────┤
                                                               ↓
                                                   最终解码 Gaussian
```

自身路径跨过所有邻域混合，不跨过信道。接收端自身路径不使用槽位序号；Context 路径使用原有正弦槽位编码，不包含全局点 ID。几何 Context 可辅助外观，外观解码分支不反馈决定几何输出。

多尺度 Context 在每个处理块内包含三个并行尺度：

1. 原分辨率的移位窗口注意力（默认 32 点窗口、2 层）。
2. 每 4 个槽位取有效点特征均值，再做窗口注意力。
3. 每 16 个槽位取有效点特征均值，再做窗口注意力。

粗尺度特征按固定槽位映射回每个 Gaussian，再学习融合。默认 256 点块中，粗尺度能覆盖比原来 32 点窗口更大的范围；**不跨 256 点处理块**。这是基于 Morton 顺序的多分辨率 Context，并非精确 kNN，也不是完整 PTv3 移植。无需新增 spconv/torch_scatter/FlashAttention 依赖。

发送端还对有效源坐标做同样的池化，用池化中心构造相对几何偏置；这些中心只在编码端使用。接收端池化只依据固定槽位和 q>0 掩码，不需要源坐标。q0 和 padding 不参与均值或注意力。池化不减少输出 Gaussian 数量，也不把同一块的 Gaussian 强制分配同一档位。

编码 Context、解码几何 Context、解码外观 Context 各有一个可训练标量门控，实际系数为 `tanh(gate)`。参数初值 0.1（系数约 0.10）是显式工程初始化，让自身路径开始时占主导，但不把 Context 关闭到没有分支梯度。**它不是新损失权重，也不保证训练不坍缩。** 门控可变正或负，只是修正幅度系数，不是语义重要性或可信度指标。

## 服务器运行

默认 SH3、hidden=96、depth=2 时，参数量从旧窗口结构的 865,706 增至 1,542,199。通信符号预算不变，但模型容量和计算量增加；不能把潜在提升全部归因于某一个结构细节。GPU 耗时/显存需要实测。

激活环境、拉取更新，再选择当前确实空闲的 GPU。下面 GPU 2 仅为示例：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git pull --ff-only --no-recurse-submodules origin main
OUT="$PWD/output/truck_multiscale_logcov_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_multiscale_logcov_bootstrap.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

默认一阶段 5000 步、每 500 步保存检查点；完成训练后自动评估每个检查点的图像和 PSNR。**图像不是每 500 步训练时即时生成，而是随后顺序生成**，保持上一版运行方式。可设置 `BOOTSTRAP_STEPS=2000` 缩短迭代；启用历史渲染时步数须为 `SAVE_EVERY` 的正整数倍。`RENDER_HISTORY=0` 可关闭离线渲染，但不改变训练。

环境变量 `PLY`、`SCENE` 可覆盖数据路径；其他配置沿用 `test_logcov_codec_bootstrap.sh`。继承的 `INIT`、`RENDER_STEPS`、`JOINT_STEPS` 不会使启动脚本加载旧模型或开启二阶段。

## 观测与验收

所有产物在同一个 `$OUT` 内：

- `training.json`：结构模式、损失和接收端信息边界。
- `loss.jsonl`：原有损失、各分支梯度、实际参数更新，新增 `context_gates`。
- `charts/training_context_gates.png` / `.csv`：三个门控的发展。
- `bootstrap_validation.jsonl`：固定留出块的位置、形状和外观误差。
- `render_history/validation_images/`：每 500 步的恢复图；`render_history/charts/`：图像质量曲线。

仍应同时检查完整解码图像、位置误差相对源 Gaussian 尺度、原生形状、透明度，以及不同档位的表现。门控增大或一阶段 loss 下降不能单独证明恢复质量提高。`codec_best_bootstrap.pt` 依旧按局部初始化目标选择，不冒充渲染最优模型。

新增 CPU 测试覆盖：关闭 Context 时自身路径独立、粗尺度跨远槽位传播、q0 不泄漏、逐点档位/符号功率、独立 packet 接收、旧哈希兼容、带噪声反传、全部输出头/门控梯度、直接/replay/checkpoint 等价，以及相同预测下损失与旧版完全相等。短拟合检查验证优化可以下降，不代表真实 Truck PSNR 或跨场景泛化；CUDA 渲染质量和性能仍需服务器实验。
