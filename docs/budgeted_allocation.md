# 总预算约束的逐点档位分配实验

入口为 `gaussian_jscc.budgeted_allocation`，启动脚本为 `scripts/test_budgeted_allocation.sh`。保留旧 `successive_allocation` 实验行为；本轮不重训编解码器，也不加入联合微调。

## 本轮训练什么

- 接受已完成的 successive Joint A 或已完成的 successive 多场景共享模型。必须明确选择一个源模型已经训练过的场景；不允许把共享模型的留出场景隐式用于分配学习。
- 全部编解码器权重、属性 mean/std 固定；新建逐场景保留/条件档位概率表。初始保留概率 0.99，保留后的 q1/q2/q3 概率均匀。
- q0/q1/q2/q3 仍为 0/8/24/48 个累计复符号，保留源模型的 16bit XYZ 和 binary16 尺度流，固定 10dB。
- 每一步硬采样一次，同一恢复场景依次经过四个训练视角，平均场景梯度后才更新一次概率表。训练过程中不物理删点。
- 每块轮换评价一个实际保留点的两个备选正档位。其他点的档位保持真实混合状态，重算整块发送、尺度、恢复与语义解码，并计入块内邻居响应。未被评价的正档位概率行不更新。
- 所有关闭点保留 MaskGaussian 式零门控反事实属性，无 q0 数量上限。该门控是一阶代理；额外真实删/留核验会重新编码变化后的保留集合，不复用原来的语义 latent。
- 概率使用保留收益幅度的 log-domain 镜像更新。两分支的图像项与成本项共用固定 `N_source` 全局增益，不做逐行幅度归一化。

## 预算与价格

必须显式指定一个预算，没有隐式默认值：

```text
--budget-total-uses B
--budget-ratio-to-q3 r
```

第二种表示 `B = floor(r * C_q3)`，`C_q3` 是相同场景、相同模型的实测全 q3 总成本。初始化后固定保存，精确续训不重新定义基准。它和此前通信惩罚 beta 不是同一个参数。

总成本为属性复符号数，加上可靠 XYZ、完整元数据/档位表、实际尺度帧的复信道使用次数。可靠侧流继续沿用输入实验明确记录的净 bit/use 假设，并同时报告真实字节数；不把模拟符号保存成 float32 文件的大小当作无线带宽。共享模型权重不计入本次场景交付。

优化目标为：

`L = source_render_MSE / D_ref + lambda * (C_proxy / B - 1)`

`D_ref` 是初始全 q3 校准视角的平均 MSE，最小数值下限 1e-12；原始数值与下限均保存。它用于固定量纲，不规定画质容忍范围。

训练中 payload 精确计数，压缩侧流使用最近实测值构造局部代理。每 25 步基于固定 seed 的真实十次部署重新测量成本，EMA 后更新非负价格；超预算会积累提高价格的压力，预算富余则降低价格。默认初始价格 0.01、价格步长 0.05、EMA 0.9、keep/tier 镜像步长各 0.1，均为可调工程参数，不宣称最优。

只有真实部署成本达标的候选才能进入 `best_feasible`。若没有达标方案，状态为 `budget_unmet`，只保留 `lowest_violation` 诊断候选，不强制删点、不伪造可行部署。

## 数据边界与保存

默认从源记录的训练相机中另留 4 个校准视角；它们不进入分配训练窗口，最终测试相机也不参与预算价格、步数或最佳模型选择。每 250 步用校准画质挑选达标方案；最终测试仅报告选定方案及初始/最新方案。

只在初始与最终/选中端点保存代表恢复图和档位贡献着色图。着色表示保留高斯对像素的可见贡献，背景不是 q0 的空间分布。`mixed` 正档位并不等于学习分配后的实际部署图；本实验主要展示保存的真实十次部署图。

最新与选中档位图完全相同时只做一次最终评估，`evaluation/latest/alias.json` 指向 `evaluation/selected`，不再重复保存图片。q0 审计的 `first_order_delta_mse` 对应训练实际使用的带符号门控估计；另存 `full_scene_tangent_delta_mse`，避免将两个估计量混淆。未完成日志尾行会隔离备份后恢复；已提交或中间记录损坏仍报错。

一个 OUT 内包含：

```text
plan.json / training.json
metrics/loss.jsonl        # 每次实际更新：MSE、价格、期望/采样代理、覆盖、耗时
metrics/control.jsonl     # 实测十次部署成本、可靠流字节、价格/EMA
metrics/validation.jsonl  # 校准和最终测试记录，source/photo 分开
metrics/audit.jsonl       # 小规模真干预与一阶预测的误差/符号核验
checkpoints/{initial,latest,final,best_feasible或lowest_violation}/state.pt
deployment/              # 最新/选中离散档位、选中概率、绑定的 codec ID 和可行性
evaluation/              # 校准数值；初始和最终/选中代表图
comparison/endpoints.csv / summary.json
charts/                  # 成本—预算—价格、真实占比和候选覆盖
complete.json            # status 可以是 complete 或 budget_unmet
```

`latest/state.pt` 保存模型、概率表、控制器、预算参考、逐点计数/最后评价编号、块游标、视角遍历状态、全部 RNG 和未完成控制/校准事件。不保存临时编解码缓存。

每块每步只评价一个正档位目标，2000 步不意味着所有点都获得相同反馈次数。日志明确记录总覆盖、当前保留点未覆盖数和完整轮换次数，不能用未评价的零收益替代缺失信息。

## 服务器运行

以下 `0.7` 是示例预算，即实测全 q3 总成本的 70%；请按本轮要验证的预算选择。GPU 编号同样按空闲情况指定。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/successive_multiscene_gpu3_20261022_230109"
SCENE=truck
GPU=2
RATIO=0.7
OUT="$PWD/output/${SCENE}_budget${RATIO}_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" bash scripts/test_budgeted_allocation.sh \
  "$RUN" "$SCENE" "$OUT" --budget-ratio-to-q3 "$RATIO" --dry-run

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_budgeted_allocation.sh \
  "$RUN" "$SCENE" "$OUT" --budget-ratio-to-q3 "$RATIO" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

预算、视角或源模型改变时使用新 OUT。原任务确认退出后，使用同一 RUN、SCENE、OUT 和原参数续训：

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/test_budgeted_allocation.sh \
  "$RUN" "$SCENE" "$OUT" --budget-ratio-to-q3 "$RATIO" >>"${OUT}.log" 2>&1 &
tail -n 40 -F "${OUT}.log"
```

可调选项包括 `--window-views 2`（对照四视角）、`--allocation-steps`、`--initial-price`、`--dual-lr`、`--ema-decay`、`--keep-lr`、`--tier-lr`、`--control-every`、`--validate-every`、`--profile-every`、`--audit-every` 和 `--no-lpips`。保留源相机划分及 batch 设置，不随意在本入口改网络。

## 性能测量范围

保留同一步语义 latent、实际恢复和 q0 反事实，候选逐批释放；不跨更新缓存预测，不在逐点候选内压缩坐标。XYZ 成本按保留集合缓存，预算控制不额外渲染。

CPU 16 点、两块、四个合成视角的调用核验：旧提速路径为 5 次完整 tier forward，新路径为 4 次；两者报告的临时缓存分别约 0.002419/0.002136 MiB（不含公共整场景与光栅化器占用）。这只验证调用/缓存结构，实际 CUDA 耗时与收敛收益须以服务器为准。

详细计时默认每 10 步开启，使用 CUDA events 聚合，避免逐块同步；阶段报告给出优化步时 p50/p90，并将校准测试、预算核算耗时分开。

本轮实现取舍：两次异构候选批次保留每块单点干预；沿用局部一阶代理而不宣称全局最优；保持旧实验代码及用户未提交内容；按已确认范围进行定向测试，不跑全仓历史训练回归。CUDA 光栅化器行为、真实速度/显存、长期率失真收益及控制器参数仍需服务器验证。
