# 逐点档位分配：固定 beta 率失真与预算约束

## 固定 beta 率失真模式（当前主线）

启动脚本为 `scripts/test_rate_distortion_allocation.sh`，复用本页描述的冻结编解码器、完整 q0 反馈、真实混合邻域候选、镜像更新与同一步缓存。默认单场景 2000 次更新、每次四视角累计，不重训编解码器，不接联合微调。

必须显式提供 `--beta BETA`，允许 0，拒绝负值/NaN/Inf；与两个预算参数互斥。beta 全程固定，没有总成本上限或档位数量要求，也不运行自适应价格控制器。

训练目标：

`L_proxy = D_train / D_ref + beta * C_proxy / C_ref`

- `D_train`：源 PLY 渲染 RGB MSE；`D_ref`：初始化全 q3 在校准视角和 10 dB 下的平均 MSE，固定保存。
- `C_ref`：同一场景全 q3 的实测总复信道使用次数，固定保存，仅为参考尺度。
- `C_proxy`：符号精确计数与最近实测压缩侧流构成的局部代理；不会将代理值标为真实部署成本。
- 每 25 步刷新十次规则部署的实际成本和侧流代理，不改变 beta；每 250 步在校准视角计算 `RD_deploy = D_calibration / D_ref + beta * C_actual / C_ref`，以最低评分选 `best_rd`。同分时依次比较更低失真、更低成本。
- 测试相机只用于初始与最终报告，不参与 `best_rd` 选择。`complete.json` 的 `selection_rd_score` 是校准选取分数，`rd_score` 是选中部署在最终测试视角的分数，二者明确区分。

本模式保存 `checkpoints/best_rd/state.pt`、`objective_reference`、固定 beta、覆盖/相机游标与 RNG，支持同参数 `RESUME=1` 精确续训。改变 beta 或在预算模式与固定 beta 之间切换必须使用新 OUT，旧预算 checkpoint 不会被静默转换。

输出仍在一个 OUT 内；`metrics/control.jsonl` 记录 `cost_refresh` 事件，没有预算残差或动态价格。校准及测试的实际成本、q0–q3 占比、源 PLY/真实照片 PSNR、SSIM、LPIPS 继续保存。`comparison/endpoints.csv` 增加 `beta,d_ref,c_ref,normalized_distortion,normalized_cost,rd_score`；`charts/rate_distortion_training.png` 展示成本与固定 beta，`charts/deployment_rd_score.png` 将训练代理与实际部署校准分数分开画出。其余覆盖和实际档位占比图保留。初始与最终代表图及档位着色图保留，同图去重。

以下 `0.1` **仅为显式参数的示例**，尚未证明是最佳惩罚强度。新双参考归一化下不能将历史 `0.005` 解释为相同惩罚；例如 beta=0.1 时，节省 `0.1*C_ref` 对目标的改善为 0.01，与增加 `0.01*D_ref` 的失真相抵。多个 beta 实验要固定源模型、场景、参考尺度和评估条件，以实际总成本—PSNR/SSIM/LPIPS 曲线比较，不能比较不同 beta 的总 loss 大小。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/successive_multiscene_gpu3_20261022_230109"
SCENE=truck
GPU=2
BETA=0.1
OUT="$PWD/output/${SCENE}_rd_beta${BETA}_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" bash scripts/test_rate_distortion_allocation.sh \
  "$RUN" "$SCENE" "$OUT" --beta "$BETA" --dry-run

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_rate_distortion_allocation.sh \
  "$RUN" "$SCENE" "$OUT" --beta "$BETA" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

原进程确认退出后，保留原 RUN/SCENE/OUT/BETA 和所有设置，添加 `RESUME=1`：

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/test_rate_distortion_allocation.sh \
  "$RUN" "$SCENE" "$OUT" --beta "$BETA" >>"${OUT}.log" 2>&1 &
```

文献对应：

- [NTSCC，式（17）](https://arxiv.org/html/2112.10961)：失真与主要内容/数字侧信息信道资源的固定权重权衡。
- [FCGS，第 3.4 节式（8）](https://arxiv.org/html/2410.08017)、[HAC，第 3.4 节式（9）](https://arxiv.org/html/2403.14530)：渲染质量与归一化编码成本。
- [RDO-Gaussian，第 3.2 节](https://arxiv.org/html/2406.01597)：逐 Gaussian 表示复杂度的率失真优化。
- [MaskGaussian，第 3.3–3.5 节](https://arxiv.org/html/2412.20522)：概率保留与 masked-render 反馈。

这里的全 q3 双参考归一化、镜像更新、四视角窗口与实际成本代理是项目工程设计；2000 步和 beta 数值没有作为原文最优参数使用。固定参考下目标等价于 `D + beta*D_ref/C_ref*C`。定向 CPU 测试验证执行、核算、选择与续训；服务器 CUDA 收敛和耗时仍需实测。

## 原预算约束模式（兼容保留）

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
