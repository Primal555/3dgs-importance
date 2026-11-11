# 一次前馈学生分配实验

独立分支：`experiment/feedforward-student`。代码入口为 `gaussian_jscc.student_policy_experiment`。

本轮先在 Truck 上验证：学生网络从随机权重开始，学习限幅1.0教师的十次部署概率，随后学习属性、latent、空间关系与交付 RD 的联系。冻结原有四个编解码模块及源高斯属性。独立预测入口支持同一场景与 manifest 指定的新场景，执行0次优化器更新；新场景实际泛化效果由后续实验评价。

## 结构与输入

- 属性64维、全保留latent64维、可见性32维编码后形成128维逐点特征。
- 缓存两级空间分组与真实XYZ邻居；每级2个邻域残差块，宽度128／256，MLP扩展2倍。
- 源逐点特征、细／粗空间特征与全局摘要共同送入输出头。逐点MLP分块并在训练中重算激活；空间索引固定，全局特征保留梯度。
- 保留头二分类，正档位头三分类。保留logit大于等于删除logit时保留，再取正档位argmax。训练硬前向和部署使用相同规则。

源属性与latent按原PLY行号排列。`scene.raw`使用Morton包顺序，预处理会显式转回原行号。原编码块与空间分组是两个独立索引体系。

可见性从8个训练相机中采集源PLY的精确T-alpha贡献、投影面积、深度、观察方向及源图像合成颜色。每相机14项描述，汇总均值、最大值、标准差和可见比例，形成43维输入。相机数量不足8时覆盖全部可用训练相机。第一相机比较缓存光栅器和主光栅器图像，失败时停止。可见性／latent／空间图均在准备阶段缓存；部署没有接收噪声或教师输出作为输入。

## 训练预算与口径

| 阶段 | 默认预算 | 学习目标 |
|---|---:|---|
| 源场景准备 | 一次；8训练相机 | 属性统计、全保留latent、可见性及空间图 |
| 动作蒸馏 | 1000次完整点集遍历 | 全点保留软交叉熵＋按保留概率加权的正档位软交叉熵 |
| RD微调 | 1000次场景更新 | 当前硬交付的全点保留反馈与正档位反馈 |

蒸馏AdamW学习率1e-4，RD学习率1e-5；切换阶段创建新优化器。梯度裁剪上限1。RD阶段前半段蒸馏权重从0.1线性降为0，后半段仅使用RD反馈。权重与预算为首版工程设置，可通过命令行调整。

RD训练每次轮换4个训练视角。保留分支持续覆盖全点；正档位每块均匀无放回选2个当前保留点，评价另两个正档位。采样点以纳入概率倒数加权，使共享参数梯度具有全点求和的目标尺度。不同数量的保留点具有不同纳入概率，按实际块内数量计算。共享参数不乘旧概率表的N_source镜像更新增益。

最多32次独立真实换挡纠偏，沿用原始条件块编码、配对AWGN和候选批次128。反馈只来自当前上下文；保存访问与精确核验计数，不将旧收益累积到新参数版本。

目标沿用 `D/D_ref + beta*C/C_ref`，`D=MSE+lambda_s*(1-SSIM)`。训练的keep梯度固定本次恢复属性，并用当前硬条件档位的载荷成本；真实纠偏补充部分重新编码影响。正档位梯度包含整块受影响属性与块尺度变化。训练侧信息使用边际代理，真实校准与最终评价重新计算完整XYZ、尺度、元数据及属性信道使用。`hard_rd_proxy`明确是单训练窗口的成本代理，最终结论看`metrics/validation.jsonl`的完整部署RD。

本轮教师为`truck_first_order_fast_cap1_beta0p5_gpu1_20261110_111326`，使用其第1600步selected概率与档位。启动核对codec、PLY、相机、beta、随机种子、试验数、部署图哈希及RD参考，并检查正档位限幅1.0、十次部署规则、选中步骤和概率文件SHA256。用该概率重新生成导出档位图，验证原始行索引与selected文件对应。校准相机和测试相机均不进入学生输入统计或梯度训练。蒸馏阶段和RD阶段每100次更新评价校准集，保存最佳与末步。蒸馏结束时的initial表示蒸馏后学生的测试结果。最终对selected／latest／teacher／全q2／全q3报告源PLY与照片的PSNR、SSIM、LPIPS、通信成本及RD。

令教师四档概率为`p`，精确枚举十次抽样的计数及平局处理，得到`d=Deployment10(p)`。保留头目标为`[d0, d1+d2+d3]`，其中`d0=p0^10`；正档位头目标为`d[1:]/sum(d[1:])`。正档位CE按每个点的保留概率加权，再除以全场景保留概率之和。因此单热目标时与原有两分支平均CE完全一致，全删除时正档位损失及梯度为0。RD阶段的蒸馏辅助项沿用相同软目标。学生仍使用独立保留硬判决和正档位argmax部署。

软标签按2048行一批在CPU计算，缓存为`preparation/teacher_soft_targets.pt`。恢复时核对教师身份及缓存内容哈希。网络初始化、结构、学习率、RD预算与视角数沿用当前设置。旧硬标签模式可通过原启动脚本使用；本轮软标签模式用独立脚本及新OUT。

蒸馏日志中keep是全点保留判断准确率，action是完整四档动作准确率，positive是仅在教师保留点上、不考虑保留头判断的正档位头准确率，q0/q1/q2/q3分别是各教师类别的最终动作命中率。训练准确率和CE来自本次优化前的前向；校准评价使用优化后的权重。

软标签模式额外输出`keep_KL`、`tier_KL`，衡量两个预测分布与教师分布的差距，越低越好。`expected_action_agreement`是学生硬选择在教师部署分布下的平均概率；原有准确率依旧对照教师导出的那一份抽样图。日志JSON同时保存两项CE与KL，两个训练阶段均可追踪。实际分配效果以校准及测试视角的画质、开销、RD评价。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false fetch origin
git switch experiment/feedforward-student
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin experiment/feedforward-student
conda activate maskgs

GPU=3
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
TEACHER="$PWD/output/truck_first_order_fast_cap1_beta0p5_gpu1_20261110_111326"
OUT="$PWD/output/truck_student_soft_cap1_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 nohup bash scripts/train_student_soft_policy.sh \
  "$RUN" "$TEACHER" "$OUT" --beta 0.5 \
  --distill-epochs 1000 --rd-steps 1000 --candidate-batch 128 \
  >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
printf '实验目录：%s\n' "$OUT"
tail -n 40 -F "${OUT}.log"
```

GPU按空闲情况设置。本轮使用上述cap1教师目录；启动日志会打印其选中步骤和概率文件哈希。

恢复：相同RUN、TEACHER、OUT和全部参数，将`RESUME=0`改为`RESUME=1`。准备缓存和`checkpoints/latest/student.pt`一起保留；准备阶段只完成部分相机时，可沿用同OUT以`RESUME=1`复用已有相机缓存。

## 直接预测

```bash
PREDICT_OUT="$PWD/output/truck_student_predict_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/predict_student_policy.sh \
  "$OUT/checkpoints/best/student.pt" "$RUN" truck "$PREDICT_OUT" \
  --beta 0.5 >"${PREDICT_OUT}.log" 2>&1 &
```

新场景使用`SCENE`名称和`--manifest /path/scenes.json`，manifest提供其PLY和原始相机数据路径。该入口只准备源特征，固定学生权重及归一化，直接输出档位。新场景为得到相同RD参照，需要在独立源相机上仿真全q3参考；此开销计入准备时间。空间分组及输入统计参数须和训练协议相同。

`student_forward_seconds`测一次网络前馈，`feature_preparation_seconds`记录特征准备／缓存读取时间，`reference_preparation_seconds`记录新RD参照的生成时间。后两者之和为`preparation_seconds_this_run`，再加前馈时间得到`allocation_seconds`。同PLY更换相机几何、成像设置或侧信息效率时会重新生成参照。后续完整交付评价另计。训练后的大准备缓存可加快同场景重评，新场景重新准备。

直接预测另外记录`predict_startup_to_policy_seconds`，覆盖本次请求从输入检查、模型／场景加载、准备、学生初始化到第一次输出档位的全部时间。它包含`allocation_seconds`之外的首次加载开销。

## 查看与保存

- `metrics/loss.jsonl`：蒸馏CE、动作准确率、RD梯度代理、训练窗口实际失真、采样权重、梯度与耗时。
- `metrics/validation.jsonl`：校准与最终完整部署结果。
- `review_summary.json`：相对教师、全q2、全q3的质量／成本／RD差值。
- `deployment/selected_tiers.npy`：原PLY顺序的最终档位；同时保留末步图。
- `checkpoints/best/student.pt`：校准最佳学生，供预测。
- `checkpoints/latest/student.pt`：精确恢复所需学生与优化器状态；不复制原codec权重。
- 下载`review.tar.gz`检查结果。包内只有日志、最终档位、运行身份及初始／最终图像，排除准备缓存和优化器。

蒸馏阶段以完整点集遍历计数，RD阶段以场景更新计数，日志分别标记。单个Truck实验检验一次前馈和继续学习的有效性，多场景训练与完整留出场景的泛化验证另行开展。
