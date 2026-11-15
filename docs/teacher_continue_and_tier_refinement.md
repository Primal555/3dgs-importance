# 教师续训与逐层精修实验

## 教师：2000→2500步

入口：`scripts/continue_teacher_constant_lr.sh`。

源实验为 `truck_teacher_constant_lr_gpu2_20261114_213933`。从末步恢复概率表、历史反馈、视角调度器和随机状态，在新目录增加500步。保留原有最佳检查点供校准比较；最终分别报告2500步末态与校准选中结果。

源实验的全部优化设置保持一致：β=0.5，保留分支学习率0.1，正档位学习率0.4，正档位限幅2.0，保留分支限幅0.25，训练窗口32视角、校准32视角。固定编解码器、D_ref、结构项权重与C_ref；沿用数字侧信息文件计费。程序只允许修改总步数，拒绝计划中的其他隐式变化。完成后的源目录保持原样。

每100步校准一次，固定跑满新增500步。`metrics/validation.jsonl` 中的 `initial` 对应源实验2000步的部署图，`latest` 对应2500步。校准视角负责选模，测试视角负责报告。观察2000、2100、2200、2300、2400、2500步的RD、画质、成本及档位分布，判断后续提升是否稳定。

## 信道编解码器：q1→q2→q3

入口：`scripts/test_successive_tier_refinement.sh`。

从 `successive_mip360_gpu1_20261023_142948/training` 的原10 dB多场景模型开始。九个训练场景各更新150次，完成一个阶段后再进入下一阶段。实验版本为 `successive_tier_refinement_multiview_v2`。

| 阶段 | 更新参数 | 训练交付档位 | 更新次数 |
| --- | --- | --- | --- |
| q1 | 信道编码器第1层、信道解码器第1层 | 全q1 | 1350 |
| q2 | 信道编码器第2层、信道解码器第2层 | 全q2 | 1350 |
| q3 | 信道编码器第3层、信道解码器第3层 | 全q3 | 1350 |

共4050次实际更新。每阶段使用新的Adam，学习率1e-5；优化器在九个场景间共享。语义编码器、语义解码器、邻居修复模块和归一化统计固定。每一步只更新当前层，其他信道层冻结。

损失沿用q3单独精修实验：源PLY渲染RGB MSE＋latent MSE，latent权重1。训练使用10 dB AWGN，每步32个不同训练视角，噪声随训练持续采样。每个场景、每个阶段独立维护相机轮换队列，复用学生网络的轮换器。检查点保留相机顺序、游标、随机状态及实际访问次数；日志记录已覆盖视角数与平均访问次数。

同一步的32个视角共享一次带噪场景重建，逐视角计算渲染梯度后取平均，再执行一次编解码器梯度回放。视角数增加不改变渲染损失与latent损失的相对权重。语义latent在场景首次使用时缓存，跨阶段复用。

每个场景保留源模型原有的验证视角，再从原训练池等间隔补充至16个，补充视角从本轮训练池移除。新增验证相机曾参与源模型训练，因此这组验证衡量本轮精修的效果；Room承担未参与训练场景的测试。源PLY、图像及相机几何先按原记录核验，再利用已加载的相机重新划分，避免重复加载图像。源实验的测试视角保持独立。

以Truck为例，原215个训练视角中划出12个，加上原4个验证视角，共16个验证视角；本轮训练池为203个。每步32视角，首次完整覆盖约7步；每个阶段150步内平均每个训练视角参与约23.6次。

Room只进行留出测试。每场景每50次更新，在16个验证视角进行4次固定配对噪声试验，记录当前档位画质、latent误差及跨噪声试验标准差。初始、q1结束、q2结束、q3结束四个固定端点，均评测q1/q2/q3及固定混合图：无噪声与10 dB AWGN，测试视角最多32个，4次配对噪声试验，记录相对源渲染及照片的PSNR、SSIM、LPIPS、latent误差和通信开销。训练使用固定阶段末步权重。`--trials` 同时控制验证、端点评估和实际交付次数，默认4。

代码检查q2阶段的全q1输出保持一致，q3阶段的全q1及全q2输出保持一致。混合交付中，邻居修复可能把高档位变化传递给附近低档位点，因此单独记录混合交付效果。

最终可比较三类变化：同一模型中各阶段的增益、q3结束相对初始的总增益，以及相对既有q3单独精修实验的最终质量。既有q3单独精修为2250次更新、每步2视角；本轮为4050次更新、每步32视角，解释结果时同时报告更新数、渲染视角使用量与耗时。

## 实际链路与输出

通信成本为复信道使用次数，XYZ、完整元数据、块尺度分别按实际文件成帧；使用LDPC(4096,6144)、16QAM、48字节文件头、每码字4字节CRC、508字节有效载荷、1536次复信道使用，无ARQ。

教师收尾实际交付selected、latest、uniform_q2；逐层精修收尾实际交付Truck的q1/q2/q3/mixed。使用原4次试验，逐文件检查CRC与SHA256，失败试验计入通信成本，画质在成功交付条件下报告。每条实际计费与原评测账本核对。

下载各实验的 `review.tar.gz` 即可分析。包内包含训练计划、曲线记录、阶段结果和实际链路报告，不包含模型权重与latent缓存。续训时保留服务器上的整个实验目录。启动脚本默认新实验；中断后使用相同参数、相同OUT，并设置 `RESUME=1`。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git switch experiment/feedforward-student
git pull --ff-only origin experiment/feedforward-student
conda activate maskgs

GPU=2
SOURCE="$PWD/output/truck_teacher_constant_lr_gpu2_20261114_213933"
OUT="$PWD/output/truck_teacher_continue2500_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 nohup bash scripts/continue_teacher_constant_lr.sh \
  "$SOURCE" "$OUT" --extra-steps 500 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
printf '教师实验：%s\n日志：%s.log\n' "$OUT" "$OUT"

GPU=1
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/successive_tier_refine_views32_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 nohup bash scripts/test_successive_tier_refinement.sh \
  "$RUN" "$OUT" --steps-per-scene 150 --views-per-step 32 --validation-views 16 \
  >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
printf '逐层精修：%s\n日志：%s.log\n' "$OUT" "$OUT"
```
