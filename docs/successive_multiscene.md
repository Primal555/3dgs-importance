# Joint B 三场景迁移与第四场景零样本测试

独立入口：`gaussian_jscc.successive_multiscene`；单场景 Joint A／B 与分配器入口保持不变。

## 实验目的与范围

从已完成的 `truck_successive_jointB_gpu2_20261022_142221/checkpoints/final/codec.pt` 继续训练，
观察 Truck 专用模型加入 Train／Playroom 后的恢复能力，以及对预留 DrJohnson 的零样本泛化。
继承四个网络的全部权重、PointNet 式两端上下文、successive 发送端残差编码、完整前缀接收、邻居修复和功率设置。
本轮不重新训练属性初始化／信道初始化阶段，不学习档位分配表。

- 训练场景：Train、Truck、Playroom，均衡轮换，每场景 **2000次真实更新**，共 **6000次 optimizer.step**。
- 训练模块：语义编码器、信道编码器、信道解码器、语义解码器全部更新，学习率均为 `1e-5`。
- 使用新的共享 Adam；不继承源 Adam，场景切换时不清空动量。
- 目标沿用 Joint B：源 PLY 渲染 RGB MSE，加权重为1的当前 `y` 恢复 MSE（目标侧 stop-gradient）。
- 每场景独立按 q1→q2→q3→随机 mixed 循环；累计符号预算为8／24／48，SNR固定10dB。
- 可靠16bit XYZ、实际交付的 binary16 块尺度均保留；无需学习 XYZ，也不在本轮更换结构。
- 保留源 Joint B 的属性 mean/std，禁止根据新训练场景或留出场景重新拟合，以免更换原有表示基准。
- bbox、Morton分块、输入数据、相机、参考图缓存逐场景独立；网络及优化器共享。

## DrJohnson 如何测试

训练开始前，先在三个训练场景和 DrJohnson 的真正测试相机上记录初始表现。
之后只用三个训练场景的训练相机更新；每250个均衡轮次，在这三个场景的验证相机上记录走势。
DrJohnson不参与中途验证、选模、统计量拟合、校准或任何优化器更新。
6000次更新后，使用同一组测试相机、相同噪声种子，对四个场景再次评估。

DrJohnson在接收端使用自身实际交付的 XYZ 和 bbox 元数据，属性通过共享编解码器恢复。
接收端不获得它的原始属性、教师 `y` 或源图像；它的源 PLY 只作为发送端输入和评估参考。
每个端点检查 clean（诊断）、无噪声信道、10dB q1/q2/q3/mixed。
**mixed是固定随机正档位组合，尚未执行学习后的重要性分配。**

训练使用已有剪枝 manifest。若其中只有三个训练场景，通过
`--heldout-manifest configs/multiscene_tandt_db.json` 定位预留的 DrJohnson。
默认该场景为原始预训练 PLY；其点数与三个剪枝训练场景不同，结果中保留各自源点数和实际通信成本。
这轮可验证跨场景适用性，同时包含剪枝程度差异；结果解读应保留这个条件。
入口会核对角色、路径别名、SH阶数及源 Truck 指纹／相机拆分，不会静默换成其他场景。

## 保存内容与验证边界

- `training.json`、`plan.json`：来源指纹、三个训练场景、留出场景、冻结统计量及步数定义。
- `metrics/loss.jsonl`：6000次真实更新，含 `scene/scene_step/layout`、渲染和latent损失、梯度、逐模块更新、分段耗时、显存峰值。
- `scenes/<scene>/evaluation/{initial,final}/`：对原 PLY／真实照片的 PSNR、SSIM、LPIPS、颜色误差、latent MSE/NMSE及通信成本；各路径保存首尾两个代表视角。
- `scenes/<train_scene>/evaluation/round_*/`：中途验证，仅数值，不额外保存图。
- `comparison/summary.json`、`endpoints.csv`：四场景训练前后对照，DrJohnson明确标为 `zero_shot_unseen_scene`。
- `charts/heldout_quality.png`：第四场景三档质量变化；`training_scene_quality.png`：三个训练场景对照。
- `checkpoints/{initial,latest,final}/codec.pt`：全部网络、固定统计量、共享Adam、随机状态、逐场景计数和轮换游标。
- `complete.json`：全部训练、四场景最终测试和汇总生成完成后才写入。

默认每场景最多32个真实测试视角、4次配对噪声；`--test-views 0`表示全部测试视角。
通信成本包括模拟payload、实测坐标／元数据／尺度帧，数字侧流沿用来源实验的净 bit/use 假设；共享模型权重不计入单次载荷。
CPU合成渲染测试只验证数据隔离、参数更新和续训行为，不作为真实GPU画质证据。

## 服务器3号卡启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_successive_jointB_gpu2_20261022_142221"
MANIFEST="$PWD/output/maskgaussian_three_20261010_190401/multiscene_pruned.json"
OUT="$PWD/output/successive_multiscene_gpu3_$(date +%Y%m%d_%H%M%S)"

# 检查来源和四场景路径，不加载CUDA、不训练、不创建输出。
CUDA_VISIBLE_DEVICES=3 bash scripts/train_successive_multiscene.sh "$RUN" "$MANIFEST" "$OUT" --dry-run

CUDA_VISIBLE_DEVICES=3 nohup bash scripts/train_successive_multiscene.sh \
  "$RUN" "$MANIFEST" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

仅预检成功后启动后台任务。SSH断开不影响nohup任务；Ctrl+C只退出tail。
初始四场景评估需要时间，日志中的初始测试不消耗6000次训练预算。
首次LPIPS权重需已缓存或网络可用；缺失会明确报错。
中断后保留原RUN、MANIFEST和OUT，确认旧进程已退出，使用同一命令并设置 `RESUME=1`：

```bash
CUDA_VISIBLE_DEVICES=3 RESUME=1 nohup bash scripts/train_successive_multiscene.sh \
  "$RUN" "$MANIFEST" "$OUT" >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```
