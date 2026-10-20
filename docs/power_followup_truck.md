# 逐点／块归一化的三项后续实验

入口：`scripts/test_power_followup_truck.sh POWER_SUITE NEW_OUT [extra options]`。
默认源为之前六组实验的父目录，程序只读取其中的 `point_awgn10` 和 `block_awgn10`，每组依次完成三个阶段。统计尺度组及无噪声训练组不进入本次训练。

## 实验一：在原始4000步模型上定位属性损伤

对同一份接收场景，分别保留正常结果、用clean属性替换DC+SH、替换大小+旋转、替换不透明度。XYZ始终保留当前16bit可靠交付坐标。clean属性来自该模型冻结的语义编解码器，不使用原PLY属性作为理想替换，原PLY仍作为图像质量参照。

q1/q2/q3均测试无噪声通信及10dB通信；同一档位、相同噪声试验的所有替换共享一份实际接收结果，不重新采样噪声。记录对原PLY和照片的PSNR、SSIM、LPIPS，以及颜色误差、逐视角逐噪声记录和相对正常恢复的指标变化。替换是不可部署的诊断，各项提升不能直接相加。

## 实验二：两组分别增加4000次信道更新

保留源模型全部权重、信道Adam状态和原训练游标，默认累计至8000步。LR=1e-4，原始y MSE，语义两端冻结。y在原始空间块上抽取并缓存到CPU，不改变逐点／块归一化或8/24/48符号预算，不增设分配器。q1/q2/q3轮换，批次与噪声使用独立且配对的随机序列，继续上一实验的累计步数采样规则。

## 实验三：两组分别增加1000次联合更新

从各自续训后的信道检查点接上联合阶段。保留信道两端的Adam动量与步数，加入语义解码器的全新Adam状态；语义编码器及属性标准化参数保持冻结。

- 信道编码器／解码器LR均为1e-4。
- 语义解码器LR=1e-5。
- 目标：原PLY同训练视角的图像MSE + (1-SSIM)/3 + 原始y的MSE。
- 完整场景replay反向，沿用源场景每批64个原始块、每步2个训练视角；不积累梯度。
- q1/q2/q3/mixed循环，mixed仅为随机正档位；没有学习资源分配或q0删点。
- 每250步验证；累计总更新数默认最终为9000，其中源4000、信道续训4000、联合1000。

SSIM与latent权重沿用现有联合目标；LPIPS只评估。联合后clean解码器可能发生变化，因此记录clean画质，但不会因clean下降自动停止训练。y目标由冻结语义编码器定义，不随解码器变化。

## 视角与功率协议

初始、信道续训结束、联合结束三个端点使用**同一组最多32个实际test相机**和4次配对噪声，拒绝用训练相机替代。程序验证测试相机名称与训练视角不重叠，保存相机名单用于续跑校验。原场景test相机不足32时记录实际数量。clean和无噪声路径无需重复加噪，分别前向一次。

中间验证仍使用原有4个留出相机，并单独绘图。不要把其均值与32视角端点均值混合排名。多次查看test用于本次诊断对照，不是保留至最终论文评估的一次性未触碰测试集。

每个评估节点均保存全场景各段／前缀平均功率、能量分布、符号PAPR，以及固定32块8192点、8次噪声的latent逐维误差和逐层补偿诊断。物理属性响应在其中8个分散块测量，不等于全场景渲染指标。归一化统计不重新校准；本次仅逐点与块两种方式。

## 输出与恢复

所有结果在一个父目录：

```text
NEW_OUT/
  plan.json, point.log, block.log, complete.json
  point/ 与 block/
    control.json, training.json, test_views.json, complete.json
    ablation/results.json, summary.csv, images/（仅q3代表视角）
    metrics/loss.jsonl
    evaluation/initial/, channel_end/, final/（统一test端点）
    evaluation/channel_*/ 与 joint_*/（中途validation）
      complete.json, power.json, diagnostics/, metrics/validation.jsonl
    checkpoints/initial/, latest/, end_channel_bootstrap/, joint_initial/, final/
  comparison/
    summary.json, test_endpoints.csv, attribute_replacements.csv
    test_endpoints.png, validation_curves.png
```

常规图像仅保存initial和final的代表视角；属性诊断只存q3的初始模型替换图，q1/q2保存数值。检查点在首次评估之前保存，所有端点都有完成标记。中断续跑跳过已完成诊断，恢复当前优化器和阶段游标，未完成端点会覆盖重算，不会重复叠加更新预算。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/truck_power_controls_20261020_184354"
OUT="$PWD/output/truck_power_followup_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_power_followup_truck.sh "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -F "$OUT/point.log"
```

GPU3须替换成当时空闲的卡。默认在一张卡上先跑完point三个阶段，再跑block三个阶段。父日志显示组别切换，各组日志显示训练进度。可用 `--modes block` 只选择一组，输出仍应使用新目录。

`--dry-run`只在CPU检查源检查点、权重配对和计划，不加载PLY、不创建输出。路径迁移时可用 `--manifest` 指定相同PLY和划分的清单。默认启用LPIPS，服务器需要有对应模型权重；可显式 `--no-lpips` 关闭。

恢复：同一RUN、OUT和全部预算参数，在脚本后加 `--resume-suite`。已完成组不再训练。输出源目录保护及参数一致性校验会拒绝覆盖原实验或不兼容的恢复。
