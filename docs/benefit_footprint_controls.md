# 候选覆盖区域与分离预测头对照

本轮保持原有 `CorrectionLearner` 损失、统一尺度和普通样本监督，测试候选区域输入与独立共同／档间预测头。
两个结构在同一份新标签上训练，各更新 2000 次。固定训练记录通过相同洗牌循环获得覆盖，保留原采集重要性权重。
旧在线分配器和之前的模型不受影响。

## 输入与网络

旧点级特征保留。新输入为目标点支持区域和整个受影响块支持区域的两份 9×9 网格。
范围取基线和四个候选档位的联合渲染 tile 边界，再加 5 像素背景上下文。
四档共用相同网格坐标，每份包含 15 个通道：目标 RGB 3、基线残差 RGB 3、候选 alpha 加权 RGB 3、
候选 alpha 和基线 alpha 各 1、有效标志 1、归一化坐标 2、每个采样单元的归一化面积 1。

候选覆盖通道由已知投影参数计算；精确候选渲染只用于教师标签，不进入网络输入。
这些 alpha 通道描述覆盖，尚未精确模拟新的遮挡合成；网格也保留有限采样分辨率。
背景字段在同一相机内缓存，基线场景与候选重建均复用原采集实现。

新网络分别预测共同价值与零均值的正档差异。两个头均复制旧头，区域 CNN 通过零初始化投影接入，
保证两组起始预测一致。共享点编码器仍存在任务耦合，本轮只分开最终预测头。
当前宽度 256 下，原网络为 206,082 个参数，新网络为 396,860 个参数。

## 数据和评价

默认新采 4096 条训练记录与 512 条验证记录，每窗口 64 组，每组沿用原有 4 个训练视角。
采集不更新概率表。冻结编解码器并沿用相同配对噪声下的完整块干预。
整块验证划分沿用原初始化器的 seed，同时核对原在线 plan 的 seed，防止改变验证划分。
初始化网络的 7 个固定归一化张量必须与同源初始化器完全一致；编解码器 ID、场景指纹、RD 参考摘要一并保存。

新模型和原模型在同一份新数据上各训练 2000 次，使用相同初始有效函数、损失、批次和 dropout 种子。
每 250 次记录全部、普通及高差距样本的 regret、命中率、档间 MAE/RMSE、逐视角两分量 MAE/RMSE。
分层阈值仅从训练标签拟合。原有 512 条旧验证集的指标不作为本次新标签上的直接数值对照。

采集约需 72 个窗口，具体取决于实际生成的标签数。日志显示训练／验证数量、教师耗时和累计时间。
采集需要 GPU；后续两组离线训练默认 CPU，可用 `TRAIN_DEVICE=cuda` 留在同一卡上训练。

## 启动和续跑

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
GPU=1
SOURCE="$PWD/output/truck_benefit_stage_one_20261106_180906"
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_footprint_split_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_benefit_footprint_controls.sh \
  "$SOURCE" "$CODEC" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

使用相同 OUT 和参数，增加 `RESUME=1` 可续采或续训。采集每窗口保存进度，训练每次评价保存优化器与采样状态。
已完成的数据会复用，最终产物中断后也可继续生成。采集阶段的同源 `plan.json` 和 `benefit_network.pt` 需要保留；
数据准备完成后，离线训练只读取新 `data` 目录。

只下载根目录 `OUT/review.tar.gz`。回放、static 缓存、初始化网络和末步权重不进入下载包。
