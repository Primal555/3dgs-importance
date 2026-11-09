# 收益网络：交叉项、能量与结构监督

目标是改善普通样本和高代价样本的档间差距学习。保留覆盖区域 CNN、配对空间输入、公共／档间分离头，增加能量输出和独立真实标签；本实验仅采集标签并训练收益网络。

## 精确标签

对每个视角，在同一背景下渲染 I0、I1、I2、I3。I0 为 q0 候选条件；参考图像为原任务的源 PLY 渲染。令 delta=Ik-I0，e0=I0-Iref：

- A=mean(2*e0*delta)/d_ref，有正有负。
- B=mean(delta^2)/d_ref，非负。
- S=lambda_s*(SSIM(I0,Iref)-SSIM(Ik,Iref))/d_ref。
- 最终两分量标签仍为 [A+B,S]，通信成本另外相加。

均值包含全图 RGB 元素；局部积分只累加有效核像素，分母仍为全图 H*W（RGB已在核内平均）。边界和 SSIM halo 沿用精确教师。同一候选渲染同时生成三项标签，未增加候选渲染次数。输入特征只使用候选属性和基线观测，真实干预图像仅用于教师标签。

旧标签无法唯一恢复 A、B，必须新采集。旧 two-component 数据会被入口拒绝。新数据保留旧 values，并记录 response_terms=[A,B,S] 和 schema，读入时验证二者一致。

## 网络与目标

原最终两分量价值输出 m、s 保留。新增非负能量头预测 b；交叉项 a=m-b，结构项为 s，最终价值为 [a+b,s]。这是一种可逆参数化，保留原始价值函数初始化。B 初始为训练集能量 RMS 的1%，A对应调整。能量为零的相同候选通过既有 identical mask 精确置零。

主损失为既有逐视角公共值回归与强化总 RD 档间差距回归之和，系数固定1。三项辅助损失分别回归 A、B、S，使用仅由训练数据拟合的固定 RMS；GradNorm 在三项辅助损失内调节，辅助系数合计固定1。排序保持训练标签拟合的温度，KL乘 tau/s，申请系数固定1；每个有效批次以全部参数限制排序梯度范数不超过数值合成梯度。之后采用原范数裁剪10与 AdamW。

数值主路径不会参与 GradNorm 权重竞争。实际任务为 value/cross/energy/structure/ranking，日志同时提供申请与应用权重、分量误差、分层 regret、逐视角／窗口档间 RMSE、完整排序正确率、梯度裁剪次数。最终组合价值始终以原 RD 单位报告。

gradnorm 字段为控制器原始诊断，其权重合计3；auxiliary_weighted_gradient_norms 与 auxiliary_target_gradient_norms 已换成实际合计1的辅助权重口径，可与主任务梯度比较。

## 数据与预算

- 来源：已完成 footprint_split 目录。读取其 data/initial.pt、metadata.json、complete.json，并使用元数据定位原冻结编解码器与 benefit_network.pt 所属管线。
- 重新采集4096训练组、512验证组；整块留出机制沿用原管线，训练和验证点不能重合，校准视角不用于标签采集。
- 编解码器冻结，参考尺度、已有输入归一化与最终价值输出尺度沿用原来源。能量 RMS、分量尺度、排序温度与高代价阈值只从训练标签拟合。
- 初始化沿用原 data/initial.pt，覆盖区域分支按原构造流程初始化；不加载此前最优覆盖区域网络。能量头复用原档间头的隐藏层参数，最后一层新建。
- 默认 batch64、micro8、64000次样本访问，即1000次更新。分层采样保持原总体比例，并做采样概率校正。保存末步结果。
- 采集使用 GPU；固定数据训练默认 CPU。五项梯度共用一次前向，并微批次累计，所需反向次数多于此前三任务版本。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
SOURCE="$PWD/output/truck_footprint_split_gpu3_20261107_000336"
GPU=3
OUT="$PWD/output/truck_response_terms_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_benefit_response_terms.sh \
  "$SOURCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

脚本默认完成采集后自动进入 CPU 训练。可加 `--phase collect` 仅采集；随后使用同一 OUT、`RESUME=1` 和 `--phase train` 训练。`--codec`、`--pipeline-root` 用于原目录迁移后显式定位来源。

恢复使用同一参数与 `RESUME=1`。训练阶段恢复网络、AdamW、辅助 GradNorm、采样顺序及累计梯度统计；中断前未提交的指标尾记录会被裁去。

检查包仅为 OUT/review.tar.gz，包含设置、指标、分层预测和三项真实／预测值。完整恢复需保留 OUT/data 的 replay 与初始化、OUT/training/latest.pt。不要只下载 review 作为恢复备份。
