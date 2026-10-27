# 连续证据与结构失真分配实验

独立入口 `gaussian_jscc.structured_evidence_allocation`，检查点 family 为 `structured_evidence_allocation_v1`。保留旧预算、固定 beta 与 evidence-v1 实验，不迁移旧概率历史。复用同一冻结教师，从新的概率表开始；不重训编解码器、不接联合微调。

## 默认实验

- Truck、10 dB、原教师的 0/8/24/48 复符号预算，可靠 XYZ16 与 binary16 分段尺度。
- 2000 次更新，每次 4 个训练视角；校准视角与训练窗口互斥，测试视角不参与更新或选模。
- 单次采样训练、十次规则部署保持不变。beta 显式传入，建议首轮沿用 0.1。
- `D=M+lambda_s*(1-SSIM)`；`L=D/D_ref+beta*C/C_ref`。M 为未截断 RGB MSE，SSIM 使用 [0,1] RGB、11×11 高斯窗口、sigma=1.5、零填充。PSNR/SSIM/LPIPS仍分别记录。

## 固定权重与参照

冻结教师的全 q3、固定校准视角、默认 4 次配对 AWGN 试验得到 M_ref 和 S_ref。默认 `--structure-share 0.2`：

```
lambda_s = 0.2/(1-0.2) * M_ref/S_ref
D_ref = M_ref + lambda_s*S_ref
C_ref = 全 q3 实际通信成本
```

20% 是可调的参考贡献份额，不是直接乘在 SSIM 上的系数。`--lambda-s VALUE` 优先于份额自动换算。参考量接近零时自动标定报错，需明确指定权重。`--distortion mse` 是纯 MSE 对照，不计算结构梯度或结构精确差分。SSIM内部使用float64局部图，与旧float32报告指标有舍入差异；额外保存 `source_objective_mse`，旧 `source_mse` 指标语义不变。

## 连续历史与温和更新

MSE/结构两分量分别保存衰减均值、二阶矩、有效权重、观测次数及最后更新时间。历史分量以固定 D_ref 归一化；通信差价每步重新计算，不在历史里平均。默认半衰期1000步，概率分布漂移仅触发刷新提示，不清空整块历史。

保留反馈按 q1/q2/q3 存三个槽，记录对应正档位相对关闭的代价。三个槽有效权重均≥0.5且不超过一个半衰期才按当前条件概率混合；否则使用本步采样档位刚更新的历史值，缺测槽不补零。所有q0都保留基础门梯度。

一个动作每步只有一份主历史观测：精确值先替换本步近似值。独立近似/精确统计仅用于诊断和风险，不能再次增加主权重。历史波动与精确误差用于减小步长。单点每步动作对之间的log-odds变化默认最多0.25；稳定性系数为0.1到1：`0.1+0.9*support*freshness*context/(1+relative_std+relative_error)`，其中 `support=w/(w+1)`，漂移提示将context降到0.5。该系数仅缩小步长，不改变目标或强制档位占比。

## 决策导向精确反馈

每步默认32个独立反事实：优先24个正档位、8个保留/关闭，不足可互借。各分支轮换三类优先级：RD边界/大概率移动、历史波动/精确误差/上下文刷新、最低精确覆盖，不能整除3的余数按训练步轮换。覆盖按“点＋具体替代动作”的精确次数安排，避免某点反复检查同一档位；保留检查独立使用对应正档位槽的精确次数。每个反事实单独改变一点并重解码完整受影响块，同相机、配对噪声，关闭/打开时重建语义上下文。

MSE使用逐像素 `2*r*delta+delta²`；结构使用float64局部SSIM图差分，复用基准图和目标统计。检查的是混合失真加通信差价后的决策方向。检查记录包括两个分量、真实/近似RD差、方向一致性、错误决策代价及选样组。抽检结果不能外推为全部候选的错误率。

通信训练仍使用边际代理：可靠坐标平均保留成本、分段尺度实际开关变化/平均代理、档位payload；不把压缩放进每个候选。控制与选模使用包含payload、坐标、档位元数据和尺度的真实包成本。可靠侧净2bit/use沿用既有假设，不能将模拟复符号换算为无条件文件MB。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948"
OUT="$PWD/output/truck_structured_evidence_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup bash scripts/test_structured_evidence_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

先预检可在脚本最后加 `--dry-run`；同配置续训使用 `RESUME=1` 和原 OUT。修改目标、权重、上限等参数不能冒充exact resume。旧分配器的检查点不能直接续训到本实验。

## 输出和判断

初始、最终selected/latest保留部署图与档位图（相同端点复用）；中间只有数值，最终保存同教师全q2/q3数字对照。`comparison/structured_controls.csv` 对比真实成本、源PLY/照片指标、两分量和RD；`structured_evidence.json` 保存所选模型的历史统计与固定参考；`charts/structured_evidence.png` 展示分量、有效历史和抽检RD误判趋势。

优先检查有效历史是否真正积累、抽检的错误RD决策是否减少，以及同成本下PSNR/SSIM/LPIPS的变化。全q2的成本/画质关系仍是阶段性参照，不保证此次一定达到。32次精确渲染与双分量反传会增加计算，CPU小型测试只验证正确性；Truck单步速度、峰值显存及实际画质须在服务器测量。

历史保存在CPU；按约71万点估算，单份完整历史快照约424MiB。下载图表分析时可只拉取 `metrics/`、`comparison/`、`charts/`、`evaluation/` 和 `deployment/`，将完整 `checkpoints/` 留在服务器供续训，避免重复下载大历史。
