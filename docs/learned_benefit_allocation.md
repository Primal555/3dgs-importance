# 可学习的换挡收益分配实验

入口：`gaussian_jscc.learned_benefit_allocation`。检查点family：`learned_benefit_allocation_v1`。

收益网络直接预测两个失真分量：`Delta_MSE/D_ref`、`lambda_s*Delta_structure/D_ref`。真实换挡渲染提供标签，一阶信息独立输入；25%训练样本组额外接受屏蔽一阶的监督。通信差价通过解析公式组合。

## 默认配置

- Truck、冻结四模块教师、10 dB、0/8/24/48、可靠XYZ16和binary16分段尺度。
- `D=MSE+lambda_s*(1-SSIM)`，`L=D/D_ref+beta*C/C_ref`，beta显式传入，首轮沿用0.1。
- 每窗口4个训练视角、32次独立反事实：20个正档位动作和4组完整q0评价。每个q0组渲染其他3种状态，得到三个“保留相对删除”的标签；全体点每窗口都有保留预测。
- 前200个窗口固定概率表；前50个窗口采集训练标签并拟合固定归一化，第51个窗口起训练网络。默认每窗口4次Adam、学习率1e-4、128组、梯度范数上限1。
- 80% Morton块训练、20%收益验证。同块全部动作保持同一划分，验证标签不进入训练、归一化和概率代价覆盖。
- 单次采样训练、十次规则部署；不接编解码器联合微调。

自定义预算至少保留4次反事实评价，以覆盖两类监督；回放容量至少为2组。最初标定窗口的训练样本独立暂存，完成归一化后释放，回放抽样不会丢失标定所需的某类标签。

## 第一步：拉取并训练收益评估网络

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

RUN="$PWD/output/successive_mip360_gpu1_20261023_142948"
OUT="$PWD/output/truck_learned_benefit_gpu1_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_learned_benefit_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
printf '输出：%s\n日志：%s\n' "$OUT" "${OUT}.log"
tail -n 40 -F "${OUT}.log"
```

`RUN`需要指向当前服务器实际存在的完整教师实验目录。可使用 `--manifest` 重新定位同一PLY与相机路径；模型、点数和场景指纹仍需匹配。脚本末尾加 `--dry-run` 可进行只读预检。

预训练完成后生成 `pretrain_complete.json` 与 `comparison/benefit_pretrain.json`，进程正常结束。该阶段的逐点概率保持不变。

## 第二步：检查评估能力，再接入分配

先查看：

```bash
python -m json.tool "$OUT/comparison/benefit_pretrain.json" | less
```

重点比较 `validation.positive.stages` 和 `validation.keep.stages` 下三种预测的 `regret_mean`、`sign_error_fraction`，并查看纠错／改错数量与窗口配对区间。`both_branches_lower_regret`表示同一验证样本上两类动作的平均错误代价均下降；还应结合高代价误判与跨窗口表现判断。

`keep`下的分量误差与RD指标都按同一条件概率混合，每个点的一组三标签计为一个样本。完整组另外报告三个正档位的选择regret。

确认后，在同一输出目录继续：

```bash
CUDA_VISIBLE_DEVICES=1 RESUME=1 nohup bash scripts/test_learned_benefit_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 --phase allocation \
  >"${OUT}.allocation.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.allocation.pid"
tail -n 40 -F "${OUT}.allocation.log"
```

该命令执行2000步在线分配，继续收集真实标签和更新收益网络。若预训练尚未完成，入口会报错。阶段B为显式启动，程序不会在阶段A结束后自动进入。

## 中断恢复

预训练恢复使用原 `RUN`、原 `OUT` 与相同参数，增加 `RESUME=1`。分配恢复还需保留 `--phase allocation`。`--phase`可以从预训练切换为分配；beta、训练预算、屏蔽率、教师与其他数值配置保持一致。

快照包含收益网络、Adam、概率表、固定归一化、训练／验证回放、覆盖调度、参考量和随机状态。网络更新次数与分配次数分别记录，`step`为两阶段累计窗口数。

分配图表使用独立的 `allocation_step` 横轴；分配单步耗时统计不混入预训练窗口。

## 输出与指标

- `metrics/loss.jsonl`：采样占比、各阶段计时、网络训练损失、梯度和显存峰值。
- `metrics/benefit_actions.jsonl`：标签取得前的一阶／网络／屏蔽预测、真实标签及动作来源。
- `metrics/benefit_prequential.jsonl`：当步标签进入训练前的预测准确性。
- `metrics/benefit_validation.jsonl`：当前网络对未训练块的回放验证。
- `comparison/benefit_stages.csv`：两类动作分别比较方向错误、错误代价、分量误差与相关性。
- `comparison/benefit_timing.json`：两个阶段的分项median/p95与显存峰值。
- `comparison/endpoints.csv`、`charts/`：完成分配后的真实部署通信量和画质对照。
- `evaluation/initial`、最终 `evaluation/selected`／`latest`：效果图、档位分布图；中间校准只保存数值。
- `deployment/selected_tiers.npy`及概率表：原PLY行顺序的实际部署结果。

监督中的正失真差值表示动作损伤画质；负值表示改善。q0采用当前条件档位概率混合三个保留差值。训练通信项为边际代理，校准选模与最终报告使用含坐标、档位元数据和尺度的真实包成本。

常规下载可只取 `comparison/`、`metrics/`、`charts/`、`evaluation/`、`deployment/`。完整 `checkpoints/` 包含回放，建议留在服务器供继续训练。
