# 当前主线：校准一阶档位分配

2026-11-10 恢复以 `c1162c2` 的校准一阶分配器为研究主线。实现入口为
`gaussian_jscc.structured_evidence_allocation`，检查点 family 为
`calibrated_structured_allocation_v2`。统一启动入口为
`scripts/train_mainline_allocation.sh`，直接调用原有脚本并转发全部参数。

## 版本依据

以下两组完整实验使用同一算法与默认配置，区别为通信惩罚 beta：

- `output/truck_calibrated_evidence_gpu1_20261028_010635`：beta=0.1。
- `output/truck_calibrated_evidence_gpu2_20261028_010738`：beta=0.5。

冻结编解码器沿用
`output/successive_mip360_gpu1_20261023_142948/training`。每次新实验重新初始化
保留概率与正档位概率；编解码器和源高斯属性固定。

## 算法与配置

- q0：每轮全点存在性梯度反馈，结合通信成本、历史证据和抽样精确纠偏。
- q1–q3：每块两轮候选调度；真实重建目标点换挡后的受影响块，将属性变化与共享渲染梯度点积，估计失真差。
- 历史：沿用独立动作残差、有效样本量和置信度；半衰期 1000 步。
- 更新：keep_lr=0.1，tier_lr=0.4，两分支 log-odds 上限均为 0.25。
- 目标：`D=MSE+lambda_s*(1-SSIM)`，`L=D/D_ref+beta*C/C_ref`。全 q3 校准得到固定参照与 20% 结构参考份额；beta 显式指定。
- 交付：10 dB AWGN，q0/q1/q2/q3 对应 0/8/24/48 个属性复符号；可靠 XYZ16、binary16 尺度及元数据计入总成本。
- 训练：2000 次概率更新，每次 4 个训练视角、32 次精确干预；单次采样训练、十次规则部署。
- 收尾：每 250 步在独立校准视角评价，按最低校准 RD 选择 `best_rd`；最终分别报告 selected、latest、全 q2、全 q3 的测试结果。
- 指标：相对源 PLY 与相对照片的 PSNR、SSIM、LPIPS，以及总复信道使用与 RD。

本轮复用既有算法路径。检查点安全读取、图表输出和对照侧信息统计等维护修复继续使用。
收益网络与熵分档实验保留独立入口，供后续研究。
完整公式、历史统计与纠偏规则见 [结构化证据分配说明](structured_evidence_allocation.md)。

## 服务器运行

β=0.5 的提速与收尾实验使用独立入口，见 [运行说明](fast_first_order_allocation.md)。下方入口保留历史执行流程。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

GPU=3
BETA=0.5
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_first_order_beta${BETA}_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 nohup bash scripts/train_mainline_allocation.sh \
  "$RUN" truck "$OUT" --beta "$BETA" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

GPU 按服务器空闲情况设置。`BETA=0.1` 对应另一组历史条件。
仅检查配置时，在前台运行上述脚本并添加 `--dry-run`；续训时使用原 OUT、相同参数和 `RESUME=1`。

分析结果下载 `complete.json`、`plan.json`、`comparison/`、`metrics/`、`charts/`、
`evaluation/` 和 `deployment/`。续训另需 `checkpoints/latest/state.pt` 及原编解码器、场景输入。

## 本轮验证

主线默认参数与历史 beta=0.5 实验的 36 项 settings 一致。
CPU 小块对照覆盖混合 MSE+SSIM、纯 MSE、开启/关闭精确纠偏及全部关闭场景：
当前路径与 `c1162c2` 原 kernel 的保留反馈、正档位反馈、候选索引及随机数状态一致。
既有定向测试 21 项通过，1 项 CUDA 用例因本地环境跳过；新入口通过 Bash 语法、
实际 dry-run、默认参数、RESUME 传递和缺少 beta 时退出的检查。
全量 GPU 的画质和耗时沿用历史实验记录，本轮验证范围为入口恢复与算法一致性。
