# 收益网络与纠偏联合适配

入口：`scripts/test_neural_correction_allocation.sh`。

默认从随机权重训练索引修复后的收益网络 700 窗口，然后在线分配 2000 步。编解码器冻结；q0 使用原有全点一阶反馈、历史和 0.25 更新上限。正档位使用收益网络，更新上限 1.0。初始化、RD 目标、候选预算、训练单次采样和部署十次规则均沿用主线。

## 每步执行

1. 采样当前场景，渲染四个训练视角并取得 q0 全点反馈。
2. 批量评估正档位候选，共用解码、配对噪声和相机缓存；最多 64 组额外真实四状态监督保持原预算。
3. 执行原有最多 32 次真实纠偏，其中正档位通常为 24 次。训练空间块上的已知动作差值加入 replay；固定验证空间块只用于检查。
4. 使用当前网络及其当前版本的预测统计计算本步反馈，真实测量替换对应动作。
5. 从完整四档标签和已知动作标签混合训练收益网络，共 64 次优化。下一步使用更新后的网络。

部分标签只监督实际测量的两个档位之差，包括归一化 MSE、加权结构项和带通信差价的软方向目标。其余档位不构造数值标签。replay 仍有 8192 组上限；极小容量配置下，每步动作标签最多占容量的四分之一，以保留完整监督。

## 历史与上下文

正档位预测均值记录网络更新版本。点再次被评价时，若网络版本改变，清理该点旧预测的均值、方差和有效样本数。概率表、候选访问次数、真实 replay 和 q0 历史继续保留。同版本预测仍按原规则累积。在线训练时，多视角的长期积累由概率表和真实 replay 承担，旧网络输出不会永久累积为稳定证据。

残差库只保存 replay 引用、动作和上下文，不复制完整输入。复用要求相机集合相同、块内每个点的实际档位相同，再按目标点及整块可见性描述的变化连续衰减。不同上下文可分别保留，受 replay 容量约束。读取残差时，当前网络重新预测保存的输入，用真实标签减去当前预测；每次网络更新后清空预测缓存。未匹配的条件使用当前网络预测，真实样本仍可通过 replay 训练改善网络。四视角集合重复较少时，外部残差复用也会较少。

补建少量真实纠偏点的输入时复用当前相机捕获与解码缓存，不增加真实渲染。仅批量传回紧凑的可见性描述。残差重算分批执行，最多 64 组一批，避免一次加载所有历史输入。

## 运行

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

GPU=3
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_neural_corrections_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_neural_correction_allocation.sh \
  "$CODEC" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
echo "$OUT"
tail -n 40 -f "${OUT}.log"
```

使用新的 OUT。已有旧分配状态不迁移。恢复时指定相同参数、相同 OUT，并设置 `RESUME=1`。冻结编解码器、紧凑收益网络、allocation checkpoint 和 `allocation/neural_replay` 均需保留；预训练阶段恢复另需 pretrain replay。

## 检查

- `allocation/metrics/loss.jsonl`：`neural_exact_training_groups`、补建输入耗时、`neural_history_invalidated_rows`、网络更新次数及实际候选调用次数。
- `allocation/metrics/network_validation.jsonl`：每次校准间隔保存固定验证集上的网络指标，观察学习与遗忘。
- `allocation/metrics/validation.jsonl`：部署校准和最终测试的 PSNR／SSIM／LPIPS、通信代价与 RD。
- `review.tar.gz`：供下载检查，排除 checkpoint 和 replay。

固定验证集使用预训练时保留的上下文。部署 RD 用于判断整个分配系统是否改善。

## 启动前口径审计（2026-11-04）

策略版本为 `current_network_context_residual_v2`。每步在候选调度之前同步网络版本，过时正档位方差及误差不参与新版本的候选与纠偏优先级；q0 核验优先级使用自身误差，密集反馈、历史及更新公式保持原设计。旧预测均值仍在点被重新访问时惰性清理，避免每步清空全场景统计。

完整四档监督和动作差值监督使用同一个 `tie_fraction` 近似平局区间：小于阈值的真实 RD 差值保留数值回归，不增加方向损失。日志中的 `neural_full_state_regression_loss`、`neural_observed_gap_regression_loss` 分别表示两类样本对 minibatch 回归损失的加权贡献；`neural_absolute_loss` 保留为两者之和。

历史统计按本步分配实际使用的反馈版本筛选，`positive_feedback_revision` 记录该版本。`network_updates` 是本步在线训练后的版本，默认比反馈版本多 64 次优化。结果报告分别记录 `positive_network_feedback_revision` 和 `checkpoint_network_updates`，避免把最近一次分配反馈误读为训练后权重已经具备的历史。累计候选访问及 observation 统计仍覆盖整个实验。

标签统一为 `[ΔMSE/D_ref, lambda_s·Δstructure/D_ref]`，网络只预测失真，通信差价在决策中单独加入一次。q0 锚定的完整监督作为辅助目标保留；网络验证中的 q0 指标明确标记为辅助项。实际删点仍由密集 q0 分支负责。训练 RD 为单次采样失真与期望成本代理，部署 RD 使用十次规则输出和完整计费，两者在日志中分别注明。
