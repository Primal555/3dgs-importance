# 分解收益评估器

本版更新逐点保留价值、档位增量与决策监督。编解码器保持冻结，档位仍为0/8/24/48，信道10 dB，坐标与尺度开销、真实换挡标签、MSE＋SSIM、概率更新和十次部署规则保持原配置。

## 模型和监督

q0分支预测同一点的两个失真分量：平均保留价值b、q1→q2增量d12、q2→q3增量d23。构造u=[0,d12,d12+d23]，三个档位的保留收益为b+u-mean(u)。因此三档均值始终等于b，相邻差始终等于对应增量；增量允许正负。

共享保留价值的输入去掉目标档位编码及目标／差额预算。每个点的b由相同上下文生成。b和增量分别使用真实三档均值、相邻差监督，归一化尺度只由训练标签拟合。

总损失为0.25×数值回归＋方向损失＋q0三档排序损失。正档位方向相对当前档位判断，q0方向严格使用当前conditional对三个保留方案混合，再与删除比较。解析通信差价在这些决策项中加入；网络输出仍仅表示失真。

方向与排序项使用softplus，权重为1+min(|真实RD差|/分支尺度,10)。0.25、1、1及权重上限属于本次实验的工程设置。保留原一阶辅助输入、25%屏蔽和0.5附加权重。模型不模仿一阶标签。

## 先在已有回放上验证

新模型从随机权重开始，Adam重新初始化；复用已有训练／验证回放、源特征归一化、原学习率和屏蔽策略。新头部尺度只拟合训练标签。不增加真实渲染，不修改源结果。默认CPU即可。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

BENEFIT_RUN="$PWD/output/truck_learned_benefit_gpu1_20261028_170916"
NEW_OUT="$PWD/output/truck_factored_replay_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/test_benefit_diagnostics.sh \
  "$BENEFIT_RUN" "$NEW_OUT" --fresh-factored --updates 1600 \
  >"${NEW_OUT}.log" 2>&1 &
printf 'PID：%s\n输出：%s\n' "$!" "$NEW_OUT"
tail -n 40 -F "${NEW_OUT}.log"
```

1600表示新网络的Adam更新次数；600和1600处都保存同口径数值。与原先200个渲染窗口的含义不同。离线评估不增加独立标签，仍用于检验已有标签分布上的模型和监督。

- `source_baseline.json`：源网络在完全相同回放上的指标。
- `history.jsonl`：新模型各轮训练／验证指标。新旧损失数值不横向排名，比较同一验证集的regret、误判率和策略代价。
- `initial_bias.json`、`final_bias.json`：共同误差与动作分层。
- `summary.json`：源文件不变校验、前后结果和源网络基线。
- `offline_final.pt`：新网络诊断权重；不自动进入在线分配。

报告的keep项新增：固定q1/q2/q3的选择regret、真实最佳档位分布、预测分布、允许真实同代价的命中率，以及共同价值／增量误差。共享门反馈的一阶三档基线明确标注口径。

## 在线预训练与分配

新增检查点family为`learned_benefit_factored_v2`。启动时选择新输出目录，重新训练收益网络；原编解码器无需重训。

```bash
TEACHER_RUN="$PWD/output/successive_mip360_gpu1_20261023_142948"
OUT="$PWD/output/truck_factored_online_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_factored_benefit_allocation.sh \
  "$TEACHER_RUN" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
tail -n 40 -F "${OUT}.log"
```

该命令仍先执行200窗口预训练、前50窗口拟合归一化，概率表固定。确认预测效果后，显式接入2000步分配：

```bash
CUDA_VISIBLE_DEVICES=1 RESUME=1 nohup bash scripts/test_factored_benefit_allocation.sh \
  "$TEACHER_RUN" truck "$OUT" --beta 0.1 --phase allocation \
  >"${OUT}.allocation.log" 2>&1 &
tail -n 40 -F "${OUT}.allocation.log"
```

脚本使用可用显卡编号。原`test_learned_benefit_allocation.sh`默认legacy模型，旧实验恢复不受影响。新结构与旧结构检查点不可混用。
