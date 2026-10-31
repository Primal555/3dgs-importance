# 空间收益分配：批量执行与断点恢复

沿用 `gaussian_jscc.calibrated_spatial_benefit_allocation`，增加运行参数 `--allocation-batch 64`。默认旧路径保持可用。

## 提速内容

- 候选重建批次由原 16 扩大至 64，收益网络同样按该批次推理。
- 同一相机下一批候选的四档整块高斯合并投影；尾块按真实有效点数打包，恢复原候选和档位顺序。
- 空间输入按等长块向量化。基线统计、3×3 足迹采样和权重按当前窗口的块／相机缓存；候选状态仍逐项对应真实重建。
- 数值有效性检查按批次执行。保留 float32、全部邻居、原始噪声槽位、binary16 尺度帧和真实渲染标签。

每步仍评价原候选池，使用原相机窗口和约 64 组真实标签；收益网络在线更新次数、损失、β、部署规则及概率更新均不变。

## 恢复与数值检查

`allocation_batch` 作为执行配置单独记录在 checkpoint 和训练日志的 `execution` 中，不进入原 `plan.settings`。可以直接恢复已有分配 checkpoint，保留概率表、网络、AdamW、replay 和固定验证集。修改 β 或其他训练配置仍会被原身份校验拒绝。

启动前自动比较当前后端的逐项／批量投影和空间特征，结果写入 `preflight.json`。本地 CPU 测试包含 q0 切换、重复块、尾块、真实标签和镜像更新；CUDA 一致性由服务器预检检查，完整 CUDA 窗口对照测试在 CUDA 可用时运行。不同批次的矩阵运算可能有常规浮点舍入差异，保持数值容差内一致。

## 服务器运行

先停止旧的慢速任务。程序将恢复最近保存的 checkpoint；日志中尚未保存的几步会重算。保留原目录及完整 replay。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_soft_direction_gpu2_20261030_152022"
CUDA_VISIBLE_DEVICES=2 RESUME=1 nohup bash scripts/test_calibrated_spatial_benefit.sh \
  "$CODEC" truck "$OUT" --beta 0.1 --phase allocation --allocation-batch 64 \
  >"${OUT}_allocation_fast64.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}_allocation_fast64.pid"
tail -n 40 -F "${OUT}_allocation_fast64.log"
```

批次可改为 128；它仅改变执行粒度。先看 64 的稳态单步时间和峰值显存，再决定是否扩大。

日志新增 `candidate_projection_calls` 和 `spatial_feature_batches`；原 `evaluated_groups`、`labeled_groups`、候选重建／投影／特征／网络分项计时继续保存。分项为主机计时，总步时在 CUDA 同步后统计。
