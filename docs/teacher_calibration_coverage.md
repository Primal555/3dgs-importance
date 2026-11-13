# 教师组：扩大固定校准覆盖

入口为 `scripts/test_teacher_calibration32.sh CODEC_RUN REFERENCE_TEACHER NEW_OUT`。

本轮从随机抽样所用的初始概率表重新训练，参考上一组 `truck_teacher_views64_shuffle_gpu2_20261112_145947`。每步仍使用 64 个训练视角，固定校准集扩展到 32 个视角。β=0.5，正档位限幅 2，保留分支限幅 0.25，候选批量 128，最多 2000 步；每 100 步校准，沿用已有的学习率调整和收尾规则。

Truck 原有 251 个视角中，32 个测试视角和 4 个编解码器验证视角维持原名单。从其余 215 个视角按列表等间隔抽取 32 个固定校准视角，剩余 183 个作为分配训练池，每步轮换取 64 个。校准扩展因此也改变了训练池。`training.json` 中的 `allocation_train_view_names`、`calibration_view_names` 和 `test_view_names` 保存实际名单。

从参考教师的 `training.json` 固定读取 D_ref、lambda_s 和 C_ref，并核对源 PLY、相机记录、编解码器权重、β、噪声设置及完整数字文件账本。启动日志分别打印训练池、窗口、校准、测试和参考尺度。新的校准视角用于选模和收尾，原来的 4 个参考视角仅说明目标尺度的来源。目标尺度保存在 `objective_reference`，其 `normalization_source` 记录来源；`plan.json` 保存引用文件的 SHA256 和完整参考值。

本组参考值为 D_ref=0.001671516249358023、lambda_s=0.00629856382701326、C_ref=43258896 次复信道使用。实际全 q3 文件账本会重新核对，避免沿用错误成本。属性和三路侧信息统一使用 `digital_side_file_v1`。本轮记录计费与属性 AWGN 结果，实际数字侧信息传输使用已有的 `digital_side_delivery` 入口单独验证。

最终按原 32 个测试视角、4 次噪声试验报告选中版本、末步和全 q2/q3；测试结果不参与选模。图片仍仅保存 2 个代表视角。校准增加的开销集中在每 100 步评价时。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git fetch origin
git switch experiment/feedforward-student
git pull --ff-only origin experiment/feedforward-student
conda activate maskgs

GPU=2
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
REFERENCE="$PWD/output/truck_teacher_views64_shuffle_gpu2_20261112_145947"
OUT="$PWD/output/truck_teacher_views64_cal32_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 nohup bash scripts/test_teacher_calibration32.sh \
  "$RUN" "$REFERENCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

检查结果下载 `OUT/review.tar.gz`。恢复使用相同 RUN、REFERENCE、OUT 及 `RESUME=1`。参考教师只需保留 `training.json` 和 `complete.json`，本轮没有读取其概率表或训练检查点。

## 当前学生短训的视角设置

`test_student_shared_controls.sh` 从 `truck_student_views32_prep64_gpu3_20261112_122850` 的 best 学生检查点继续 200 步。继承 64 个准备视角的特征缓存、211 个训练视角的调度器，每步使用 32 个训练视角。共享真实检查和配对更新诊断均使用当前训练窗口。每 50 步用原 32 个测试视角、4 次噪声试验报告结果，以第 200 步为固定终点。源长训记录中的 4 个校准视角在本次短训中不承担选模或停止功能。
