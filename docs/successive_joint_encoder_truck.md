# 条件增强联合微调 B：语义编码器也参与适应

独立启动脚本 `scripts/test_successive_joint_encoder_truck.sh`，复用入口
`gaussian_jscc.successive_joint --train-semantic-encoder`。不改变原 A 脚本及其默认冻结范围，
不覆盖来源实验或 A 结果；B 必须使用新的输出目录。

## 对照设计

A、B 均直接读取**同一个已完成条件增强信道实验的 final 检查点**，不是在 A 微调后再训 B。
载入全部权重，另建空状态 Adam；不重新训练随机信道，不学习档位分配。

| 设置 | A | B |
|---|---|---|
| 语义编码器 | 冻结，包括训练态缓冲量 | **参与更新** |
| 信道编码器 | 更新 | 更新 |
| 信道解码器，含空间分支 | 更新 | 更新 |
| 语义解码器 | 更新 | 更新 |
| 可训练模块学习率 | 全部 1e-5 | 全部 1e-5 |
| 步数 | 2000 次 Adam 更新 | 2000 次 Adam 更新 |
| 预算及信道 | 累计8/24/48，固定10dB AWGN | 相同 |
| XYZ、属性统计 | 可靠16bit XYZ，统计固定 | 相同 |
| 损失 | 源PLY渲染MSE＋1倍y恢复MSE | **相同形式** |

两组保留相同 seed、q1/q2/q3/随机正档 mixed 顺序及相机选择规则。
源权重、尺度实际字节边界和可靠交付成本均保留。mixed 不是学习后的部署分配。

## 当前 y 的含义与反向路径

每批原始属性和可靠几何重新经过**当前正在更新的语义编码器**，生成 y：

`y = E_semantic(features, XYZ)`

`y_hat = D_channel(AWGN(E_channel(y)))`

`L = MSE(render(D_semantic(y_hat), XYZ), render(source PLY))`

`    + latent_weight * MSE(y_hat, stop_gradient(y))`

默认 `latent_weight=1`，沿用 A 的工程设置。停止梯度仅作用于恢复误差的**目标侧**；
不是把发送端 y 整体 detach。渲染目标和恢复误差都能通过预测链路回到语义编码器。
不使用旧教师缓存的 latent 来约束新编码器，也不新增语义对齐、颜色或感知损失。
真实照片、SSIM、VGG-LPIPS、颜色偏差仅用于评估。

训练继续采用完整场景 replay：无梯度生成场景、计算场景梯度，再用相同噪声
逐块重算并回传到四个模块。额外训练编码器会增加反向计算和峰值显存，不能保证
与 A 的耗时、显存完全相同。属性均值/标准差与可靠 XYZ 不优化，保存时核对统计哈希。

## 防止误读“恢复 MSE 降低”

语义编码器可以改变表示尺度，因此仅看绝对 y MSE 不足以判断是否更可恢复。
每次更新的 `metrics/loss.jsonl` 同时记录：

- `latent_target_mean`、`latent_target_std`、`latent_target_power`：当前 y 的有效点全维统计；
- `latent_mse`、`latent_nmse`（MSE / 当前目标功率）、分档 MSE；
- 语义编码器等各模块的梯度、实际参数更新量、真实学习率；
- 完整渲染 MSE、各视角误差、GPU峰值与耗时。

均值与二阶矩按有效点数汇总，排除填充；标准差由全场景二阶矩减均值平方计算，
不是直接平均各块标准差。日志中的 y 统计对应此次更新**之前**的前向。
日志每25步也打印目标功率、NMSE、编码器更新量。

这些是**诊断记录，不是额外防塌缩约束**。若绝对MSE下降，但y功率大幅缩小、
NMSE不改善且渲染质量无提升，不能算作成功。与A比较，以同测试视角和噪声下的
实际q1/q2/q3 PSNR、SSIM、LPIPS、颜色及模糊结构为主。
clean 也会随语义两端改变，不强制保持不变，不将其视为固定上限。

## 拉取与后台启动

GPU0只是示例，改成实际空闲卡。RUN 保持为已经完成的10000步**纯信道实验**。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

GPU=0
RUN="$PWD/output/truck_successive_gpu0_20261022_091448"
OUT="$PWD/output/truck_successive_jointB_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_successive_joint_encoder_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -F "${OUT}.log"
```

每250步数值验证，每100步检查点；仅初始、最终保存两个代表视角图。
最终32测试视角、AWGN四次配对噪声；clean和无噪声各一次确定性评估。
同时报告源PLY/照片指标和包含XYZ、档位元数据、尺度的实际通信成本。
全部结果位于同一 OUT。

中断后保留原 RUN、OUT，不再生成时间戳：

```bash
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_successive_joint_encoder_truck.sh \
  "$RUN" "$OUT" --resume >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -F "${OUT}.log"
```

精确续训恢复四个模块的Adam动量、RNG、游标与待完成评估；不允许将A输出
按B恢复，也不允许修改步数/学习率后假装精确续训。
`--dry-run --device cpu`只检查来源和计划，不加载PLY、不创建输出、不训练。

## 本机验证边界

定向CPU测试验证真实四模块更新、仅渲染梯度能到达编码器、replay梯度一致性、
实时目标统计、尺度计费及四模块精确续训；渲染/相机使用合成替代。
另核对A默认冻结行为，未运行全仓回归或本地全量场景训练。
这不代表真实Truck画质或CUDA峰值；实际收益交由服务器实验判断。
