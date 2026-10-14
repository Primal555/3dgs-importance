# 信道互逆能力与特征误差定位对照

本测试从 `truck_progressive_budget82448_gpu0_20261014_165028` 的
`progressive/checkpoints/final/codec.pt` 出发。不是加载 clean 教师后重新初始化，
也不是无损直接传 y。原 8/24/48 复符号预算、48 维 y、网络深度、可靠 16bit XYZ
均保持不变，语义编码器和解码器始终冻结。

## 两个训练分支

- `noiseless`：关闭 AWGN，仍完整经过信道编码、功率归一化、前缀选择和信道解码。
- `awgn10`：继续使用 10 dB AWGN。

两端 SNR 条件输入都固定 10 dB；不能用极大 SNR 数值替代真正的无噪声开关。
每组默认额外训练 2000 个真实更新，源模型 4000 → 总步数 6000。
继承全部模型权重以及 Adam 的一阶/二阶动量和 step 计数，默认保留源学习率
1e-4。可用 `--lr` 同时覆盖两组，但本轮对照建议保持默认。

为避免 AWGN 抽样消耗随机数后改变点批次或 mixed 掩码，新续训流程对数据抽样使用
独立、按总更新步数确定的随机流。两个分支逐步使用相同批次、档位和原始 y。
源 RNG 在每个分支起点恢复，噪声使用训练随机流。
这是保留优化器的受控续训，不是原旧程序原封不动多跑 2000 步：数据抽样的配对方式
变为上述新规则，两个分支一致。不会调整网络结构或改写源目录的任何文件。

## 误差检查

初始/最终各固定均匀抽取 32768 个原场景点；语义 y 必须先在原处理批次内完整
提取，之后才能抽点，避免改变 PointNet 上下文。padding 排除，q0 不参与。
对 q1/q2/q3 的同一组点执行无噪声预测及 8 次独立 AWGN 预测；同一 trial 的符号槽
噪声在档位间配对。诊断不更新参数，也不消耗训练 RNG。

每一维报告原 y 的均值/标准差、无噪声偏差/MSE、噪声下偏差/MSE、误差标准差、
总误差贡献，以及逐点均值误差和噪声波动的拆分：

`finite-trial MSE = mean_over_points(mean_over_trials(error)^2) + within-point trial variance`。

另报所有点共同误差均值的平方、噪声相对无噪声输出的平均移动，以及逐点均值误差
的跨维协方差。逐点均值误差不等于所有点共同偏差；8 次有限 trial 的样本均值仍含
估计噪声，不宣称是精确的总体偏差或因果证明。方差使用除以 T 的总体式计算，
使上述有限样本恒等式成立，而不是除以 T-1 的无偏估计。

同时记录各符号段归一化前后功率和径向幅度响应的诊断值；这些是统计，不改变
归一化或发送功率。特征维度没有预先赋予 DC/SH 等语义，不能直接把某一维称为颜色。

## 启动与输出

激活 maskgs 并选择空闲 GPU 后：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test_channel_inverse_truck.sh SOURCE_RUN NEW_OUT
```

脚本接受 suite 根目录或其 `progressive/` 子目录，在一张卡上顺序跑两组。
默认只训练特征 MSE，不进行渲染更新。每 500 步评估同四个验证视角下的
clean / finite-noiseless / 10 dB q1-q3/mixed，记录 Source PLY、照片 PSNR/SSIM 和颜色
偏差；只保存初始/最终图像，不额外重复整套多 SNR 测试。

主要输出：

- `plan.json`：源文件 SHA256、实际预算、继承状态和训练条件。
- `noiseless.log` / `awgn10.log`：两组独立日志。
- 各组 `diagnostics/initial`、`diagnostics/final`：`latent_errors.json` 和逐维 CSV。
- 各组 `metrics/loss.jsonl`、`metrics/validation.jsonl`：真实更新和画质曲线。
- `comparison/summary.json`、`psnr_curves.png`、`latent_errors.png`：横向对比。

原 checkpoint 和元数据保留不动。新增输出中只写 initial、latest、end、final
checkpoint，不保存每个评估轮次的历史模型。下载分析时可不下载 `checkpoints/`；
续训所需文件留在服务器即可。diagnostic JSON 的 MSE 是点样本统计，validation JSON
的 latent MSE 是完整场景统计，不能把两者当成同一数值。

中断后对相同输出、相同参数追加 `--resume-suite`；精确恢复该分支的 Adam/RNG 和
进度，已完成分支跳过。不能把这套带不同信道开关的 checkpoint 交给旧的
`modular_experiment --resume` 静默续训，应继续使用本脚本。

## 判读

无噪声分支只改善无噪声：说明原 10 dB 训练折衷可能妨碍精确互逆。
两组都改善：额外优化量仍有价值。
无噪声分支也停滞：更应优先定位结构或表示限制。
噪声分支均值误差大：关注系统性恢复偏差；trial 波动大：关注噪声稳健性。
每组都检查 q1/q2/q3，不能仅靠训练总 loss 排名。
