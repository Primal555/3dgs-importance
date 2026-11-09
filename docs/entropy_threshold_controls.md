# 独立双阈值与 β 对照

本实验读取已完成的熵模型实验，为冻结的信息量分数独立校准两个边界：

\[
q_i=\begin{cases}1,&I_i\le t_1\\2,&t_1<I_i\le t_2\\3,&I_i>t_2\end{cases}
\]

每点信息量按原始 PLY 行序使用。同分点保持同一档位，全部输入点保留。编解码器、信息量及 8／24／48 复符号档位固定，优化器更新数为 0。

默认用 0、0.1、0.25、0.5、0.75、0.9、1 分位点构造 28 组有序边界，覆盖三个固定档位。再增加接近全 q2 通信开销的候选，并保留原实验的 η 候选。按档位图哈希去重，同一图只测一次。阈值只在这个有限候选集合中选择。

预算候选先按 q1 节省的属性符号数估计 q3 数量，再以完整实际开销校验；超出预算时降低 q3 数量并重新计费。坐标、尺度和元数据均包含在 C 中。元数据长度随档位图变化，最终以实测完整成本为准。

## 目标与评价

继承源实验的 D_ref、C_ref、λ_s、相机和配对噪声：

\[
D=\operatorname{MSE}+\lambda_s(1-\operatorname{SSIM}),\qquad
L_\beta=D/D_{\rm ref}+\beta C/C_{\rm ref}.
\]

同一批真实编码、加噪、解码及渲染结果分别计算 β=0.1、0.5 的目标。在校准视角选好之后，分别评价对应档位图在测试视角的 PSNR、SSIM、LPIPS。另列在 C≤C_q2 候选中校准失真最低的一组，供检查质量达到全 q2 且资源更低的阶段目标。全 q2 本身也参与选择。

本轮与旧校准一阶 β=0.1、0.5 组的参考值一致：D_ref=0.001671516249358023，C_ref=46193136，λ_s=0.00629856382701326。β=0.5 对同一成本变化施加 β=0.1 的 5 倍权重。当前固定保留点集合、信息量排序及阈值搜索决定了可选方案，最终删点率和档位分布不能沿用旧分配器的预期。

## 服务器运行

```bash
conda activate maskgs
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
SOURCE=/填写上一轮完整熵实验目录
GPU=1
OUT="$PWD/output/truck_entropy_thresholds_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_entropy_threshold_controls.sh \
  "$SOURCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

运行需要源实验的完整指标、`cache/information_scores.pt`，以及原编解码器 checkpoint、PLY、图像和相机。新实验不会改写源目录。迁移数据时可指定 `--codec-run` 和 `--manifest`，仍验证原模型与数据身份。中断后使用相同 OUT 并设置 `RESUME=1`，已完成的候选评价会复用。

`comparison/summary.json` 中的 `selections` 分别列出 `beta_0p1`、`beta_0p5`、`q2_budget`；`against_q2` 包括成本节省、源渲染／照片 PSNR 差及 SSIM／LPIPS 差。`selection.json` 保存全部校准候选。测试结果不参与选阈值。

下载 `OUT/review.tar.gz` 即可分析。包内包含指标和少量最终图像，档位图单独保存在 `deployment/`。
