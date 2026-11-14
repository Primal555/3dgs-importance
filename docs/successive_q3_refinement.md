# q3增强段单独优化实验

本实验检验：保持q2恢复能力与通信预算，现有q3新增的24个复符号还能带来多少渲染收益。入口为 `gaussian_jscc.successive_q3_refinement`。

## 初始化与训练

- 从 `successive_mip360_gpu1_20261023_142948/training/checkpoints/final/codec.pt` 初始化全部权重。
- 仅更新 `channel_encoder.stages.2`、`channel_decoder.stages.2`。语义编解码器、前两级信道模块、邻居修复、属性统计全部冻结；冻结模块继续参与前向与输入梯度传播。
- 全部有效点使用q3，8／24／48前缀、16 bit XYZ、块尺度和原始Morton分块保持一致。
- 固定10 dB AWGN，每次更新采样新的噪声。语义特征缓存到CPU，每批按需拼接，保留原始块和补齐掩码；渲染反传沿用原来的重放实现。
- 损失为源PLY渲染RGB MSE＋latent MSE，默认latent权重1。使用新的Adam，恒定学习率1e-5，跨场景共享优化器。梯度检查沿用原联合训练的有限性检查，不增加梯度截断。
- 默认沿用9个训练场景，每场景250次实际更新，共2250步。每步训练视角、照片缩放、训练／校准／测试划分全部继承源实验。当前主干每步2个训练视角、4个校准视角，最多32个测试视角。
- Room仅做留出评测。每场景50步记录一次固定校准视角的q3指标，25步保存一次可恢复状态。初始与最终端点评估q2、q3的无噪声与10 dB结果，AWGN各4次配对试验。
- 固定使用末步权重，不按校准或测试结果选择checkpoint。初始／最终对比报告源场景与照片的PSNR、SSIM、LPIPS、MSE，以及latent MSE／NMSE。

q2的冻结参数、端点画质及latent误差会核验。记录q3相对初始模型的变化，以及q2→q3每增加一百万次复信道使用带来的源图像MSE下降。缓存的正确性通过与完整语义编码前向的损失和参数梯度比较进行验证。

## 通信计费与交付

评测复用 `DigitalDeploymentCost`、实际包头和 `side_information_accounting.py`。属性按实际复符号数计费；XYZ、完整元数据、块尺度分别按文件成帧。协议为LDPC(4096,6144)、16QAM、48字节文件头、每码字4字节CRC、508字节有效载荷、1536次复信道使用，含补齐，无ARQ。历史2 bit/use仅用于兼容场景构造，最终报告使用数字文件协议。

最终默认在Truck的q2与q3上各执行4次实际数字侧信息交付，保留CRC／SHA256、整次成功率、失败试验成本和成功条件下画质。多场景质量矩阵标注为可靠侧信息假设，实际链路结果单列在 `actual_delivery/`。共享编解码器预装。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git switch experiment/feedforward-student
git pull --ff-only origin experiment/feedforward-student
conda activate maskgs

GPU=3
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/successive_q3_refine_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" RESUME=0 \
nohup bash scripts/test_successive_q3_refinement.sh "$RUN" "$OUT" \
  --steps-per-scene 250 --lr 1e-5 \
  >"${OUT}.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"${OUT}.pid"
printf 'PID：%s\n实验目录：%s\n' "$PID" "$OUT"
tail -n 40 -F "${OUT}.log"
```

`DIGITAL_PYTHON`默认指向 `$HOME/miniforge3/envs/gs_digital/bin/python`。可用 `--dry-run` 查看完整计划。调整显卡或输出目录后，其他参数可沿用。

中断后使用相同RUN、OUT和参数，将 `RESUME=0` 改成 `RESUME=1`，并把日志重定向改为 `>>"${OUT}.log"`。恢复会载入q3专用优化器、场景进度与随机数状态；已完成的评测和数字交付逐项复用。

## 结果与判断

下载 `OUT/review.tar.gz` 即可检查；其中包含计划、损失、校准曲线数值、端点质量、Truck图像与实际链路记录，不包含checkpoint、缓存或临时传输载荷。

```bash
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
r = json.loads((out/'review_summary.json').read_text())
print('q2保持一致：', r['q2_unchanged'])
print('场景        q3初始源PSNR  q3最终源PSNR  照片PSNR变化  latent MSE变化')
for row in r['q3_changes']:
    if row['path'] != 'awgn':
        continue
    d = row['change']
    print(f"{row['scene']:10s} {row['q3_initial_source_psnr']:12.4f} "
          f"{row['q3_final_source_psnr']:12.4f} {d['photo_psnr']:+12.4f} {d['latent_mse']:+.6g}")
print('实际数字交付：', json.dumps(r['actual_delivery'], ensure_ascii=False, indent=2))
PY
```

同预算下q3渲染提高，说明现有增强段还有优化空间。latent下降而画质持平，提示继续关注误差与渲染敏感性的对应关系。额外续训有效本身不能区分训练时长与共享参数更新的影响。本轮只测试增强段可改善的幅度；后续可据此讨论32符号等中间档位。
