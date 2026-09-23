# 5000 步中心坐标对照：自身注意力 Transformer 与历史轻量解码器

入口：`scripts/test_center_self_vs_light.sh`。
这是新的独立对照，不再运行邻居扰动或逐层探针。

## 两组定义

| 目录 | 中心解码器 |
| --- | --- |
| self_only | 上轮只关注自身的 Transformer：保留 V/输出投影、四层前馈与残差、多层读出；XYZ tap 使用 affine |
| historical_light | 历史逐点 MLP＋小窗口/多尺度 Context，带可学习门控，保留其原始 XYZ 读出结构 |

这里“自身注意力”明确指 `center_attention_scope=self`，不是恢复整块自注意力。
历史轻量版仍含局部注意力；本实验不把它改成纯 MLP。
历史来源同 `97ef4cc` XYZ 解码分支。历史分支没有 Transformer 的 tap 归一化；
其配置中的 `center_readout_norm=layernorm` 是未使用字段的兼容默认值，不代表
给历史 XYZ 自身路径新加了 LayerNorm。self_only 则显式使用 affine。

两组使用相同中心编码器结构和随机初始权重、32 维中心 latent、PLY、Morton 分块、
训练/留出划分、采样序列及验证视角。编码器在两组中分别更新，不是冻结编码器对照。
解码器结构及参数量不同，不宣称等参数量；self_only 中 Q/K 无效，保存参数量也
不等于有效参与学习的参数量。结束时自动审计共同初始权重、采样和划分。

- 两组均**随机初始化**，不接旧 checkpoint。
- 各 **5000 步**，仅中心训练；不进入属性或联合渲染阶段，不因质量门槛提前结束。
- 相同世界中心距离损失，固定 Adam **2e-4**、batch32、block256、hidden96、seed42。
- 不裁剪梯度、不模拟通信噪声、不改变档位，不发送额外 XYZ。
- 同一张 GPU 串行运行，两个独立子进程，默认合计 10000 个优化步骤。
- 每 **500 步**保存图像、检查点、PSNR/SSIM、坐标误差；第 0 步也保存验证结果。

## 服务器后台启动

先确认空闲卡，按实际情况更改下例的 `2`：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main || exit 1

OUT="$PWD/output/truck_center_self_vs_light_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=2 STEPS=5000 BLOCKS_PER_BATCH=32 \
  nohup bash scripts/test_center_self_vs_light.sh "$OUT" \
  > "$OUT/console.log" 2>&1 &
echo $! > "$OUT/run.pid"
tail -f "$OUT/console.log"
```

退出 tail 或本地电脑断线不会停止后台任务。`PLY`、`SCENE`、`LR`、`SEED` 可覆盖；
脚本先跑 self_only，再跑 historical_light。根目录必须为新目录；不会覆盖旧结果。

## 同图对照输出

两组完成后在根目录自动汇总：

- **`loss_comparison.png`**：两组在同一坐标轴上的原始 loss（淡线）与每 50 步均值
  （实线）。左图全程，右图仅显示后 80% 步数并单独缩放纵轴，便于看后期差异。
- **`comparison.png`**：拟合块抽样/留出块 RMSE、同图 loss、梯度、中心诊断 PSNR。
- `loss_comparison.csv`：逐步 case、step、loss、梯度、训练步耗时，便于自行作图。
- `comparison.json`：参数量、平均训练步耗时、优化步骤累计耗时、最后 500 步平均
  loss、最后一步与验证最优步指标，以及公平性审计。
- `self_only/`、`historical_light/`：完整日志、检查点、曲线和逐视角图像。

训练步耗时不包括验证、渲染、保存、加载，不能当成总运行时长；总耗时在 console.log。
PSNR 看 `render.center_only.source_psnr`，图像看
`self_only/images/005000/view_00/center_only.png` 和对应 historical_light 路径。
`full.png` 包含未训练属性，不用于本轮评价。既比较相同步数，也比较各自验证最优步，
不能只选择对某一组有利的时间点。

每组自身曲线在训练过程中更新；**根目录的双组对照图在两组均完成后生成**。
一组失败则停止。精确续训可用该组的 `training_state.pt`，配对脚本不自动续跑第二组。
两组完成后可以重新汇总：

```bash
python scripts/compare_center_decoders.py --out "$OUT" --summarize-only
```

本轮依照用户要求不运行本地训练或测试；已补充对应回归用例，执行留待后续。
