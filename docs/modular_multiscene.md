# 三场景模块化 JSCC：随机初始化实验

## 实验约定

Train、Truck、Playroom 共用四个网络（语义编码／解码、信道编码／解码），从随机权重开始。
使用服务器已有的剪枝场景 manifest，不读取历史 codec 权重，也不修改 Truck 单场景入口。
每个场景独立保存 bbox、Morton 分组、训练／验证／测试相机和评估结果。
属性归一化仅由所选训练场景计算，各场景权重相同；DrJohnson 不参与数据统计或训练。
本轮只训练编解码器，不训练逐点档位分配表。

| 阶段 | 每场景真实更新次数 | 训练模块 | 学习率 |
|---|---:|---|---|
| 属性／颜色 | 5000 | 语义编码器、语义解码器 | 1e-4 |
| clean 场景渲染 | 1000 | 语义编码器、语义解码器 | 1e-4 |
| 信道恢复 y | 2000 | 信道编码器、信道解码器 | 1e-4 |
| 通信联合微调 | 2000 | 信道编码器、信道解码器、语义解码器 | 1e-4、1e-4、1e-5 |

逐场景轮换，每次执行一次真实 optimizer.step；三场景共 30000 次更新。
Adam 只在阶段边界重建，不在场景切换时重建。每个场景的独立阶段计数控制颜色与档位轮换。
每四次属性更新改做一次方向颜色监督；通信阶段按 q1/q2/q3/mixed 轮换，不与颜色计数绑定。

结构沿用两端 PointNet 式上下文、48 维 y、8/24/48 复符号、full-prefix 渐进信道解码、
pointwise 去噪与 plain 邻居修正，16bit delta-zlib 可靠 XYZ。
属性初始化沿用局部响应损失／独立颜色步骤；渲染采用 MSE+(1/3)(1-SSIM)，联合阶段另加 y-MSE（权重 1）。
固定 AWGN 10dB；渲染反向采用 replay。无梯度裁剪，不引入新学习率调度。

## 记录与续训要求

每 500 个每场景更新统一验证并保存一次，记录真实全局 step 和各场景阶段 step。
分别对原 PLY 渲染和照片记录 PSNR、SSIM、LPIPS，另记录颜色误差、latent MSE/NMSE、梯度、更新量和显存峰值。
输出包含各场景结果与等场景权重的宏平均，不能让相机或点数较多的场景支配汇总。
仅初始／最终验证保存代表图；过程只存数值，最终使用全部测试相机、不以训练视角替代。
checkpoint 保留四个网络、共享统计、Adam、随机状态、场景轮换游标和相机／PLY 指纹。
准确续训须使用原输出目录和相同配置；恢复时归档超出 checkpoint 的日志。
CPU 合成测试只验证流程与梯度实现，不作为真实 CUDA 画质或收敛证据。

## 服务器启动

在服务器 Bash 中执行（先确认所选显卡空余；下面以 0 号卡为例）：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

MANIFEST="$PWD/output/maskgaussian_three_20261010_190401/multiscene_pruned.json"
OUT="$PWD/output/modular_multiscene_ssim_gpu0_$(date +%Y%m%d_%H%M%S)"

# 先确认路径及配置，不加载 checkpoint，不创建输出目录。
python -m gaussian_jscc.modular_multiscene --manifest "$MANIFEST" --out "$OUT" --dry-run

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/train_modular_multiscene_ssim.sh \
  "$MANIFEST" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
printf '输出：%s\n日志：%s\n' "$OUT" "${OUT}.log"
tail -n 50 -f "${OUT}.log"
```

只有确认 `--dry-run` 成功再执行后台启动；服务器已有剪枝 manifest 的路径如有变化，应显式替换。
启动脚本不复制 Truck 权重、不重跑 MaskGaussian，也不回退使用原始未剪枝 PLY。
`tail` 中 Ctrl+C 只退出看日志，不会终止后台训练。SSH 断开后任务继续，但本地代理隧道断开可能影响首次 LPIPS 权重下载，建议提前缓存。
需要先检查 LPIPS 时运行 `python -m gaussian_jscc.perceptual_metrics --device cpu`；缺少权重会明确报错，不会伪造指标。

## 准确续训

先确认旧训练进程已退出。使用原来的 OUT，不要产生新时间戳，并保留第一次使用的参数：

```bash
OUT="/data/home/zhangyueheng/projects/3dgs-importance/output/你的本轮目录"
MANIFEST="$PWD/output/maskgaussian_three_20261010_190401/multiscene_pruned.json"
CUDA_VISIBLE_DEVICES=0 RESUME=1 nohup bash scripts/train_modular_multiscene_ssim.sh \
  "$MANIFEST" "$OUT" >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 50 -f "${OUT}.log"
```

输出结构：

- `training.json`：随机初始化、共享结构、各阶段学习率和完整配置。
- `metrics/loss.jsonl`：按实际更新顺序记录 `step / scene / scene_step / phase`。
- `scenes/<name>/metrics/{validation,test}.jsonl`：分场景 clean、无噪声、10dB 各前缀指标。
- `scenes/<name>/images/{initial,final}`：只保留两个端点的代表图。
- `metrics/macro_{validation,test}.jsonl`：三个场景等权宏平均；PSNR 是 dB 指标的平均，不是池化像素 MSE 转 PSNR。
- `charts/`：分场景阶段损失与场景／宏平均画质曲线；`metrics/phase_summary.json` 记录耗时。
- `checkpoints/{initial,latest,end_<phase>,final}/codec.pt`：共享模型及准确续训状态。
- `complete.json`：全部阶段与最终测试完成的标志；仅进程退出不等于训练完成。

本轮评估 mixed 是固定随机的正档位组合，不是学习后的资源分配；逐场景分配表留待编解码器训练完成后另开实验。
DrJohnson 保留为以后零样本跨场景评估，不由本入口加载或计算统计。
