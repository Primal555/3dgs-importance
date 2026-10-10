# MaskGaussian 预处理三个训练场景

这个入口是原仓库 `prune_finetune.py` 的 PLY 初始化适配，不是 JSCC 训练。
依次处理 Train → Truck → Playroom，同一时刻只训练一个场景。DrJohnson 不参与此轮。
原始 PLY 与原 manifest 保持不变，处理结果写入新目录。

## 默认流程

1. 从 `configs/multiscene_tandt_db.json` 校验三个场景的 pretrained PLY 数值、实际相机记录和引用照片路径。路径校验不等同于完整解码验证所有照片。
2. 原样载入中心坐标、SH、opacity logits、log scales 和 quaternion，不量化、不换通道顺序。
3. 按原 MaskGaussian 将 mask logits 初始化为 `[10, 1]`（几乎全部保留）。PLY 不含 Adam 状态，因此使用新 Adam，不能等同于恢复原始 3DGS checkpoint。
4. 保留原 post-training 目标、学习率参数组及调度：

   `L = (1 − lambda_dssim) L1 + lambda_dssim [1 − SSIM + lambda_mask mean(mask)^2]`

   默认 `lambda_dssim=0.2`、`lambda_mask=0.1`，所以 mask 项的实际系数为 `0.02`。不是 JSCC 的通信预算惩罚。
5. 从绝对 iteration 30000 继续至 35000，即每场景 **5000 次新增更新**；PLY 路线包括最后一次更新后再导出。XYZ 调度仍使用绝对步数，属性按原代码每 400 步衰减，并非重新从 0 步开启一套训练计划。
6. 保留原 post-training 的最终导出方式：`save_ply_default` 单次采样导出标准 PLY；没有新增十次采样部署或周期性物理删点。训练日志中的采样保留数，不是最终导出的精确点数。
7. 标准 3DGS renderer 对导出的 PLY 渲染留出视角，再运行原 `metrics.py`。summary 同时记录点数、文件字节数、PSNR、SSIM、LPIPS；与初始 PLY 对相同留出照片的 PSNR/SSIM 比较。不是对原 PLY 渲染图像做指标。

照片默认缩小 2 倍，并保存在 CPU，训练仍使用 CUDA。需要原 MaskGaussian 和普通 3DGS 的两个 CUDA 光栅化扩展及 `simple_knn`。
PLY 初始化不再创建无用的 SfM Gaussian/KNN 模型；原相机读取器仍要求 sparse 点文件，并可能首次生成 `sparse/0/points3D.ply` 缓存，不会改写 pretrained Gaussian PLY。
LPIPS 使用原 `metrics.py` 的 VGG 权重，首次可能需要下载；该阶段失败时不会发布“完成”的 manifest，已导出的 PLY 不会被删除。

## 启动与检查

服务器 Bash，先激活原环境。这里 GPU 1 仅是示例，按空闲显卡修改。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

OUT="$PWD/output/maskgaussian_three_$(date +%Y%m%d_%H%M%S)"

# 先只校验路径、PLY 属性与命令；不会创建输出，也不调用 CUDA。
CUDA_VISIBLE_DEVICES=1 bash scripts/prune_three_scenes.sh "$OUT" --dry-run

# 后台顺序处理三个场景，断开 SSH 不影响运行。
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/prune_three_scenes.sh "$OUT" \
  > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

总日志显示阶段切换；详细训练进度在每个场景日志中。Train 完成后再进入 Truck：

```bash
tail -n 30 -f "$OUT/train.log"
# 后续按需换成 truck.log / playroom.log
```

支持短测或先只处理 Train：

```bash
CUDA_VISIBLE_DEVICES=1 PRUNE_STEPS=500 bash scripts/prune_three_scenes.sh \
  "$PWD/output/maskgaussian_train_short_$(date +%Y%m%d_%H%M%S)" --scenes train
```

可以通过 `LAMBDA_MASK`、`PRUNE_STEPS`、`RESOLUTION` 或 Python 模块参数改设置。
不会自动寻找 GPU，不会覆盖已有输出目录；遇到缺输入时在启动前失败。

## 产物

- `summary.json`：各场景初始/最终点数、实际 PLY 大小及画质变化。
- `multiscene_pruned.json`：全部成功后才发布的新 manifest，指向处理后的三个 PLY；可传给 `gaussian_jscc.multiscene --manifest ...`。
- `train.log` / `truck.log` / `playroom.log`：独立的训练、渲染、评价日志。
- 每个场景目录：最终 `point_cloud/iteration_35000/point_cloud.ply`、`initialization.json`、`loss.jsonl`、初始 `baseline_metrics.json`、最终 `results.json` / `per_view.json` 与留出视角图像。

不默认保存大量中间 PLY 或 Adam checkpoint。中途中断时会保留已经完成的产物，但不支持精确续训；重新运行需用新目录，可通过 `--scenes` 跳过已处理的场景。
生成的 manifest 使用绝对服务器路径；将结果下载到本地后如需在本机使用，应调整路径。

本机没有 CUDA，单元测试覆盖参数载入和流程调度，不替代服务器真实剪枝、画质与速度验证。
