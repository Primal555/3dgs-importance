# 两组属性/颜色受控实验

入口为 `scripts/test_appearance_controls.sh`，默认只测试 Truck。两组实验分别回答不同问题，不改变 q0 学习、不训练分配器，也不增加新的损失权重或增强层结构。

## 1. 颜色恢复：输出头不足，还是需要整个属性编解码器适配？

从同一份已有实验的 `checkpoints/end_render/codec.pt` 开始，使用原场景渲染 RGB MSE，固定 q2、保留全部点，各训练 2000 步：

- `color_heads`：只更新 DC 和 SH 输出头。编码器、符号路径、解码主干、形状与透明度输出头，以及全部归一化统计保持不变。
- `all`：更新完整属性编解码器。坐标仍通过原有可靠量化路径交付，不学习 XYZ。

两组使用相同批次、噪声、训练相机和新的 Adam 状态；这是受控微调，不是恢复原优化器状态。程序会逐项核验冻结参数/缓冲区完全未变。额外评估未经微调的同一 checkpoint，作为起点对照。

训练前默认进行一次 q2 属性替换渲染，分别替换 DC、SH、DC+SH、形状、透明度等，帮助解释颜色误差是否与其他属性的补偿有关。这只是诊断，不参与训练。`--skip-oracle` 可跳过替换渲染，但不会跳过场景身份检查。

固定 q2 微调不保证 q1/q3 也改善；最终同时测试三个前缀，以观察是否有其他档位退化。颜色组使用已有 checkpoint 是为了隔离改动，并非要求未来正式训练依赖历史失败结果。

## 2. 属性预训练方向：球面随机方向还是实际训练相机方向？

两组均从相同随机权重、相同单场景属性统计开始，各运行 2000 步属性预训练 + 1000 步场景渲染训练：

- `random`：保留当前随机球面方向的局部 Gaussian 响应监督。
- `camera`：改为每个 Gaussian 到真实训练相机的逐点观察方向。方向为 `normalize(point - camera_center)`，与当前渲染器的 SH 计算一致。

除了预训练方向，编解码结构、目标、前缀循环、丢点比例、批次、噪声、渲染视角、更新次数和学习率均一致。方向采样有独立随机数发生器，避免挤占通信噪声等随机序列。

注意：这里仍是廉价的**局部正交响应监督**，不是把预训练变成整场景相机渲染。它没有增加可见性筛选、遮挡、透视投影或像素面积加权。真实相机方向只改变观察方向分布。只使用训练相机；训练阶段验证相机和最终测试相机不参与方向监督。默认步数用于发现变化，不代表已经收敛。

## 公平性、数据和指标

- 两组实验内部比较，不把颜色组的已有 checkpoint 初始化与方向组的随机初始化横向排名。
- 启动前核验 PLY 点数与内容指纹；训练前核验原实验的训练/验证相机名单，防止迁移路径指向另一个场景或拆分。
- 不读取或学习档位表；最终测试忽略已有档位表，只比较固定 q1/q2/q3。
- 最终使用各组 **final** checkpoint，而非从测试视角选最佳模型。默认使用全部独立测试相机、两次噪声试验。
- 同时记录相对原 PLY 渲染和真实照片的 PSNR、SSIM、MSE；颜色指标包含 RGB 有符号偏差、RGB MAE，以及 R-G/G-B 差值误差 RMSE。它们不是感知色差 Delta E，也不进行后处理颜色校正。
- 不仅依据 RGB 平均偏差判断颜色改善，因为不同区域的偏差可能抵消。结合差值误差、局部区域及图像观察。
- 每 250 步记录数值；训练图像仅保存初始/最终两轮，每轮最多两个代表视角。最终测试也只保存两个代表视角的图像，但指标默认覆盖全部测试相机。训练 mixed 图片是随机布局诊断，不是学习分配后的部署结果。
- 固定数字链路效率的通信等效换算沿用原实验设置，并显式标注假设；不把复符号的 float 文件大小当作通信比特数。

可选预先指定 ROI，JSON 结构为场景 → 相机 `image_name` → 区域名 → `[x0,y0,x1,y1]`，坐标归一化到 `[0,1]`。例如：

```json
{"truck": {"your_actual_image_name": {"red_detail": [0.70, 0.10, 0.82, 0.28]}}}
```

示例不对应你的实际红框，应替换成真实视角名及区域。通过 `COLOR_ROIS=/absolute/path/rois.json` 指定，各组共用。ROI 不进入 loss；对应逐视角记录见 `per_view.csv` 和测试 `validation.jsonl`。

## 服务器启动

先激活环境、拉取代码，并确认下面的 GPU 是空闲的。GPU 1/2 只是示例，可以修改为实际空闲卡号。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
RUN="$PWD/output/multiscene_pruned_aligned_gpu0_20261010_194134"
```

颜色组使用一张卡，内部依次完成两个微调组和评估：

```bash
OUT="$PWD/output/truck_color_controls_gpu1_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=1 SUITE=color SCENES=truck \
  nohup bash scripts/test_appearance_controls.sh "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

另开终端，方向组使用另一张卡：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
RUN="$PWD/output/multiscene_pruned_aligned_gpu0_20261010_194134"
OUT="$PWD/output/truck_direction_controls_gpu2_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 SUITE=directions SCENES=truck \
  nohup bash scripts/test_appearance_controls.sh "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

`Ctrl+C` 只退出 `tail`，不会停止后台任务；关闭本机终端也不会因 SSH 断开而中止 `nohup` 任务。

若只想使用一张 GPU，设置 `SUITE=all`，两个实验会依次运行，所有结果保存在一个共同目录的子目录中。可通过 `COLOR_STEPS`、`BOOTSTRAP_STEPS`、`RENDER_STEPS`、`VALIDATE_EVERY`、`LR` 覆盖预算。默认学习率 `1e-4`。

如果场景数据换了存放位置，设置 `MANIFEST=/absolute/path/multiscene_pruned.json`。它必须指向同一份剪枝后 PLY 和同一套相机数据，不能换成未剪枝场景。

启动前可先用同一命令去掉 `nohup`/重定向、在末尾加 `--dry-run`，检查路径及将执行的命令；该模式不会创建输出目录或启动训练。

## 检查/下载哪些结果

每个实验的共同目录包含：

```text
plan.json / progress.json / complete.json
comparison.csv / comparison.json
charts/                         # 两组共图的曲线、loss、最终测试对比
color_oracle/truck/              # 仅颜色实验：替换渲染诊断
color_initial_test/             # 仅颜色实验：微调起点的测试指标
color/color_heads/ / color/all/  # 仅颜色实验
directions/random/ / directions/camera/  # 仅方向实验
```

各训练组的 `loss.jsonl` 及 `scenes/truck/validation.jsonl` 记录完整数值历史，`scenes/truck/validation_images/` 仅保留初始/最终图像，`test/` 为最终固定前缀测试。

先下载共同目录的 `comparison.*`、`charts/`、各组数值日志和初始/最终图像即可；无需把所有 checkpoint 下载回来。`complete.json` 表示所有子任务及配对/冻结核验完成；异常退出不会写总完成标记。

本机 CPU 测试覆盖真实编解码器、通信噪声、反向传播、冻结核验及完整实验调度；渲染边界使用可微合成测试函数。它验证代码流程，不证明真实场景颜色或画质已改善。真实 CUDA 渲染及效果需要服务器结果确认。
