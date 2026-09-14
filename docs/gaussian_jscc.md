# 面向 3DGS 的局部上下文 JSCC 编解码器

> 架构更新：新训练默认使用 [Geometry-first 编解码器](geometry_first_jscc.md)。
> 本页后文的 seed→Context→更新位置、组间平均 loss 和旧 checkpoint 续训说明属于历史架构；
> 当前结构、物理损失、符号分路和运行命令以升级说明为准。旧模型仍可独立评估。

实现位于 `gaussian_jscc/`。它借鉴 ROI-JSCC 的变长 latent 前缀选择、打包和补零机制，并以 PyTorch 重新实现 FCGS 式多尺度空间网格聚合。它不是 ROI-JSCC 或 FCGS 的直接复现，不能加载它们的预训练权重；原始 MaskGaussian 训练流程没有被修改。

## 当前技术路线

```text
3DGS PLY + 外部分配 q_i ∈ {0,1,2,3}
                    │
        发送端按 Morton 顺序排列并分块
                    │
        q=0 删除；q=1/2/3 保留
                    │
      完整 xyz + opacity/scale/rotation/SH
                    │
       发送端多尺度空间网格上下文
                    │
            q / SNR 条件编码器
                    │
   最大长度 latent → 按 q 取 0/8/16/32 个复符号
                    │
          功率归一化 → 模拟信道
                    │
        接收端按完整 q 序列拆分符号
                    │
       从带噪 latent 并行预测初始 xyz
                    │
       按预测 xyz 建立接收端空间网格
                    │
          联合精修 xyz 与其他属性
                    │
              标准 3DGS PLY
```

逐点坐标不再放进可靠元数据。完整归一化 xyz 与所有其他 Gaussian 属性一起经过 JSCC。接收端先预测初始位置，再使用这些位置对接收特征做空间聚合；初始位置有单独辅助损失。低 SNR 下初始位置错误可能改变邻居关系，这是需要由训练和实验检验的风险。

元数据仍包含场景全局包围盒 `lower/span`，共 6 个浮点数，用于把网络输出的 `[0,1]^3` 坐标恢复到 PLY 坐标系。这是场景级常量，不随 Gaussian 数量线性增加。

关键约定：

- `q=0` 不传该 Gaussian 的属性，不参与块内上下文；它的 2-bit 零档标记仍需发送。
- `q=1/2/3` 默认发送 `8/16/32` 个复信道符号。档位只控制 latent 长度，不乘透明度。
- 一个复符号由两个实数表示。接收端为未发送的 latent 位置补零，补零不计传输开销。
- 符号联合承担内容表达与抗噪保护，没有人为划分“信息符号”和“保护符号”。
- 编码器使用真实位置构建发送端上下文；解码器只使用由接收 latent 预测的位置，不读取发送端逐点坐标或干净特征。
- 每块默认包含 Morton 顺序中的 4096 个原始 Gaussian。块间没有自回归依赖，也没有跨块特征交换。
- 网格是有界局部稠密网格，默认分辨率为 4、8、16，并带 xy/xz/yz 三平面；它不是 FCGS 的稀疏 CUDA / 熵编码实现。
- 改变档位后应重新编码，因为编码网络本身也以 q 为条件。

## Morton 排序与 FCGS 上下文的职责

二者有关联，但不是同一机制。

`Morton ordering` 将量化后的 xyz 位交错成一个一维键，使空间接近的 Gaussian 大概率在序列中也接近。本实现只在发送端用它确定稳定顺序和定长分块，从而限制每次网格计算的点数并提高块内空间局部性。量化整数只用于排序，不发送给接收端；`--morton-bits` 控制排序键精度，不代表坐标通信精度。

`FCGS-inspired grid context` 在每个块内部把 Gaussian 特征按位置和插值权重 splat 到多尺度 3D 网格 / 三平面，再从相应位置 query 聚合特征。它负责真正的邻域信息交换和冗余建模。

```text
Morton：决定“哪些点放进同一块、以什么顺序发送”
FCGS 式网格：决定“同一块内的特征怎样按空间位置聚合”
```

Morton 本身不提取特征、不判断重要性、不分配码率；网格上下文本身也不决定跨块边界。空间近邻可能被切到相邻块，因此当前分块存在边界损失。

## 与路线一和路线二的接口

公共编解码器支持外部分配图，也支持 [路线二四档 mask 联合训练](route2_joint_jscc.md)。

`--rate-map tiers.npy` 必须是长度 `[N]` 的整数数组，值为 0、1、2、3，并与输入 PLY 的原始行顺序对应。发送端做 Morton 排序时同步重排 q；完整排序后 q 序列以每项 2 bit 发送。接收输出为保留点的 Morton 顺序，不再是原 PLY 顺序。

- 路线一的分配网络可输出该数组。
- `train-route2` 联合更新每个 Gaussian 的四档 logits 和公共编解码器，并保存匹配的 `route2.pt` / `codec.pt`。渲染、信道和期望资源损失共同影响选择。
- `transmit/evaluate --allocation route2.pt` 直接使用学习后的档位，`export-route2` 可以导出原始 PLY 行顺序的 `tiers.npy` / `probabilities.npy`。
- 原有 `train` 命令仍只训练公共编解码器。联合训练另用直通 Gumbel 近似，前向是真实硬档位，反向是有偏连续梯度近似。
- 不提供 `--rate-map` 时，codec 训练会随机采样混合 / 统一档位，以覆盖不同符号长度。这只是训练公共 codec，并不是用手工分档替代路线一网络。
- MaskGaussian 的普通 `point_cloud.ply` 不含完整存在概率；代码不会用 opacity 冒充重要性。

## 数据包和开销

```text
packet/
  metadata.bin   # zlib 压缩：完整 2-bit q 序列、数量、包围盒、配置、模型校验值
  received.npy   # 已经过模拟信道的复符号，float32 [K,2]
  stats.json     # 负载、元数据和总信道使用次数
```

完整 q 序列既给出删除标记，也使接收端知道每个 Gaussian 应从变长流读取多少符号。`decode` 只需要 packet 和共享 `codec.pt`，不需要源 PLY、逐点坐标、图像或单独 q 文件。

元数据仍采用可靠到达假设；CRC 只能检测文件损坏，不是 FEC。默认用理想复 AWGN 容量估算其资源：

```text
metadata uses = ceil(metadata bits / log2(1 + 10^(SNR/10)))
total uses = JSCC payload complex symbols + metadata uses
```

也可用 `--metadata-code-rate 0.5 --metadata-modulation-bits 2` 按指定码率和 QPSK 位数计费，但代码仍未模拟真实包头编码、译码失败、丢包或重传。`received.npy` 是仿真载体，其磁盘字节数不能当通信码率。

AWGN 使用单位平均复符号能量和复噪声方差 `10^(-SNR/10)`。Rayleigh 模式假设完美接收 CSI 并逐符号 MMSE 均衡。一套权重在 `--snr-range` 中随机采样 SNR，区间外性能没有保证。

共享权重默认预先部署，不计入每次场景传输，`stats.json` 会另报模型张量体积。如果对每个新场景重新训练后还要发送权重，必须另计模型分发成本。

## 服务器训练和测试

在原 MaskGaussian 环境运行，不需要安装 ROI-JSCC / FCGS 的自定义算子。渲染需要本仓库已有的 `diff_gaussian_rasterization`。所有输出目录要求事先不存在，防止覆盖结果。

```bash
cd /root/MaskGaussian-main/MaskGaussian-main
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PROJECT="$PWD"
PLY="$PROJECT/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply"
SCENE="$PROJECT/data/tandt_db/tandt/truck"
CODEC_OUT="$PROJECT/output/truck_gaussian_jscc_full_xyz"

python -m unittest discover -s tests -p test_gaussian_jscc.py -v
```

不要使用 `input.ply`，它只是稀疏初始化点云。训练完整 xyz 和属性：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m gaussian_jscc train \
  --ply "$PLY" \
  --out "$CODEC_OUT" \
  --steps 20000 \
  --render-steps 2000 \
  --source "$SCENE" \
  --resolution 2 \
  --snr-range 0 20 \
  --rates 0 8 16 32 \
  --block-size 4096 \
  --morton-bits 16 \
  --seed-position-weight 0.2 \
  --channel awgn
```

前 20000 步训练局部块的完整归一化位置与属性。后 2000 步恢复完整场景并采样训练视角，优化渲染 L1、SSIM、属性损失以及初始位置辅助损失。迭代数是训练预算，不是已验证的收敛标准。完整场景渲染比属性块训练慢很多。

该修改改变了 checkpoint 结构和数据包格式，旧版含粗坐标的 `codec.pt/metadata.bin` 不兼容，必须重新训练并重新发送。

单次发送与独立解码：

```bash
CUDA_VISIBLE_DEVICES=0 python -m gaussian_jscc transmit \
  --ply "$PLY" \
  --checkpoint "$CODEC_OUT/codec.pt" \
  --uniform-tier 2 \
  --snr 10 \
  --out "$PROJECT/output/truck_jscc_packet_full_xyz_snr10"

CUDA_VISIBLE_DEVICES=0 python -m gaussian_jscc decode \
  --checkpoint "$CODEC_OUT/codec.pt" \
  --packet "$PROJECT/output/truck_jscc_packet_full_xyz_snr10" \
  --out "$PROJECT/output/truck_jscc_received_full_xyz_snr10.ply"
```

实际分档时，将 `--uniform-tier 2` 换为 `--rate-map /path/tiers.npy`。

使用同一权重测试多个 SNR / 档位：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m gaussian_jscc evaluate \
  --ply "$PLY" \
  --checkpoint "$CODEC_OUT/codec.pt" \
  --source "$SCENE" \
  --resolution 2 \
  --snrs 0 5 10 15 20 \
  --tiers 1 2 3 \
  --trials 3 \
  --save-images \
  --out "$PROJECT/output/truck_jscc_full_xyz_evaluation"
```

`results.json` 包含信道开销、位置 RMSE、属性 MSE和渲染指标。视图从左到右为 GT、传输前 PLY 渲染、通信恢复后渲染。若使用真实 q 文件，用 `--rate-map` 替代 `--tiers`。去掉 `--source` 可只测试属性收发，不产生 PSNR / SSIM。

## 独立编解码器损失基准

`benchmark-codec` 固定所有 Gaussian 为同一个正档位，不使用 `route2.pt`、q0 或重要性分配。`none` 信道衡量有限 latent 与编解码器造成的重建损失，`awgn` / `rayleigh` 衡量加入信道后的总损失。两组都按相同档位和 SNR 条件运行，因此可以区分网络瓶颈与信道额外退化。

若要从头训练一个不含随机删点增强的纯 codec，使用普通 `train` 命令并设置 `--attribute-drop 0`；如果只是检查联合模型中的通信部分，也可以直接把 `train-route2` 产生的 `codec.pt` 交给下面的基准命令。

```bash
CUDA_VISIBLE_DEVICES=0 python -u benchmark_codec.py \
  --ply "$PLY" --checkpoint "$CODEC" --source "$SCENE" \
  --tiers 1 2 3 --snrs 0 5 10 15 20 \
  --channels none awgn --trials 3 --resolution 2 \
  --save-images --lpips \
  --out "$PWD/output/truck_codec_benchmark"
```

先做接口和参数误差检查时可去掉 `--source --save-images --lpips`，并使用 `--device cpu --snrs 10 --trials 1`。基准报告位置的世界坐标 RMSE / 包围盒对角线归一化 RMSE、激活后透明度 MAE、log-scale RMSE、四元数夹角误差、DC / 高阶 SH RMSE、实际复符号数与可靠元数据开销。提供相机时还报告通信恢复渲染相对输入 PLY 渲染的 PSNR、SSIM、L1 和 LPIPS，以及两者各自相对 GT 的质量。

默认完成真实打包和独立接收后删除临时 `received.npy`，防止大场景多档位、多 SNR 测试占用数 GB。使用 `--keep-packets` 才会把每轮包保存在对应目录；`--save-ply` 可保存每轮恢复场景。输出的 `charts/` 会包含 codec 渲染保真度、分属性误差、质量—SNR、信道开销和率失真图。
`none` 是确定性通路，每个档位 / SNR 只运行一次；`--trials` 只重复带噪信道。

自动测试覆盖前缀长度、功率和 AWGN、网格邻居梯度、初始位置与编解码梯度、激活重计算、PLY 往返、2-bit 档位打包、零档 / 空块、包完整性、独立解码和 CLI 全流程。CUDA 光栅器反传测试只在具有对应环境时运行。通过这些测试不等于已证明论文级视觉性能或 4090 显存目标。

借鉴源与许可证见 `gaussian_jscc/NOTICE.md`。
