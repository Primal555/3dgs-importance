# 3DGS 数字交付对照实验

本目录提供 MesonGS 与 MaskGaussian 的数字交付对照：官方源码、压缩适配、真实 LDPC/QAM 文件传输、独立接收恢复、统一图像评估及服务器启动脚本。两个官方源码目录均为 Git 子模块，版本由本项目的 Git 提交固定。

## 论文与固定版本

| 方法 | 正式发表 | 官方仓库 | 固定提交 |
| --- | --- | --- | --- |
| MesonGS | ECCV 2024 | <https://github.com/ShuzhaoXie/MesonGS> | `e491557a39e480105ae944eb2481d141808efbef` |
| MaskGaussian | CVPR 2025 | <https://github.com/kaikai23/MaskGaussian> | `2f581d03d9e493babf7ab31e68c40f92e591f478` |

论文原文：

- MesonGS：<https://shuzhaoxie.github.io/data/24-eccv-mesongs.pdf>。
- MaskGaussian：<https://openaccess.thecvf.com/content/CVPR2025/html/Liu_MaskGaussian_Adaptive_3D_Gaussian_Representation_from_Probabilistic_Masks_CVPR_2025_paper.html>。

## 当前准备状态

- 已拉取两个官方仓库，保持上游源码不变。
- 已拉取 MaskGaussian 的普通光栅器、掩码光栅器、simple-knn 及所需 GLM 子模块。
- `digital_link.py` 已实现 Sionna 2.2.0 的实际 5G NR LDPC 编解码、公开固定比特加扰、Gray QAM、AWGN 和软解调，按实际编码后长度计费。
- `mesongs_adapter.py` 已实现官方 CUDA 编码组件调用、压缩包独立解码和标准 PLY 导出。
- `evaluate.py` 复用主项目的相机、光栅器、PSNR、SSIM 和 LPIPS/VGG 0.1。
- `run.py` 串联完整流程，支持断点继续，输出紧凑的 `review.tar.gz`。
- 本机为 `torch 2.9.0+cpu`。真实 CPU LDPC/QAM 文件传输、官方序列化压缩包的解码及流程测试已验证；完整场景的 CUDA 编码、光栅化和 LPIPS 权重自检需要在服务器执行。

## 服务器安装与启动

在主项目根目录、现有 `maskgs` 环境中执行。安装脚本会获取两个子模块，并分别建立 `gs_digital` 和 `gs_mesongs` 环境。

```bash
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs
bash scripts/setup_comparison_baselines.sh all
```

也可按阶段执行：`sources` 拉取源码，`digital` 安装数字链路，`mesongs` 安装 MesonGS。MesonGS 默认安装独立 CUDA 12.1 工具链；已有相容工具链时可设置 `MESONGS_CUDA_HOME=/path/to/cuda`。默认编译架构为 RTX 4090 的 `8.9`，其他 GPU 通过 `TORCH_CUDA_ARCH_LIST` 指定。

默认先运行 Truck、10 dB、4 次信道试验，MesonGS 使用 `c1` 配置且不做渲染微调：

```bash
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
GPU=3
OUT="$PWD/output/digital_baselines_truck_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_digital_baselines.sh \
  "$RUN" "$OUT" --scene truck >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

`RUN` 只用于读取该编解码器实际使用的源 PLY、照片路径、分辨率、图片子目录、背景和保存的测试相机名单；本实验不加载或更新该编解码器权重。保存有名单时严格沿用其名称和顺序；显式设置 `--views` 才在该名单内取子集。运行前可用相同命令追加 `--dry-run` 查看计划。

可选参数：

- `--methods maskgaussian`：只跑剪枝 PLY 的数字交付。
- `--meson-configs c1 c3`：增加第二个官方压缩工作点。
- `--meson-iterations 8000`：增加官方渲染微调步骤，默认 0。
- `--snrs 6 8 10 12`：扫描多个 Es/N0；每个点分别执行真实传输。
- `--trials 1`：先进行一次场景规模的流程检查。
- `--batch-size 64`：增大数字链路的码字批次，默认 32。
- `--no-lpips`：显式关闭 LPIPS；默认启用并先检查权重。
- `--keep-intermediates`：额外保存成功接收的码流与 PLY，默认清理这些大中间文件。

如果独立环境放在其他目录，设置 `DIGITAL_PYTHON`、`MESONGS_PYTHON` 和 `EVAL_PYTHON` 为对应 Python 的绝对路径。

恢复中断实验时沿用原始 OUT 和全部参数：

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/test_digital_baselines.sh \
  "$RUN" "$OUT" --scene truck >>"${OUT}.log" 2>&1 &
```

也支持显式输入，便于使用其他已有剪枝结果：

```bash
python -m comparison_baselines.run \
  --ply /path/to/pruned/point_cloud.ply --source /path/to/photos/truck \
  --scene truck --out /path/to/new/output \
  --digital-python /path/to/gs_digital/bin/python \
  --mesongs-python /path/to/gs_mesongs/bin/python
```

## 结果、计费和运行开销

- `summary.json`、`comparison.csv`：每种方法、SNR 下的交付成功率、真实总信道使用次数，以及成功交付条件下的画质均值。
- `results/<method>/snr_<value>/trial_<index>/link.json`：逐次传输的 BER、块错误、CRC 错误、实际平均发送功率、开销、种子、版本和耗时。
- `results/.../evaluation/metrics.json`：统一相机及协议下，相对照片、源 PLY 的逐视角和平均指标。每个方法最多保存两张对照图。
- `artifacts/<method>/sender.json`、`encode.json`：输入哈希、压缩配置和版本。压缩文件保留在服务器供恢复使用。
- `review.tar.gz`：收集指标、配置和代表图，下载此文件即可分析结果。

默认数字链路为 `k=4096`、`n=6144`、16QAM、20 次 BP 迭代和复 AWGN 10 dB Es/N0。每个信息块内含 CRC32；文件长度、SHA256、分组填充均实际发送。总信道使用为 `codewords * 6144 / 4`，没有按每符号净 2 bit 估算。公开固定的计数器加扰独立于噪声种子，接收端按同一协议解扰，无需传输额外幅度。

每次 trial 都实际编码、加噪、解调和解码。完整接收包通过校验后，其解压结果相同，因此后续成功 trial 复用该包已经验证的渲染指标；复用来源写入 `evaluation_report`。传输失败记录为 `failed`，不赋予伪造的画质值。`quality_conditioning` 明确标注为 `successful_file_delivery`，应同时查看成功率。

实际通信成本覆盖完整数字文件。`file_channel_cost` 与真实传输共用文件头、CRC、补齐和调制编码的计费公式。

### JSCC 侧信息统一账本

JSCC 的属性按实际发送复符号计费，可靠 XYZ、档位元数据、binary16 块尺度各封装为一个数字文件，采用与基线相同的 LDPC／QAM 配置。默认每码块含 508 字节净载荷，发送 1536 个复符号。每路文件增加 48 字节头，分别补齐，因此侧信息成本为：

```text
K_side = sum_j ceil((B_j + 48) / 508) * 1536
K_total = payload_complex_symbols + K_side
```

`scale_stream_bytes` 已包含尺度流内部帧头；外层数字文件封装在此基础上另计。共享权重预置，trial 不乘进单次交付开销。数字基线的压缩包已经包含其解码侧信息，整包按同一函数计费。

新运行的预算／RD 分配评估在成本记录中增加 `comparison_accounting`，其中的 `total_channel_uses` 用于统一通信成本对比。原顶层 `total_channel_uses`、训练成本代理和选模 RD 继续沿用该实验的既定口径，保证历史续训一致。

已有实验无需 GPU 或重训即可生成统一的 `summary.json`、`comparison.csv`：

```bash
python -m comparison_baselines.side_information_accounting \
  --allocation output/truck_neural_corrections_gpu1_20261104_115640 \
  --baseline output/digital_baselines_truck_gpu3_20261105_094813 maskgaussian \
  --baseline output/mesongs_raw_truck_gpu1_20261105_123433 mesongs_c1 \
  --out output/communication_accounting
```

`--allocation` 和每个 `--baseline` 也接受对应的 `review.tar.gz`。JSCC 目录需包含 `review_summary.json`、`allocation/training.json` 与最终评估指标。工具核对场景、相机名单及顺序、分辨率、背景、图片子目录、SNR 和数字链路成本；同时保留各组源 PLY 身份及数字组成功率。

JSCC 侧信息当前按可靠交付假设计费，未执行三路文件的信道误码仿真，字段为 `side_link_simulated=false`。数字基线保留实际文件交付成功率及成功条件下的画质。统一账本只重算通信成本，原始画质、历史 RD 和选中的档位保持不变。

## 两组实验的作用

### MaskGaussian 剪枝场景＋无损打包＋数字链路

MaskGaussian 根据可学习的存在概率剪枝，并通过掩码光栅化为未选中的基元继续提供反馈。官方输出为标准 3DGS PLY。

首先复用当前 JSCC 实际输入的剪枝后 PLY，记录其路径、文件哈希和点数。该 PLY 经无损打包、LDPC 编码、QAM 调制和信道传输后，在接收端恢复并渲染。打包需要包含接收端恢复场景所需的全部内容，按实际传输字节计费。

这组比较回答：对于相同的源场景，完整数字交付和学习式属性交付在画质、通信开销及噪声鲁棒性上有何差异。已有剪枝数据可直接用于这组交付实验，无须再次剪枝。

MaskGaussian 输出仍包含 float32 属性。它能够作为剪枝后完整数字交付的参照；属性压缩方面的比较同时使用 MesonGS。数字组添加的无损打包和通信链路应在结果中明确标注。

### MesonGS 压缩场景＋数字链路

MesonGS 接收已有 PLY，执行重要度剪枝、属性变换、分块量化和向量量化，保存可解码的压缩包。它同时提供属性压缩能力，作为更紧凑的数字交付对照。

先使用与 JSCC 相同的剪枝后 PLY 进行压缩，再将完整 `bins.zip` 接入相同数字链路。分别记录官方无微调流程和后续微调流程；配置、输入文件与优化时间随结果保存。

最终图表的主指标为相同测试照片下的 PSNR、SSIM、LPIPS 与总复信道使用次数。相对源 PLY 渲染的指标另行记录。

## 服务器获取固定源码

在主项目根目录执行：

```bash
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

git submodule update --init --depth 1 -- \
  comparison_baselines/MesonGS \
  comparison_baselines/MaskGaussian

git -C comparison_baselines/MaskGaussian \
  -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --recursive --depth 1 -- \
  submodules/diff-gaussian-rasterization \
  submodules/mask-diff-gaussian-rasterization \
  submodules/simple-knn
```

HTTPS 地址替换仅作用于该条命令，不修改全局 Git 设置。训练及评估所需的子模块按上述明确路径初始化。

核对版本：

```bash
git -C comparison_baselines/MesonGS rev-parse HEAD
git -C comparison_baselines/MaskGaussian rev-parse HEAD
git -C comparison_baselines/MaskGaussian submodule status --recursive
```

## MesonGS 复现入口与配置

- 官方环境说明：`MesonGS/README.md` 的 Installation 部分。仓库同时说明 CUDA 11.x 与 CUDA 12.x 的安装方式，建议建立独立环境。
- CUDA 扩展：`submodules/diff-gaussian-rasterization`、`submodules/simple-knn`、`submodules/weighted_distance`。
- 压缩与微调入口：`mesongs.py`。
- 官方运行示例：`scripts/mesongs_block.sh`。需要填写数据、PLY、输出和 CSV 路径，并指定 GPU。
- 输入 PLY 参数：`--given_ply_path`。
- 无微调设置：`--iterations 0`，仍会执行重要度计算、剪枝、量化和压缩。
- 独立解压渲染入口：`render.py --dec_npz --given_ply_path /path/to/bins.zip`，其余场景参数参见官方 README。
- 压缩包位置：`point_cloud/iteration_<step>/pc_npz/bins.zip`。

启动前需要处理的配置：

1. `utils/system_utils.py` 中的 `MAIN_DIR` 是上游保留的本机路径。独立 `render.py` 使用该路径写评估记录，运行前需配置为服务器上的 MesonGS 目录。
2. 固定版本的 `mesongs.py` 内置 `universal`、`syn_small`、`config2`、`config3` 配置。它会用配置值覆盖剪枝比例、码本大小等参数；实际配置以该入口代码为准。
3. 上游 `prune_mask()` 使用 `<=` 阈值，因此设置剪枝比例为 0 也可能移除最低分点。需要原样保留全部输入点时，应在后续适配中显式处理该边界，并记录变更。
4. `configs/c1.json` 和 `configs/c3.json` 存在于仓库，但该固定版本的 `mesongs.py` 没有读取它们的入口。不能仅凭 JSON 文件名认定实验已经应用了对应配置。

默认实验通过 `mesongs_adapter.py` 映射实际 JSON 配置、处理零剪枝边界和文件路径，并调用官方压缩组件，避免重复保存发送端评估图。适配不修改上游仓库。接收端支持本次编码实际生成的六文件 NPZ 压缩包，复用官方 RAHT 树和逆变换方程，保留官方坐标端点打包语义并导出标准四元数 PLY；具体变更和解码来源写入 `encode.json`。

## MaskGaussian 复现入口与输入要求

- 官方环境说明：`MaskGaussian/README.md`；建议建立独立环境。
- 从头训练入口：`train.py -s /path/to/scene --eval -m /path/to/output`。
- 后剪枝入口：`prune_finetune.py`，官方示例为 `scripts/run_prune_finetune.sh`。
- 后剪枝要求：`--start_checkpoint /path/to/chkpnt30000.pth`。固定版本代码会读取训练检查点；仅提供 `--start_pointcloud` 不会替代该读取路径。
- 官方导出：`Scene.save()` 调用 `save_ply_default()`，按保留掩码导出标准 float32 PLY，输出不含训练概率表。
- 评估入口：`render.py -m /path/to/output --skip_train`，随后执行 `metrics.py -m /path/to/output`。

本项目此前支持 PLY 初始化的剪枝适配位于主项目的 `prune_finetune.py` 和 `utils/prune_initialization.py`，与此处固定的官方入口分别保存。复用已有剪枝结果时应记录产生该文件的实现和配置。

## 数字交付的计费约定

数字链路按实际编码后的分组长度统计信道使用次数。坐标、属性、码本、量化参数、必要元数据和纠错冗余均计入。

无损打包恢复后的 MaskGaussian PLY 应与发送前逐字节一致。MesonGS 应从接收的压缩包独立解压并渲染，评估使用统一的相机、分辨率和指标实现。

共享的原始图像和 PLY 继续存放在主项目 `data/` 下，实验输出放在主项目 `output/` 下，各基线直接引用其绝对路径。
