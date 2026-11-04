# 3DGS 数字交付对照实验

本目录保存 MesonGS 与 MaskGaussian 的官方源码，用于准备场景压缩和数字交付对照。两个源码目录均为 Git 子模块，版本由本项目的 Git 提交固定。

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
- 已核对压缩、剪枝、保存和渲染入口。
- 本机核查环境为 `torch 2.9.0+cpu`，`torch.cuda.is_available()` 为 `False`。CUDA 扩展尚未编译，完整训练、压缩和渲染结果尚未复现。
- LDPC、QAM、分组与信道测试需要后续实现；本目录目前提供官方源码和复现准备说明。

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

本轮保留上述上游行为，后续复现适配在主项目中单独记录。

## MaskGaussian 复现入口与输入要求

- 官方环境说明：`MaskGaussian/README.md`；建议建立独立环境。
- 从头训练入口：`train.py -s /path/to/scene --eval -m /path/to/output`。
- 后剪枝入口：`prune_finetune.py`，官方示例为 `scripts/run_prune_finetune.sh`。
- 后剪枝要求：`--start_checkpoint /path/to/chkpnt30000.pth`。固定版本代码会读取训练检查点；仅提供 `--start_pointcloud` 不会替代该读取路径。
- 官方导出：`Scene.save()` 调用 `save_ply_default()`，按保留掩码导出标准 float32 PLY，输出不含训练概率表。
- 评估入口：`render.py -m /path/to/output --skip_train`，随后执行 `metrics.py -m /path/to/output`。

本项目此前支持 PLY 初始化的剪枝适配位于主项目的 `prune_finetune.py` 和 `utils/prune_initialization.py`，与此处固定的官方入口分别保存。复用已有剪枝结果时应记录产生该文件的实现和配置。

## 数字交付的计费约定

后续数字链路按实际编码后的分组长度统计信道使用次数。坐标、属性、码本、量化参数、必要元数据和纠错冗余均计入。

无损打包恢复后的 MaskGaussian PLY 应与发送前逐字节一致。MesonGS 应从接收的压缩包独立解压并渲染，评估使用统一的相机、分辨率和指标实现。

共享的原始图像和 PLY 继续存放在主项目 `data/` 下，实验输出放在主项目 `output/` 下，各基线直接引用其绝对路径。
