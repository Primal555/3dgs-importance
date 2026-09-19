# MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks [CVPR 2025]

## Gaussian JSCC codec extension

The default codec is **fully learned joint JSCC**: XYZ and attributes share
a learned variable-length payload; the receiver jointly decodes local noisy
features without source/predicted-coordinate grids. Each Gaussian has its own
q0/q1/q2/q3 decision. There is no handcrafted coordinate reference, repetition,
or reserved geometry sub-budget.

Use `python -m gaussian_jscc train-learned` (`train` is the same entry point)
or `bash scripts/train_codec_learned.sh`.
Training now uses **multiview source-render RGB MSE**, not a weighted sum of
per-Gaussian attribute errors. Optional feature bootstrap is initialization only.
Launch scripts default to random weights and zero bootstrap steps; inherited
`INIT` is ignored unless `INITIALIZATION=checkpoint` is explicitly selected.
For a full-scene, reduced-view/step observation run, use
`bash scripts/test_render_first.sh`; see [render-first protocol and outputs](docs/render_first_training.md).
See [architecture, objectives and server commands](docs/learned_joint_jscc.md),
[mask joint training](docs/route2_joint_jscc.md),
[training performance](docs/training_performance.md) and
[position/attribute ablation](docs/hybrid_codec_ablation.md).

To test whether position recovery blocks render-only cold starts, run the
[explicit XYZ-delivery comparison](docs/position_delivery_ablation.md): matched
random initializations with learned XYZ, reliable float32 XYZ, or reliable
quantized XYZ. Extra coordinate bits are counted; this is not an equal-total-rate
comparison and does not simulate error correction for that side stream.

The preserved **quantized-12-bit, render-only** baseline remains available:
`CUDA_VISIBLE_DEVICES=2 bash scripts/train_quantized12_render_only.sh`
(choose a free GPU). It fixes LR=1e-4, all training views, zero bootstrap/joint
steps and defaults to 5000 steps. See [historical snapshot and continuation commands](docs/quantized12_render_only.md).
It retains replay backward and constant LR for historical comparisons.

For the optional two-stage experiment, use
`CUDA_VISIBLE_DEVICES=2 bash scripts/test_local_response.sh` (choose a free GPU).
It starts from random weights with 12-bit/axis reliable XYZ, trains isolated
Gaussian responses for 2000 steps, then full-scene rendering for 300 steps.
Both learning rates start at 1e-4, with phase-local validation plateau decay
(3 bad checks, 0.5% relative improvement threshold, factor 0.5, floor 1e-6).
The latest two-stage launcher uses **direct backward without codec recomputation**;
it retains full-scene codec activations, so CUDA memory requirements are higher.
There is no mixed mechanism or automatic fallback. LR events and per-step CUDA
peak memory are logged. See
[local-response objective, limitations and logs](docs/local_response_pretraining.md).
This replaces attribute-wise weighted losses in pretraining, not the final
scene-rendering objective; it does not yet establish improved rendering quality.

The independent `benchmark_codec.py` and packet-only receiver remain available.
Training and evaluation export PNG/SVG charts, machine-readable logs and CSV data.
The implementation adapts ROI-JSCC prefix transport and FCGS-inspired sender
aggregation; upstream MaskGaussian scene training is unchanged.

Historical codec implementations, training scripts and migration branches have
been removed from the mainline. They remain recoverable from Git history
(snapshot `a545aa5`). Old checkpoints require their historical code revision;
only learned_joint v4 is loaded by this mainline. Existing experiment output
files are not deleted. Current learned-v4 weight and packet identities are preserved.

<div id="top" align="center">
 
<a href="https://arxiv.org/abs/2412.20522"><img src="https://img.shields.io/badge/Read-Paper-B31B1B.svg" height="23"></a>
<a href="https://maskgaussian.github.io/"><img src="https://img.shields.io/badge/Project-Page-048C3D" height="23"></a>
<a href="https://github.com/kaikai23/MaskGaussian"><img src="https://img.shields.io/github/stars/kaikai23/MaskGaussian" height="23"></a>
</div>


## :mega: Updates
[04/2026] 🎈: MaskGaussian is integrated into [Hunyuan-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0).

[07/2025] 🎈: We propose **mask-Grendel-GS**, combining **MaskGaussian** and **Grendel-GS** to support pruning in distributed training. This framework holds potential for pruning messive gaussian points in large scale scenes, where the excessive gaussian number is a main bottleneck. [https://github.com/kaikai23/mask-Grendel-GS/](https://github.com/kaikai23/mask-Grendel-GS/)

[03/2025] 🎈: Post-training code is released. Now you can also directly use MaskGaussian to prune an already trained 3D-GS!

[02/2025] Accepted to [CVPR 2025](https://cvpr.thecvf.com/).

[01/2025] We release the code.

## Overview
We introduce MaskGaussian to prune Gaussians while retaining reconstruction quality. It dynamically samples a subset of Gaussians to render the scene during training. Not sampled Gaussians also receive gradients through [mask-diff-gaussian-rasterization](https://github.com/kaikai23/mask-diff-gaussian-rasterization) and update their chance to be used in future iterations.

Our method improves rendering speed, reduces model size, GPU memory, and training time, and supports both training from scratch and post-training refinement.

<img height="250" alt="image" src="https://github.com/user-attachments/assets/4855522d-9fb2-4044-90f2-1ff9cb62b1d1" />


## Installation
1. **Clone the repository**
```
git clone --recursive https://github.com/kaikai23/MaskGaussian.git
cd MaskGaussian
```
2. **Install dependencies**
```
conda create -n maskgs python=3.9
conda activate maskgs
pip install "numpy<2.0" plyfile tqdm icecream torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit  # can be skipped if cuda-11.8 is already installed
CUDA_HOME=PATH/TO/CONDA/envs/maskgs pip install submodules/mask-diff-gaussian-rasterization submodules/diff-gaussian-rasterization submodules/simple-knn/
```

## Data Preparation
First, create a ```data/``` folder inside the project path by 

```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── gs_datasets
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```

### Public Data

- The MipNeRF360 scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/). 
- The SfM data sets for Tanks&Temples and Deep Blending are hosted by 3D-Gaussian-Splatting [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).

## Training and Evaluation in One Go
To train, render and evaluate our method on the 3 datasets in the paper, simply run:
```
python run_all.py
```
The training output is logged in `train.log` under the `output/scene_name` folder, and the final metrics are recorded in `results.json` under the same folder. The training time can be read from `train.log`, and GPU memory consumption can be read from `GPU_mem` card in tensorboard records by running `tensorboard --logdir /path/to/output` (when testing GPU memory, please use --data_device cpu).
Finally, note that **the output of our method is 100% in vanilla format and can be viewed directly in any 3dgs viewer**, such as popular [SuperSplat](https://superspl.at/editor) and [antimatter15](https://antimatter15.com/splat/).

## Training a single scene
To train a single scene, run:
```
python train.py -s /path/to/input_scene --eval -m /path/to/output
```
with optional parameters:

• **--lambda_mask**: the coefficient of mask loss

• **--mask_from_iter**: the start iteration for mask loss

• **--mask_until_iter**: the end iteration for mask loss

There are 3 settings in the paper, and their configurations can be found in `run_all.py`.

Last, to render and evaluate the test set, run:
```
python render.py -m /path/to/output --skip_train
python metrics.py -m /path/to/output
```
Since we do not save mask, no special handling is required for evaluation.

## Post-training and evaluation
To prune an already trained 3DGS, specify its checkpoint path in `scripts/run_prune_finetune.sh` and run:
```
bash scripts/run_prune_finetune.sh
```

## Vanilla 3D-GS baseline and pruning comparison

`input.ply` is only the COLMAP/Blender initialization point cloud. It is not a
trained 3D-GS baseline and must not be used as the denominator of a pruning
ratio. This repository includes a compatible vanilla training entry point:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_3dgs.py \
  -s /path/to/scene \
  -m /path/to/vanilla_output \
  --eval \
  --data_device cpu \
  --test_iterations 7000 30000 \
  --checkpoint_iterations 7000 15000 30000
```

The last iteration always produces both
`point_cloud/iteration_30000/point_cloud.ply` and `chkpnt30000.pth`. The latter
uses the original 12-item 3D-GS format expected by `prune_finetune.py`.

Render and evaluate the vanilla and MaskGaussian outputs on the same held-out
cameras, then calculate the representation-size reduction:

```bash
CUDA_VISIBLE_DEVICES=0 python render.py -m /path/to/vanilla_output --skip_train
CUDA_VISIBLE_DEVICES=0 python render.py -m /path/to/mask_output --skip_train
CUDA_VISIBLE_DEVICES=0 python metrics.py -m /path/to/vanilla_output /path/to/mask_output

python compare_models.py \
  --baseline /path/to/vanilla_output \
  --mask /path/to/mask_output \
  --make_visuals
```

`compare_models.py` writes `comparison_vs_baseline.json` and creates panels
containing ground truth, vanilla rendering, MaskGaussian rendering, and an
amplified difference image. Its Gaussian reduction is
`1 - N_mask / N_vanilla`. Because these are independent from-scratch training
runs, this is a representation-size reduction, not one-to-one Gaussian deletion
tracking.

For the complete baseline/render/metric/visual workflow, use:

```bash
bash scripts/run_baseline_comparison.sh \
  /path/to/scene \
  /path/to/vanilla_output \
  /path/to/mask_output \
  0
```

The GPU still performs all training and rendering. `--data_device cpu` only
keeps source camera images in CPU memory to reduce VRAM use.

## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## TODO List
- \[ \] Code of MaskGaussian + Taming-3DGS.
- \[x\] Support post-training.

## Contact

- Yifei Liu: liuyifei@pjlab.org.cn

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@InProceedings{Liu_2025_CVPR,
    author    = {Liu, Yifei and Zhong, Zhihang and Zhan, Yifan and Xu, Sheng and Sun, Xiao},
    title     = {MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {681-690}
}
</code></pre>
  </div>
</section>

## Acknowledgement

This project is built upon [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), [Compact-3DGS](https://github.com/maincold2/Compact-3DGS), and [LightGaussian](https://github.com/VITA-Group/LightGaussian). We thank all authors for their great work!
