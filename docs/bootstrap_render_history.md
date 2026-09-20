# 补做第一阶段的渲染历史评估

`evaluate_bootstrap_history.py`只读取已保存的检查点，完整场景前向编解码并渲染；
**没有优化器、反向传播或权重更新，不会增加第二阶段训练。**
当前已保存codec_500.pt到codec_5000.pt，因此可以直接补做500、1000、…、5000步。
没有第0步检查点就不伪造第0步渲染；`codec_best_bootstrap.pt`也不替代对应步数权重。

默认固定10 dB AWGN、q1/q2/q3/mixed、8个间隔采样的test视角、每布局2次噪声试验。
每个检查点使用相同相机、同一混合档位分配和相同噪声种子；不同档位使用不同种子。
`--views 0`评估所有test视角。中途改变batch大小或分辨率后应作为新评估，不与旧曲线拼接。

## 服务器运行

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main || exit 1
TRAIN="$PWD/output/truck_learned_xyz_bootstrap_20260920_101516"
EVAL="$PWD/output/truck_xyz_render_history_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 nohup python -u evaluate_bootstrap_history.py \
  --training "$TRAIN" \
  --ply "$PWD/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply" \
  --source "$PWD/data/tandt_db/tandt/truck" --out "$EVAL" \
  --start 500 --stop 5000 --every 500 \
  --snr 10 --channel awgn --trials 2 \
  --views 8 --split test --resolution 2 --blocks-per-batch 32 --device cuda \
  > "${EVAL}.log" 2>&1 &
echo $! > "${EVAL}.pid"
tail -f "${EVAL}.log"
```

GPU 2仅作示例，选空闲卡；需要安装CUDA光栅化扩展及完整Truck相机/图像数据。
本机CPU只能执行模拟renderer的流程测试，不能给出真实场景PSNR。
必须使用本轮原始PLY；脚本核对点数与SH阶数，但同点数不同PLY仍需用户保证身份正确。
检查点配置与归一化统计不一致时停止，缺失任意请求步数也停止，不静默跳过。
所有输出进入新目录，训练目录和权重只读；逐检查点保存结果，异常前完成的结果仍保留。

## 输出

- `validation_images/000500/1_view00.png`：500步、q1、第一个视角。其余步数同理。
  图片四列是Photo、Source PLY、Received、对Source PLY的绝对误差×4。
  图像只保存trial0，显示RGB裁剪到[0,1]，不代表全部噪声试验。
- `quality_vs_step.png/.svg`：PSNR、SSIM、MSE和XYZ误差随训练步数变化。
- `metrics.csv`：每步、每布局的汇总数据，便于Excel查看。
- `results.json`：完整逐视角/逐噪声指标、checkpoint SHA256。
- `validation.jsonl`：评估期间追加的指标记录。
- `evaluation_config.json`：相机名称、分辨率、步数、信道、种子及指标口径。

`source_psnr/source_ssim`以原始PLY渲染为目标；`photo_psnr/photo_ssim`以真实照片为目标。
`reference_photo_psnr`是原始PLY对照片的参考质量，不能与source_psnr混比。
PSNR/MSE使用未裁剪RGB，SSIM使用显示RGB；PSNR按视角/试验的dB值取平均。
本脚本复用已存在的指标实现，不重复定义。曲线10个检查点使用折线，少于8点只画标记；
不对评估值平滑。四档布局用不同颜色和标记，源目标与照片目标独立面板。
全场景指标与第一阶段少量局部验证块的XYZ误差不属于同一统计范围。
输出不包括LPIPS；若反复据此选模型，应将这些相机视为验证集，再保留独立最终测试。

## 本地验证范围

CPU测试用真实codec和AWGN、替代的可微渲染函数检查流程；确认相同权重不同步数下
指标完全一致、权重文件hash不变、没有开启梯度/优化器，且PNG/CSV/JSON输出存在。
这不是CUDA实际渲染或当前训练模型的画质测试。
