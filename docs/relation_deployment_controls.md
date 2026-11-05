# 已部署档位的隔离评价

输入为已经完成的 `relation_benefit_context_v2` 实验目录。读取原 `deployment/selected_tiers.npy`，固定原codec、10dB、4次噪声试验、评估种子、图像损失权重及 D_ref/C_ref。代码只评价，优化器更新数为0。

两项变体：

| 名称 | 原q0 | 原q1 | 原q2 | 原q3 |
|---|---:|---:|---:|---:|
| restore_q0_to_q1 | q1 | q1 | q2 | q3 |
| kept_uniform_q2 | q0 | q2 | q2 | q2 |

每项分别报告原准备视角（本轮4个）与测试视角（本轮32个）。准备视角沿用 `training.json/preparation_view_names`，测试视角身份和顺序完全沿用原评价。真实照片只用于指标评价。

新变体合并36个相机，每次噪声试验只进行一次整场景解码，再按相机身份分别统计。原网络分配、全q1和全q2的测试指标在严格核对codec、档位哈希、相机、种子、试验数、SSIM权重与指标聚合后复用，只补测它们的准备视角。全部配置使用相同噪声种子和固定槽位协议。

通信成本按完整场景计费，在两个视角集合中相同；源PLY、模型、坐标、档位元数据和尺度流口径保持原实验约定。源目录只读。无需关系网络检查点、bank或replay；需要原codec检查点、PLY、相机及照片。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

SOURCE="$OUT"  # 原700窗口关系网络实验目录
GPU=3
CONTROL_OUT="$PWD/output/truck_relation_controls_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_relation_deployment_controls.sh \
  "$SOURCE" "$CONTROL_OUT" >"${CONTROL_OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${CONTROL_OUT}.pid"
tail -n 40 -F "${CONTROL_OUT}.log"
```

显式 `--run CODEC_RUN` 与 `--manifest MANIFEST` 支持重定位，数据身份和相机划分仍须匹配。中断后保留相同 `CONTROL_OUT`、输入及参数，使用 `RESUME=1`，可复用已完成的配置评价。

输出 `review_summary.json` 的 `scopes.preparation` 和 `scopes.test` 各包含5种配置的画质、统一完整成本、RD及两个变体相对原分配／全q1／全q2的差值。`maps/` 保存两个变体的原PLY顺序档位图；每变体保存2张最终图，分别对应首个准备视角和最后一个测试视角。精简 `review.tar.gz` 不含模型与训练缓存。

评价解释：q0恢复同时改变被恢复点及编解码上下文；固定删除图全q2也会改变信道恢复条件。两项差值分别测量整个干预的效果，不能相加作为严格的损伤分解。
