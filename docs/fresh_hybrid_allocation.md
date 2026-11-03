# 固定收益网络，从头训练分配器

入口 `scripts/test_fresh_hybrid_allocation.sh`。读取原混合实验的配置和 `benefit_network.pt`，固定其 700 窗口末步收益网络，完整训练新的分配器。

| 状态 | 本版处理 |
| --- | --- |
| 编解码器、源高斯属性 | 复用并冻结 |
| 收益网络与固定尺度 | 复用并冻结 |
| 保留概率 | 全点初始化 0.99 |
| q1–q3 条件概率 | 全点初始化为三档均匀分布 |
| 收益历史、真实纠偏 | 清空 |
| 候选覆盖、评价计数 | 从零开始 |
| 相机调度、随机状态 | 按原 seed 新建 |
| 分配步数 | 从第 1 步开始，默认完整 2000 步 |

场景、通信惩罚、MSE＋SSIM 目标、信道、候选规则、空间输入和评估条件沿用源配置。q0 使用原密集反馈算法和原上限。正档位独立上限默认 1.0，实际更新仍受候选 trust 调节；`--unbounded-tier-updates` 直接执行原始正档位镜像更新。数值溢出检查和 -700 支持下限保留。

新运行只读取源 `allocation/plan.json` 和紧凑收益网络，不需要旧分配 checkpoint、分配快照或预训练 replay。支持原混合管线根目录及其 `allocation` 目录作为 source。其他空间输入变体需使用其对应入口。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

SOURCE="$PWD/output/truck_hybrid_tiers_gpu0_20261103_124400"
GPU=3
OUT="$PWD/output/truck_fresh_tier_step1_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_fresh_hybrid_allocation.sh \
  "$SOURCE" "$OUT" --tier-max-log-odds 1.0 \
  >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
echo "$OUT"
tail -n 40 -f "${OUT}.log"
```

GPU 选择空闲卡。使用新 OUT，将上限参数换成 `--unbounded-tier-updates` 即可测试不限幅；换成 `--tier-max-log-odds 0.25` 可进行相同初始化的原强度对照。

启动先打印网络复制与 `fresh_allocation` 状态，然后加载场景并做 q3 标定、初始评估。首次训练输出为 `allocation 1/2000`。源网络、编解码器和场景身份会在分配器加载时校验。

## 恢复与输出

恢复使用同一 SOURCE 字符串、同一 OUT 和同一上限/步数配置：

```bash
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/test_fresh_hybrid_allocation.sh \
  "$SOURCE" "$OUT" --tier-max-log-odds 1.0 >>"${OUT}.log" 2>&1 &
```

恢复加载本轮 allocation checkpoint 和 OUT 内的固定收益网络，原混合实验目录可移走。源编解码器与数据集路径仍须有效。

`plan.json` 记录全新初始化、网络 SHA256、复用的预训练窗口和原配置。`allocation/metrics/loss.jsonl` 包含候选覆盖、实际概率、真实纠偏与限幅前后步幅。完成后比较初始/末步/选中分配和全 q2/q3，并保存初始及选中图像、档位分布图。

检查与下载优先使用 `review_summary.json`、`review.tar.gz`。结果包排除网络和 checkpoint；本轮恢复状态另留在服务器。原实验继续运行并写入其原目录。

本地定向验证覆盖首步全新初始化、无旧 checkpoint 启动、冻结收益网络、限幅/不限幅断点恢复、源目录移走后的恢复及原入口兼容。最终部署质量和 CUDA 速度由服务器运行确认。
