# 连续掩码 RD 分配实验

该入口在当前 successive 编解码器上恢复完整 A 路线。历史依据为 `3dc7df8`、`0e4e411`、`c495a21`；复用仍保留的实现接口，新增独立实验入口。收益网络与直接换挡实验保留。

## 本轮训练内容

编解码器与高斯属性全部冻结。每个高斯维护一个保留／删除概率和三个条件档位概率。每轮全部高斯参加单次硬 Gumbel 采样；通过真实 mixed 场景渲染与局部 ST 梯度更新概率表。q0 点以零存在门控的反事实属性获得保留梯度，不可见点的图像梯度可以为零。资源期望梯度覆盖全部点。

正档位沿用整块候选的局部梯度近似，q0 反事实解码也依赖候选上下文。实际前向使用当前硬采样场景。这两项近似仍是本实验需要通过最终部署效果检验的部分。

部署使用固定种子的十次采样：十次全为 q0 才删除，其余选择正档位众数。默认初始保留概率 0.99，正档位等概率。

## 目标与收尾

\[
L=\frac{\operatorname{MSE}+\lambda_s(1-\operatorname{SSIM})}{D_{\mathrm{ref}}}
  +\beta\frac{C}{C_{\mathrm{ref}}}.
\]

参考值取固定校准条件下全 q3 的表现；结构项在参考失真中的初始占比为 20%，随后固定 `lambda_s`。默认 `beta=0.1`。通信单位为复信道使用，包含属性、可靠 XYZ、元数据与 binary16 块尺度；可靠 bit/use 沿用源实验设定。训练采用单次采样的资源期望、实测 XYZ 每保留点代理及本轮元数据／尺度常量，部署评价采用完整实测流开销。

默认参数：

| 项目 | 设置 |
| --- | --- |
| 档位与信道 | 0／8／24／48，10 dB AWGN |
| 训练上限／最少训练 | 2000／500 步 |
| Adam 学习率 | 保留 0.01，正档位 0.001，eps=1e-15 |
| 每步训练视角 | 4 个，打乱轮转 |
| 部署校准 | 每 100 步，4 个独立校准视角，4 次固定噪声试验 |
| RD 有效改进阈值 | 相对改进 0.1% |
| 停滞降学习率 | 连续 3 次，乘 0.5，最低初始值的 1/8 |
| 提前结束 | 最低学习率后再停滞 3 次；同期档位变化 ≤0.5%，开销波动 ≤0.2% |

降学习率和收尾均依据校准视角。测试照片不参与调度。持续变化的概率表训练至步数上限后结束；报告注明结束原因。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

GPU=2
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_dense_rd_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_dense_rd_allocation.sh \
  "$CODEC" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
echo "$OUT"
tail -n 40 -f "${OUT}.log"
```

`GPU` 按实际空闲卡调整。若源路径迁移，可通过 `--manifest` 提供同一场景的实际路径，点数与内容指纹须匹配。

## 续训和读取

正常 `kill -TERM PID` 会请求在当前步结束后保存。日志出现 `interrupted: latest table/Adam/RNG saved` 后即可重启。强制结束从最近 checkpoint 恢复。

```bash
# 新终端中重新指定原 OUT。
OUT="$PWD/output/原实验目录名"
GPU=2
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/test_dense_rd_allocation.sh \
  "$CODEC" truck "$OUT" --beta 0.1 >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -f "${OUT}.log"
```

恢复要求原 checkpoint、源编解码器和场景输入都在。默认每 100 步保存，校准后也保存。源码没有复制源编解码器到输出目录。

训练进度读取 `progress.json`、`metrics/training.jsonl`、`metrics/control.jsonl`。完成后直接读取：

```bash
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
print(json.loads((p/'complete.json').read_text()))
s = json.loads((p/'review_summary.json').read_text())
for name, r in s['rows'].items():
    print(name, 'PSNR=', round(r['source_psnr'], 4),
          'SSIM=', round(r['source_ssim'], 5), 'LPIPS=', r['source_lpips'],
          'uses(M)=', round(r['total_channel_uses']/1e6, 5),
          'RD=', round(r['rd_score'], 7), 'tiers=', r['tier_counts'])
print('相对全 q2：', json.dumps(s['against_q2'], ensure_ascii=False, indent=2))
PY
```

`latest` 为末步，`best_rd` 为最低校准 RD 对应的概率表。二者都在相同测试条件下测量。`against_q2` 记录画质差、成本节省和目标是否达到。只对 `initial` 与 `best_rd` 保存效果图和档位图。

日常检查下载 `review.tar.gz` 即可。需要恢复训练时，另存 `checkpoints/latest/state.pt`。本入口没有标签 replay、大量历史 checkpoint 或重复的场景 PLY。

## 验证范围

CPU 定向测试覆盖实际 successive 小模型、共享 checkpoint 加载、q0／正档位梯度、冻结验证、归一化多视角梯度、成本顺序、调度器与中断恢复一致性。测试渲染使用合成可微函数；实际 CUDA 光栅化与训练速度由服务器运行确认。
