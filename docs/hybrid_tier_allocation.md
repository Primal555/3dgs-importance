# 全点一阶保留＋空间网络正档位

入口 `scripts/test_hybrid_tier_allocation.sh` 自动完成两个阶段：

| 阶段 | 默认预算 | 更新对象 |
| --- | --- | --- |
| 空间收益网络预训练 | 700 窗口 | 收益网络；编解码器和概率表固定 |
| 场景档位分配 | 2000 步 | 全点保留概率、候选点正档位概率；收益网络默认冻结 |

本轮收益网络从随机权重开始，前 50 窗口收集标签并拟合固定尺度，其后每窗口 64 次更新，合计 41600 次网络优化。预训练完成后自动使用第 700 窗口末步网络，不需要重新执行启动命令。

## 索引修复（2026-11-04）

默认入口在预训练及分配中均使用 `AlignedSpatialWindow`。`scene.raw` 按空间顺序存储，点编号沿用原始 PLY 编号：静态统计通过 `scene.order` 写回原始编号对应行；关闭点的探针直接从相同渲染行的 `scene.raw` 读取。计划、预训练 replay 的计划及导出网络均记录 `aligned_spatial_inputs=true`，推理核验网络与分配配置的布局一致。

修复后请使用新的 OUT，从头训练 700 窗口收益网络，再进行 2000 步分配。编解码器继续复用。旧网络和 replay 的输入曾发生索引错配，不能直接用于修复后的布局。旧实验的 fresh／snapshot 专用入口按保存的布局恢复；将旧网络与新布局混用会报错。两种布局的 checkpoint、replay 不做自动转换。

## 分配机制

保留分支沿用 `calibrated_structured_allocation_v2`：渲染存在门控的一阶导数、全点每轮反馈、历史、真实纠偏和置信度保持原路径。正档位候选沿用 coverage_demand 两轮规则；空间网络预测 MSE、结构误差两个价值分量，以 V(q)-V(current) 替换正档位数值近似，后续仍使用历史、真实纠偏和镜像更新。网络输出不会写入 keep_costs。

网络与最近空间配对版相同，输入包含候选恢复属性、邻居空间响应、图像残差和可见性统计；不输入一阶预测。网络输出的结构分量已包含 lambda_s/D_ref，接入时还原为旧分配器的分量单位，通信差价独立计算。旧纠偏日志中的 first_order_delta_* 字段为兼容字段；本轮请结合 proxy_estimator、proxy_delta_* 判断具体估值来源。

默认 beta=0.1、MSE＋SSIM、0/8/24/48、10 dB，沿用可靠 16bit XYZ 和 binary16 尺度成本。保留镜像步长 0.1、正档位步长 0.4、最大单次对数优势变化 0.25；十次部署规则不变。编解码器及属性始终冻结。默认固定分配预算 2000 步，对照全 q2、全 q3，报告末步和最低校准 RD 对应结果。

候选与一阶保留共用相同采样、实际解码、语义特征和信道噪声。神经网络路径跳过旧正档位候选解码，使用 batch128 合并四状态投影和空间特征。窗口只缓存必要数据，下一步释放。

## 冻结与在线学习

默认 `--network-mode frozen`。冻结的是收益网络参数，保留表与正档位表继续学习；每轮仍调用收益网络评估候选。旧版最多 32 次真实纠偏仍在执行，不会用这些标签更新冻结网络。

显式选择 `--network-mode online` 后，候选还会按预训练设置每步选最多 64 组训练点，额外获取完整四状态监督，再执行 64 次网络更新。原预训练验证空间块不参与训练。活动 replay 从预训练末步状态初始化到独立目录，优先硬链接不可变分片，源输出不受 GC 影响。此模式增加了真实标签预算，效率与结果应单独报告。日志含 neural_labels、neural_teacher_seconds、network_updates，可确认网络是否更新。

在线更新可适应删点和正档位概率变化后的上下文；其监督仍来自真实四状态干预，概率表的选择不充当标签。网络的优化器、历史样本和固定归一化尺度由预训练延续。2000 步最多新增 128000 次优化，累计网络更新可达 169600 次。当前实现没有自动清空旧预测历史；该历史仍按原规则衰减，因此在线网络的改进需要通过部署 RD 验证。在线分配阶段记录训练损失和网络更新次数，未另行新增固定验证集评估曲线。

首轮默认冻结，目的是观察第 700 窗口评估器参与分配的效果。在线模式支持后续检查当前分布适应是否有益。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs

GPU=1
CODEC="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
OUT="$PWD/output/truck_hybrid_tiers_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"

CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_hybrid_tier_allocation.sh \
  "$CODEC" truck "$OUT" --beta 0.1 --network-mode frozen >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
echo "$OUT"
tail -n 40 -f "${OUT}.log"
```

两阶段自动衔接，总预算 700＋2000。GPU 改为空闲卡。若要在线版本，使用新的 OUT，将 frozen 改为 online；不要在恢复同一 OUT 时切换模式。

同配置恢复：

```bash
# 新终端需重新指定原 OUT、CODEC、GPU。
CUDA_VISIBLE_DEVICES="$GPU" RESUME=1 nohup bash scripts/test_hybrid_tier_allocation.sh \
  "$CODEC" truck "$OUT" --beta 0.1 --network-mode frozen >>"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
```

也可显式 `--phase pretrain` 只完成预训练；之后使用同一 OUT、RESUME=1、--phase allocation 开始分配。默认 auto 会自动完成全过程。

## 输出和检查

- `pretrain/metrics/benefit_validation.jsonl`：收益网络学习曲线。
- `benefit_network.pt`：第 700 窗口的紧凑收益网络权重、尺度和训练信息，不含编解码器或 replay 数据。
- `allocation/metrics/loss.jsonl`：分配进度，含 dense_first_order／spatial_network 分工、网络更新次数和实际候选解码次数。
- `allocation/comparison/`：同条件 q2/q3 对照、部署 RD 和档位变化。
- `review_summary.json`、`review.tar.gz`：检查结果时优先读取、下载。

只对初始和最终选中部署结果保存效果图及档位图；末步若为另一张分配图，保存指标与档位数组。结果包排除 checkpoint、replay、编解码器和候选缓存。

恢复预训练需要完整 pretrain checkpoint 与其 replay。分配阶段开始后，冻结模式恢复需要 benefit_network.pt、allocation checkpoint、源编解码器和场景；在线模式还需 allocation/neural_replay。精确恢复保持路径与运行配置一致。

```bash
python - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not (p/'complete.json').exists():
    for name in ('pretrain/pretrain_complete.json','allocation/progress.json'):
        f=p/name
        if f.exists():print(name,json.loads(f.read_text()))
else:
    s=json.loads((p/'review_summary.json').read_text())
    print(s['result'])
    for name,r in s['rows'].items():
        print(name,'PSNR',round(r['source_psnr'],4),'SSIM',round(r['source_ssim'],5),
              'LPIPS',r['source_lpips'],'uses(M)',round(r['total_channel_uses']/1e6,5),
              'RD',round(r['rd_score'],7),'tiers',r['tier_counts'])
    print(json.dumps(s['against_q2'],ensure_ascii=False,indent=2))
PY
```

本地定向验证使用真实小编解码器、真实空间价值网络及 CPU 候选特征，完整相机渲染使用合成函数；真实 CUDA 性能和最终画质在服务器运行确认。
