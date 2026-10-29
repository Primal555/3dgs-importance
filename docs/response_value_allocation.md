# 候选渲染响应统一价值网络

入口：`scripts/test_response_value_allocation.sh`。独立family为`render_response_value_v1`，保留旧实验入口。

## 本轮配置

冻结源编解码器，不重训。10 dB；q0/q1/q2/q3为0/8/24/48；可靠16bit XYZ和binary16块尺度；保持原通信计量、beta、镜像更新及十次部署规则。

网络取消一阶预测、梯度、梯度范数和辅助屏蔽。四档价值共享网络，q0固定为零，正档位换挡由两个价值相减。训练采用尺度归一化数值SmoothL1，取消方向置信间隔损失。报告同时保留误判代价、数值误差、共同误差、三档及四档选择regret与高代价错误。

每个空间块两轮轮转选点，包含已关闭点；两条概率分支使用同一批点。每个点在实际邻居档位下重新解码四个状态，相同噪声起点，存在集合变化时重新语义编码。输入包含实际候选属性、邻居变化、投影椭圆、方向颜色，以及覆盖区域7×7 RGB／参考残差／累计alpha／深度矩。alpha和深度矩属于可见性代理。

每窗口32次真实干预上限，完整四状态组复用当前状态，最多10组。输入先冻结，再渲染标签。真实标签包含受影响邻居的变化，标签图像不作为特征。空间块验证标签不训练。窗口不运行渲染反向传播。

## 服务器启动

旧replay缺少候选和patch信息，本轮需要GPU重新收集。CPU小网络续训结果不能直接作为本轮输入。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs
RUN="$PWD/output/successive_mip360_gpu1_20261023_142948"
OUT="$PWD/output/truck_response_value_gpu1_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/test_response_value_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 >"${OUT}.log" 2>&1 &
printf 'PID：%s\n输出：%s\n日志：%s\n' "$!" "$OUT" "${OUT}.log"
tail -n 40 -F "${OUT}.log"
```

默认200个窗口预训练，前50窗口拟合训练特征与标签尺度，后150窗口各4次Adam更新。概率表保持初始化不变。检查`comparison/benefit_validation.json`和`metrics/benefit_validation.jsonl`，以价值误差、实际方向regret、三档及四档选择regret判断。最佳检查点评分同时包含这些误差，并使用固定训练标签尺度归一化。

确认评估网络效果后，沿用同一OUT显式开启2000步分配：

```bash
CUDA_VISIBLE_DEVICES=1 RESUME=1 nohup bash scripts/test_response_value_allocation.sh \
  "$RUN" truck "$OUT" --beta 0.1 --phase allocation >"${OUT}.allocation.log" 2>&1 &
tail -n 40 -F "${OUT}.allocation.log"
```

若预训练进程中断，使用相同命令加`RESUME=1`即可精确恢复。迁移目录时使用原有`--manifest`。beta和其余配置必须与对应OUT一致。

## 性能与保存

候选计算复用当前档位结果，并按每块的相对档位轮换执行其余三个状态。每批两轮选点的候选编解码调用从8次降为6次，仍输出q0至q3四个完整块结果。保留集合不变时复用语义编码；选中点在q0与正档位间变化时刷新对应集合的编码。信道前向保留原批次大小、噪声位置、块尺度交付与邻居恢复。

日志中的`candidate_codec_calls`记录实际候选编解码调用，新增`candidate_semantic_calls`记录候选阶段的语义编码刷新次数。缓存只存活一个窗口，不写入检查点。已有200窗口预训练结果可以直接用本页相同的`RESUME=1 --phase allocation`命令继续。

CPU单线程小块测试（8块、每块128点、两端PointNet上下文，预热2次后测10次中位数）中，候选计算从0.249秒降至0.168秒，约加速1.49倍。该数值仅对应候选计算；真实标签渲染、patch提取和训练预算保持不变，服务器整步提速以`step_seconds`实测为准。新增实际语义缓存约占“点数×latent维数×每元素字节数”的显存，候选临时缓存限制在当前一个编解码批次内。

记录`candidate_decode_seconds`、`response_patch_seconds`、`response_context_render_seconds`、`exact_response_render_seconds`、单步时间和显存峰值。候选流式处理，完整候选块只为本窗口真实标签子集暂存。新增候选输入的成本需要服务器实测，不能沿用旧CPU replay的耗时。

保留既有最初和最终部署图、定期数值验证与原子检查点。最终部署仍记录PSNR、SSIM、LPIPS、实际档位图和完整通信成本。
