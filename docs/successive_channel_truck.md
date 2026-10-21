# 条件增强信道编解码器：Truck 独立实验

入口为 `gaussian_jscc.successive_experiment`，脚本为
`scripts/test_successive_channel_truck.sh`。语义两端和属性标准化参数全程冻结；
源目录只提供同一语义教师及场景拆分，信道两端全部随机初始化、Adam为空。
默认累计预算8/24/48复符号、48维实数y、固定10dB AWGN、原始256点块。
已有实验和检查点的默认结构、序列化标识不变。

## 实际结构

基础编码器读取y，生成8个复符号。发送端用无噪声基础前缀和量化尺度运行本地
解码，得到停止梯度的估计；增强编码器读取原始y和其残差，生成额外16个符号。
第三层再读取原始y及前两层估计残差，生成额外24个符号。三次映射各使用隐藏192、
三个残差MLP块。原始y保留在增强输入中，用于学习抵抗低档噪声的冗余。
本地模拟没有真实接收噪声，也没有邻居修复和接收端反馈。计算具有顺序依赖，
不再是旧编码器一次输出全部符号的同一计算量。

接收端每级读完整可用前缀、该前缀对应尺度、前级恢复和SNR；后两级累加残差。
量化尺度恢复输入幅度，log2尺度同时提示对应噪声幅度。额外逐点补偿并入主干；
末端仅保留一个独立空间修复分支，按已交付XYZ的世界距离找原块内16个邻居。
低档可以使用实际收到的高档邻居信息，未传的自身符号和q0均不会进入修复。

## 尺度交付和成本

每段用**该段实际发送点**的块RMS归一化，floor为0.01。尺度向上舍入为IEEE
binary16，发送信号也除以同一个量化值；接收端不会使用未交付的精确尺度。
训练用直通估计近似量化梯度。超过binary16有限范围明确报错，避免静默溢出。

`SCL1`帧含4字节标记、4字节条目数，以及按q推导的块/段顺序排列的2字节尺度。
没有发送者的段不交付尺度。满256点、三段均存在时，内容为48bit/块，即
0.1875bit/点；另加每个原codec批次8字节帧头和可靠链路开销。短块以实际条目计费。
`scale_stream_bytes`、`scale_content_bits`和`scale_channel_uses`都会保存；总通信量
包括模拟payload、XYZ、档位/模型元数据和尺度帧。bit到信道使用的换算仍明确标注
`net_bits_per_use_assumption`，不把它当作普适物理常数。XYZ/尺度仍假设可靠，未模拟FEC/丢包。

`encode_channel_packet(y,q,snr)`返回模拟复符号payload与实际尺度字节；
`decode_channel_packet(received,q,snr,scale_frame,xyz)`可以仅凭接收数据恢复y。
评估实际执行尺度序列化/解析；训练前向的尺度值与字节解码一致。
旧symbol-only encode/decode和分配器入口会明确拒绝本结构，防止漏传尺度。
本实验尚不接学习分配器；mixed为固定随机正档位评估。

## 训练和损失

| 阶段 | 更新数 | 更新参数 | 学习率 |
|---|---:|---|---|
| main | 8000 | 条件信道编码器、三级接收主干 | 1e-4 |
| neighbor | 1000 | 零输出初始化的空间修复分支 | 1e-4 |
| cooperative | 1000 | 信道两端主干、空间分支 | 主干1e-5，空间1e-4 |

main每次对同一批数据的三个前缀分别监督，取均值。也就是说，8000次更新包含
每档8000次主干读出监督，与历史轮换各档2000次不属于等计算量单因素对照。
之后两阶段q1/q2/q3/mixed独立轮换。cooperative共同监督空间修复前、后输出，
并非解冻语义解码器或整场景渲染反向训练。阶段切换清空Adam；阶段内精确续训
保留动量、RNG、训练游标和颜色EMA。

每步共同目标为原始y MSE加0.1倍局部Gaussian黑/白背景响应误差；目标属性来自
冻结语义解码器D(y)，不是正确原PLY属性。各前缀共享同一组训练方向；八个方向
的一半来自训练相机、一半为球面采样。测试相机不进入训练损失。

每四步施加独立颜色辅助，**共同恢复目标始终存在**。其余步骤也计算颜色误差作为
连续诊断，这会增加少量SH计算；只有辅助步骤才反传该项。默认250步后开始维护
共同/颜色梯度平方和的EMA（动量0.01），500步后开启颜色辅助。
颜色系数为0.25倍EMA梯度范数比，并受本次范数比和系数1限制。
这限定的是对所监督latent的辅助梯度尺度，不保证Adam实际参数更新比例；
相应权重、梯度和实际模块更新会记录；`optimizer_group_updates`进一步区分编码器、
接收主干与邻居分支的学习率、梯度和更新范数。系数0.1、0.25及热身步数是公开可调的
工程起点，未宣称来自论文最优设置。没有最初四批校准，也没有纯颜色替换步骤。
main/cooperative每四步另加0.25倍无噪声主干y MSE，复用同一发送符号。

## 服务器启动

GPU0仅为示例，请改为实际空闲卡。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/truck_channel_response_scratch_gpu0_20261021_153959"
OUT="$PWD/output/truck_successive_gpu0_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=0 nohup bash scripts/test_successive_channel_truck.sh \
  "$RUN" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 30 -F "${OUT}.log"
```

默认源必须是已完成且语义冻结、8/24/48、逐块归一化的信道结束检查点，防止误选
适应后语义教师；可以传之前power-stability根目录。源的信道参数和动量均不读取
为初始化。路径迁移可传`--manifest`，仍核对PLY指纹和相机拆分。

附加`--dry-run --device cpu`只检查元数据，不建输出、不加载PLY、不训练。
中断后用相同RUN、OUT和选项附加`--resume`；完成目录直接返回完成状态。
学习率/步数等改变需要另开实验，精确续训拒绝不一致设置。

## 输出

每500步及阶段结束保存数值验证；默认32测试视角、四次噪声，记录clean、无噪声、
10dB各档的原PLY/照片PSNR、SSIM、VGG-LPIPS和颜色偏差。中间验证用原留出小集合，
最终用固定测试集合，避免混用曲线口径。图片仅保存初始/最终两个代表视角。

`diagnostics/*.json`在固定原块上保存主干/空间修复后的latent、局部响应和方向颜色
误差，以及前缀读出误差；其单次固定噪声结果不替代32视角多次噪声画质指标。
所有结果在一个OUT目录，含loss.jsonl、evaluation、diagnostics、comparison及
initial/latest/end_main/end_neighbor/end_cooperative/final检查点。

本机仅验证真实PyTorch信号、尺度字节、梯度、训练阶段、精确续训和旧入口兼容性；
CPU合成渲染测试不证明Truck实际质量或GPU峰值，需要服务器实验确认。
