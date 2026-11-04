# 数字交付对照实施计划

用户要求补齐两个官方仓库获取、数字链路交付、统一评估和服务器启动流程。以现有剪枝后 PLY 作为两个基线的公共输入，保持当前 JSCC 训练代码和官方子模块源码不变。

## 接口与任务

1. `comparison_baselines/digital_link.py`：真实 Sionna 2.2.0 PyTorch 5G LDPC 编码、Gray QAM、AWGN、软解调和 LDPC 解码。`LinkConfig(k=4096,n=6144,bits_per_symbol=4,decoder_iterations=20,batch_size=32,device='cuda')`；`transmit_file(source,destination,*,config,snr_db,seed,noiseless=False,progress=None)->dict`。文件封装包含长度和 SHA256；每个信息块内包含 CRC32，块填充、文件头和校验均计费。输出 `status=success/failed`、`total_channel_uses`、分组和错误统计；仅成功时输出完整文件。分批执行，固定种子，禁止理想传输替代。添加真实 CPU 小文件、无噪声、低 SNR 失败和计费测试。
2. `comparison_baselines/mesongs_adapter.py`：独立环境中调用固定上游源码，CLI `encode --source --ply --out --repo --config c1 --iterations 0 --resolution 2 --images images --device cuda` 与 `decode --archive --ply --repo`。编码输出 `out/bins.zip`、`out/encode.json`；解码只依赖收到的压缩包与公共算法，输出标准 SH PLY。配置使用官方 JSON 的实际数值，记录配置和版本。修复适配层的零剪枝边界，隔离上游硬编码路径；不要修改子模块。定向测试及静态检查。
3. `comparison_baselines/evaluate.py`：统一 CUDA 评估 CLI `--source --source-ply --received-ply --out --resolution 2 --images images --views 32 --image-views 2 [--no-lpips]`。复用主项目 `read_ply/load_cameras/render/image_metrics/LPIPSEvaluator`，同一相机同时对照片和源 PLY 计分，记录身份信息、每视角指标和平均指标，最多保存两张对照图。仅从收到的 PLY 恢复场景；源 PLY 仅作为评估教师。CPU 验证指标聚合和视角选择，GPU 检查在服务器执行。
4. 主进程实现 `comparison_baselines/run.py`、文件无损打包、仓库准备/环境安装/启动 shell、文档。主运行接口指定源 PLY 和照片路径，独立选择数字链路 Python、MesonGS Python、主项目评估 Python。MaskGaussian 组对 PLY 做无损流式压缩；MesonGS 组先编码真实压缩包。默认 10 dB、4 trials，逐组运行。只在收到的文件校验成功后解包/解压并统一评估。记录所有试验的成功率和成本，失败试验不生成伪造画质；成功条件指标明确标注。输出紧凑 JSON/CSV/代表图；大中间文件放 scratch，可选保留。

## 验证与交付

- 测试先失败再实现，覆盖真实文件字节恢复、低 SNR 错误、CRC/SHA、所有开销、路径和输入身份、解码流程与评估聚合。
- Sionna 与 MesonGS 使用独立环境，避免替换现有 maskgs 的 torch/numpy/CUDA 扩展。
- 只运行本次定向测试，不运行全仓回归。完成一次独立代码审查后修复有效问题。
- 本地 CPU 能执行的真实 LDPC/QAM 小文件验证必须执行；完整 CUDA 场景压缩、渲染执行状况如实报告。
- 仅提交本轮文件，推送 GitHub，给出仓库初始化、环境安装和服务器启动命令。

## 实施记录

- 已完成数字文件传输、MesonGS 适配、统一评估、无损打包、总入口、依赖环境与启动脚本。
- 已完成独立审查并修复：官方坐标端点打包范围、旧记录 settings/protocol 兼容、保存相机名单继承、照片内容身份、SNR 路径唯一性、base 环境路径和中断恢复窗口。
- 数字链路使用公开固定计数器加扰，并报告实际发送功率；文件头、CRC 和填充全部经过真实信道编码。
- 每次 trial 均真实传输；对 SHA256 相同且成功接收的数字包复用确定的解压画质，记录指标来源，减少重复渲染。
- 本地最终定向验证：49 项测试、34 项子测试通过；含真实 CPU LDPC/QAM、官方 save_npz 序列化契约与解码，以及使用替代 CUDA 渲染器的 CPU 流程集成。
- 完整场景的 CUDA 编码、原生光栅化与 LPIPS 权重自检交由服务器启动流程执行；本地未宣称完成该硬件验证。
