# 关系收益网络实验

该版本（`relation_benefit_context_v2`）固定现有 0/8/24/48 编解码器，学习高斯之间的替代与遮挡关系。服务器先完成源 PLY 与全 q1/q2/q3、混合背景的 finite-noiseless 属性重建，随后对四种背景各执行一次条件前向，等权聚合有符号失真分量，再一次选出全部高斯的四档。训练与部署采用相同 bank 输入。

## 网络与输入

128 维共享编码，两层三维邻域分组向量注意力，一层投影关系注意力，以及四候选交互。每点三维邻居16个，每视角投影邻居16个。投影构图覆盖源与全部候选的包围圆并分配前后景邻居，邻接关系可以跨编码块。局部两跳闭包在分批时完整保留。

输入包含有符号属性／SH 差、局部与块统计、源投影、可见性、源图像与 q2 残差 profile、候选投影变化。q0 保留原始覆盖描述并设置存在性0。共同头与中心化档间头分别输出 MSE、结构项，无 Sigmoid 或强制档位单调。

SH3 的基础维度为 node379、candidate120、view186；每背景增加 node126、view78 和投影几何7维。背景编码参与空间关系与投影关系推理，归一化在训练点、全部背景上拟合并固定。LayerNorm、相对／绝对尺度下限及 asinh 输入用于控制极端属性幅值。

## 教师与数据边界

画质参考始终是源 PLY 渲染。采集背景依次轮换全 q1、全 q2、全 q3、mixed。mixed 按原行号和固定种子生成 q0/q1/q2/q3，概率为0.1/0.3/0.3/0.3。改变一个点时重算整个编码块，包含 q0 的语义重编码、邻居解码变化与功率归一化。四档共用噪声槽位，背景之间也匹配槽位；两个独立固定试验取均值。局部教师输出 float64 的 `[ΔMSE, lambda_s*Δstructure]/D_ref`，相对同背景 q0。CUDA 启动预检在四背景下分别比较两组干预与完整光栅器，另含候选进入新 tile 的确定性检查。

默认从训练相机中固定选4个代表视角，网络与教师使用同一视角和均匀权重。单点标签反映该视角集合下的干预价值；最终32个测试视角的混合场景画质独立记录。可用 `--window-views` 调整预处理与监督视角数；缓存、教师和训练身份随之改变。输入图像参考来自源 PLY，测试照片只用于最终指标。

候选输入始终无实际信道噪声。教师固定种子与实际测试种子分开；预测网络不读取未来接收噪声。初始50窗口只用训练标签拟合固定输出尺度。训练／验证点固定隔离，验证标签不进入回放或尺度拟合。

## 训练与计费

默认700窗口、每窗口64个完整标签组，前50窗口拟合尺度，后650窗口每窗口64次更新，总41,600次网络更新。每次8组。四背景各175窗口，最多44,800个背景—点标签；每背景从不同点游标开始无放回轮转。全部历史标签持续保留，以(context, source)作为身份去重。每次更新均匀选择背景，再在该背景的全部历史标签内均匀抽样；按背景维护索引，抽样复杂度仅随批量增长。默认128个独立验证点在四背景下形成512组固定验证标签，同时报告逐背景与四背景均值决策指标。共同项、档间项采用固定尺度 SmoothL1，方向项沿用 RD-gap Bernoulli KL。日志记录整个窗口的损失均值、梯度均值／最大值及裁剪比例。

目标使用 `D/D_ref + beta*C/C_ref`。`C_ref` 是统一数字侧信息账本下全q3的总复信道使用；逐点使用全q3平均侧信息成本作为边际代理。实际混合场景重新编码／加噪／渲染，按压缩后的三路侧信息字节独立计算 LDPC／16QAM 头、CRC、补齐开销，得到真实部署 RD。记录同时保留代理范围，单点预测之和不作为真实场景失真。

## 启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

RUN="$PWD/output/successive_mip360_gpu1_20261023_142948/training"
GPU=3
OUT="$PWD/output/truck_relation_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_relation_benefit.sh \
  "$RUN" truck "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```

本版新建 OUT，从头训练关系网络。`--phase train` 只训练并输出网络；默认自动训练、四条件推理、一次选档和最终评价。恢复时使用相同 OUT 及参数并设置 `RESUME=1`。`--windows` 可增加目标窗口；扩训时旧部署产物移入 history，新完成标记与结果保持同一步数。`--phase predict` 配合 `RESUME=1` 对已完成的同场景训练执行最终预测。

完整历史回放版本可自动迁移此前version2检查点，前提是全部唯一历史标签仍保留。迁移保留输入bank、网络、优化器和既有标签，写入 `replay_migration.json`。已经淘汰标签的旧检查点会明确拒绝迁移。默认配置前128窗口尚未超出原8192组容量；当前仍在预处理时，可等第一个 `checkpoints/latest/state.pt` 保存后停止旧进程，拉取新版并用原OUT恢复，以复用已经完成的构图。

## 新场景直接预测

```bash
CUDA_VISIBLE_DEVICES=3 python -u -m gaussian_jscc.relation_benefit_predict \
  --run "$RUN" --network "$OUT" \
  --manifest configs/multiscene_tandt_db.json --scene drjohnson \
  --out "$PWD/output/drjohnson_relation_$(date +%Y%m%d_%H%M%S)"
```

新场景仅预处理与前向推理，不构造教师、优化器或拟合归一化。沿用网络的 D_ref、lambda_s 及全部尺度；新场景重新计算全q3通信成本。源场景角色及预处理／测试相机身份随结果保存。该入口验证零更新部署能力，画质收益由实际测试结果判定。

## 输出与运行开销

- `metrics/loss.jsonl`：标签覆盖、整窗训练统计与耗时。
- `metrics/benefit_validation.jsonl`：四档／三档 regret、固定档位基线、共同和档间误差。
- `deployment/inference.json`：分层前向计数、耗时、档位比例和边际成本定义。
- `review_summary.json`：最终选档与全q1/q2/q3的真实 PSNR、SSIM、LPIPS、总复信道使用、RD。
- `review.tar.gz`：紧凑指标和最终两视角对照图／档位贡献图，排除缓存与检查点。
- `cache/bank.pt`、`checkpoints/latest` 与 `checkpoints/final`：服务器上精确续训需要的文件。bank 为共享 CPU 输入，标签回放只保存索引和小型目标。

SH3、4视角、百万点的持久 bank 约14.1GiB CPU存储；Truck 711,691点约10.0GiB，构建与教师缓存另需临时内存。源属性、三档候选重建和关系图共享，mixed 仅额外执行一次完整解码。教师GPU同时仅保留当前背景的捕获，切换后复用CPU背景解码缓存。推理复用静态节点与候选编码，并以逐层CPU缓存运行各条件图。投影图在大面积密集重叠时广相查询可能出现平方级开销，预处理日志记录分阶段进度和覆盖规模。

构图日志约每5秒报告当前视角、完成根节点、阶段、桶／查询进展、累计候选命中量和耗时。ETA仅根据当前视角已完成根节点估计；各视角覆盖密度不同，不能将其作为整个训练的剩余时间。完整历史标签的float64目标张量最多约8.2MiB，另有索引及Python对象开销。

本地验证使用真实小型 codec 与 CPU 数学光栅；真实 CUDA 认证在服务器启动时完成。

???????????82????1?CUDA????????GPU????????CLI???????????????????????????????CUDA???????????????????
