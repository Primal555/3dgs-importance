# Modular Gaussian JSCC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增独立 Truck 实验，将语义表示与信道恢复分为四个可独立训练的模块，保留可靠 XYZ 解码辅助，并能区分表示、有限预算和噪声造成的失真。

**Architecture:** SemanticEncoder 输出未做功率归一化的逐点 y；ChannelEncoder/ChannelDecoder 处理 SNR、4/12/24 前缀和接收掩码；SemanticDecoder 用 y 或 y_hat、已交付 XYZ 和 active 恢复属性。clean 路径绕过通信模块，四阶段通过明确的参数冻结和独立 optimizer 控制训练。

**Tech Stack:** Python、PyTorch、unittest、现有 Gaussian 光栅器与 replay、Bash；不增加 PointNet++ 或第三方训练依赖。

**Spec:** `docs/superpowers/specs/2026-10-12-modular-gaussian-jscc-design.md`（包含用户确认的 XYZ 辅助要求）。

## Global Constraints

- 独立 `architecture=modular_jscc_v1`；保留旧 learned_joint 的配置序列化、model_id、脚本与 checkpoint 行为。
- 默认随机初始化；SOURCE_RUN 仅提供场景路径、配置及相机划分，不提供权重。
- 默认 latent_dim=48、hidden=96、depth=2、grid_dim=16、levels=(4,8)、block_size=256、batch=64、window=32、heads=4。
- rates=(0,4,12,24)、power_floor=0.01、每轴16bit量化、delta_zlib等级6、坐标可靠交付假设可见。
- 默认 decoder_geometry=xyz，保留现有零初始化坐标嵌入与相对位置偏置；none 是显式对照，不新增邻域搜索。
- 默认四阶段更新数2000/1000/1000/1000，模块学习率均1e-4，默认不裁剪、无隐含调度。
- 局部响应属性目标与原PLY渲染MSE保持不变；latent MSE只诊断，不引入新加权目标。
- 不训练分配器、不随机删点、不加通信惩罚；q0仅是固定关闭点或padding。
- 默认10dB AWGN，支持显式训练SNR列表；每个真实更新只抽一次，replay保持相同SNR与噪声。
- 输出统一到一个目录；每500步保存/验证，阶段起止额外记录；仅初始与最终保存2个相机的clean和10dB q1/q2/q3/mixed图像。
- 首轮本机只能验证接口、反向、传输与CPU合成渲染；不能宣称已验证Truck CUDA画质。

## Review Focus

1. 旧checkpoint没有新配置字段：必须加载原架构并保持原model_id（Task 3）。
2. padding、q0和不满窗口：不得污染保留点或产生全mask softmax NaN（Task 2）。
3. 冻结语义解码器：参数不更新，但仍向信道解码器回传梯度（Task 4）。
4. 阶段边界中断：resume不得重复或跳过更新，不能丢失Adam动量（Task 5）。
5. 接收端独立运行与有限预算无噪声：不得读取源PLY/y，且无噪声仍执行通信模块（Tasks 2、3、6）。

## 文件划分

- `gaussian_jscc/modular_config.py`：独立配置与校验。
- `gaussian_jscc/modular_core.py`：四个网络模块，禁止训练/文件读写逻辑。
- `gaussian_jscc/modular_codec.py`：模块组合、clean/通信路径、packed接口与诊断。
- `gaussian_jscc/model_factory.py`：显式配置和模型分发。
- `gaussian_jscc/modular_training.py`：阶段冻结、batch loss/replay适配、参数观测。
- `gaussian_jscc/modular_checkpoint.py`：实验训练状态、RNG与精确续训。
- `gaussian_jscc/modular_evaluation.py`：多路径验证和最终SNR扫描。
- `gaussian_jscc/modular_experiment.py`：Truck输入、训练调度、CLI、输出。
- `gaussian_jscc/modular_plots.py`：按阶段绘图与PSNR-SNR对照。
- `scripts/test_modular_codec_truck.sh`、`docs/modular_codec_truck.md`：启动与说明。
- 共享文件只调整 `transport.py`、`optimization.py` 的显式分发；旧配置/骨干保持不变。

### Task 1: 独立配置与语义编码器

**Files:** Create `modular_config.py`、`modular_core.py`；Test `tests/test_modular_codec.py`。

**Interfaces:** `ModularCodecConfig.to_dict()/from_dict(values)`、`attr_dim` 与现有helper兼容；`SemanticEncoder(cfg).forward(features: Tensor[B,N,3+attr_dim], xyz: Tensor[B,N,3], active: BoolTensor[B,N]) -> Tensor[B,N,latent_dim]`。

- [ ] 写失败测试 `test_config_defaults_and_invalid_dimensions`：默认48/96/2/32/4、4/12/24、xyz；拒绝latent_dim<=0、不兼容hidden/heads和非可靠16bit坐标。
- [ ] 写失败测试 `test_semantic_encoder_has_no_channel_conditions`：接口不接收q/SNR；给定features/active重复输出一致，padding输出为0；输出维度可配置，输出没有逐点单位功率约束。
- [ ] 运行 `python -m unittest discover -s tests -p test_modular_codec.py -v`，确认失败来自缺失接口。
- [ ] 实现配置和编码器：沿用五组嵌入、fuse、两个ContextBlock；使用可训练96维常量condition控制门控，末层线性96->latent_dim。排除tier/SNR输入和symbol功率归一化。
- [ ] 同一命令验证通过，提交本任务限定文件。

### Task 2: 信道模块、语义解码器与组合接口

**Files:** Extend `modular_core.py`；Create `modular_codec.py`；Test `tests/test_modular_codec.py`。

**Interfaces:** `ChannelEncoder.forward(y, active, snr)->full[B,N,48]`（含独立增量归一化）；`ChannelDecoder.forward(received[B,N,48], receive_mask[B,N,48], q[B,N], snr)->y_hat[B,N,L]`；`SemanticDecoder.forward(y_hat, active, delivered_xyz=None)->attributes[B,N,attr_dim]`。

`ModularGaussianCodec(cfg)` 注册attr_mean/attr_std及四个顶层模块；提供与现有 transport/training 兼容的 `encode(features,xyz,q,snr)`、`encode_full(features,xyz,retained,snr)`、`decode(symbols,q,snr,return_seed=False,delivered_xyz=None)`、`forward(...)`、`forward_tier_batches(...,paired_noise=False)`。新增 `forward_clean_batches(features,xyz,active)` 返回固定量化XYZ拼接属性；诊断用 `encode_representation(features,xyz,active)`，接收端decode不调用它。

- [ ] 写失败测试 `test_clean_bypasses_channel_but_noiseless_executes_it`：用调用计数验证clean不调用信道模块、有限预算none调用两模块；不能靠mode命名通过测试。
- [ ] 写失败测试 `test_progressive_prefix_and_power`：固定输入/active/SNR，任意正档位图前缀相同；4/8/12增量段平均复能量<=1，零能量输入无NaN；mask区别未收到与数值为0。
- [ ] 写失败测试 `test_xyz_auxiliary_is_delivered_and_detached`：有辅助时用量化坐标，坐标无梯度、无XYZ输出头；激活坐标分支后改变delivered_xyz可影响属性，输出XYZ仍精确等于交付值。clean/通信用相同量化输入。
- [ ] 写失败测试 `test_q0_padding_tail_and_missing_side_information`：q0整窗口、全q0块、13点尾块均有限且关闭点全0；缺坐标、形状不匹配和活跃点非有限坐标明确报错；缺失后缀及padding坐标扰动不影响活跃输出。
- [ ] 先运行目标测试确认失败，再实现逐点MLP与原LocalFeatureBlock语义解码骨干；仅信道解码器使用q/SNR条件，语义解码器保留序号编码及可选XYZ辅助。
- [ ] 跑 `test_modular_codec.py` 和 `test_xyz_context.py`；通过后提交。

### Task 3: 模型工厂、checkpoint与独立接收

**Files:** Create `model_factory.py`；Modify `transport.py` 的load/receive配置分发；Test `tests/test_modular_transport.py`。

**Interfaces:** `config_from_dict(values)->CodecConfig|ModularCodecConfig`、`build_codec(cfg)->GaussianCodec|ModularGaussianCodec`；旧 `CodecConfig.from_dict` 不接受新架构。transport checkpoint继续显式保存config/state_dict；训练状态由Task 5扩展。

- [ ] 写失败测试 `test_legacy_identity_and_modular_roundtrip`：旧配置序列化与已有夹具model_id不变；模块化模型保存/载入配置、输出、model_id一致；未知架构与不匹配包明确拒绝。
- [ ] 写失败测试 `test_receiver_only_without_source`：实际transmit后销毁发送端对象和源文件，在新接收对象只加载checkpoint/packet的条件下解码；q0空洞/尾块/quantizedXYZ正确，payload等于sum(rates[q])。
- [ ] 先确认失败，再实现工厂分发并更新transport中硬编码CodecConfig的解析点；保留完整XYZ与元数据计费，明确decoder使用侧信息而非源数据。
- [ ] 运行 `python -m unittest discover -s tests -p test_modular_transport.py -v` 和旧 `test_gaussian_jscc.py`、`test_xyz_context.py`；通过后提交。

### Task 4: 四阶段冻结、目标与replay

**Files:** Create `modular_training.py`；Modify `optimization.py` 顶层四模块参数分类；Test `tests/test_modular_training.py`。

**Interfaces:** `configure_phase(model, phase, module_lrs)->Optimizer`；`bootstrap_step(model,features,q,geometry,snr,kind,phase,directions)->(loss,diagnostics)`；`render_step(model,batches,geometry,snr,kind,task,phase)->(loss,diagnostics)`。

- [ ] 写失败测试 `test_phase_trainable_sets_and_frozen_snapshots`：2000/1000/1000/1000阶段对应spec参数集合；运行更新后冻结参数逐bit不变；阶段切换新建optimizer。
- [ ] 写失败测试 `test_frozen_decoder_keeps_channel_gradient`：channel_bootstrap中semantic_decoder参数无梯度/更新，但channel_decoder与channel_encoder均得到非零梯度。不可在整个decoder外包no_grad。
- [ ] 写失败测试 `test_replay_matches_direct_all_phases`：CPU合成渲染、配对AWGN，直接反向与现有full_scene_step(batch_forward=...)的replay梯度一致；冻结/解冻后分别复验。
- [ ] 写失败测试 `test_loss_and_latent_diagnostics_are_separate`：属性阶段只local_response_loss，渲染阶段只RGB MSE；改变诊断latent误差不改变objective；非有限梯度在optimizer.step前报错。
- [ ] 确认失败，再复用local_response_loss和full_scene_step（attr_weight=0）；通过batch_forward选择clean或通信解码，不复制replay算法。
- [ ] 记录semantic_encoder/channel_encoder/channel_decoder/semantic_decoder梯度与更新范数；旧learned分组保持原结果。
- [ ] 运行 `test_modular_training.py`、`test_render_first.py`、`test_local_response.py`；通过后提交。

### Task 5: 精确resume与语义checkpoint分叉

**Files:** Create `modular_checkpoint.py`；Test `tests/test_modular_checkpoint.py`。

**Interfaces:** `save_training_checkpoint(path,model,optimizer,progress,rng,scene_record)`；`restore_training_checkpoint(path,device)->(model,optimizer,progress,rng,scene_record)`；`initialize_channel_run(path,device)->model` 仅接受完成semantic_render的模块化checkpoint，并载入语义权重/统计、新建通信模块。

- [ ] 写失败测试 `test_resume_preserves_adam_rng_and_schedule`：小样本不中断8步与第4步保存续训最终权重、Adam、布局、SNR序列一致；Python/NumPy/torch/CUDA可用时RNG均保存。
- [ ] 写失败测试 `test_phase_boundary_and_zero_length_stages`：阶段末尾续训不重复/跳过；零步阶段明确跳过，计数仍准确；通信尚未训练标记false，只有实际更新后true。
- [ ] 写失败测试 `test_semantic_init_not_resume`：分叉仅语义权重和统计一致，通信模块新初始化，Adam和计数重置；错误架构/场景指纹/相机划分禁止静默接受。
- [ ] 确认失败，实现原子checkpoint保存和按阶段恢复optimizer；在manifest中显式记录resume与init来源。
- [ ] 运行 `test_modular_checkpoint.py`，通过后提交。

### Task 6: 多路径验证、最终评价与通信代价

**Files:** Create `modular_evaluation.py`；Test `tests/test_modular_evaluation.py`。

**Interfaces:** `evaluate_paths(model,scene,*,step,phase,snrs,trials,seed,save_images,out)->list[dict]`；scene复用现有raw/geometry/groups/group_ids/cameras/reference协议，不含learned mask。

- [ ] 写失败测试 `test_validation_is_rng_and_parameter_neutral`：固定四相机/mixed图，验证前后参数、训练模式、RNG一致；配对噪声在档位之间按固定槽位抽取。
- [ ] 写失败测试 `test_path_labels_and_cost_scope`：clean不报4/12/24 payload或虚构传输成本；finite-noiseless仍经过两信道模块；输出channel_trained、actual_snr、tx_condition_snr及量化XYZ成本假设。
- [ ] 写失败测试 `test_current_clean_remeasured_after_channel_render`：更改semantic_decoder后同轮clean重新测量；不得读取旧阶段缓存分数。
- [ ] 写失败测试 `test_final_all_views_two_trials_and_endpoint_images`：最终测试覆盖全部留出相机、至少两噪声试次、SNR 0/5/10/15/20；仅初始最终10dB与clean保存2相机图，mixed不缺失。
- [ ] 确认失败，复用image_metrics/save_panel、preserved_rng与PositionCostMeter；坐标流不按每视角/试次重复压缩。记录属性误差、latent诊断、PSNR/SSIM及试次离散程度。
- [ ] 运行 `test_modular_evaluation.py`；通过后提交。

### Task 7: Truck完整实验入口与可选direct对照

**Files:** Create `modular_experiment.py`、`scripts/test_modular_codec_truck.sh`、`docs/modular_codec_truck.md`；Test `tests/test_modular_experiment.py`。

**Interfaces:** CLI `python -m gaussian_jscc.modular_experiment --run SOURCE_RUN --out NEW_OUT`；可选 `--manifest`、四个 `--*-steps`、四个 `--*-lr`、`--latent-dim`、`--decoder-geometry xyz|none`、`--train-snrs`、`--resume`、`--init-semantic`、`--with-direct-control`（resume与init互斥）。

- [ ] 写失败测试 `test_four_phase_cpu_experiment_and_log_integrity`：合成小场景每阶段2次更新，完成计数与训练日志一致；四模块观测齐全，冻结快照/阶段耗时齐全；CPU显存明确不可用而非虚构0。
- [ ] 写失败测试 `test_random_init_source_run_is_metadata_only`：source_run checkpoint不可读时仍能根据路径/配置启动；确认未加载旧权重；scene身份和相机分割与reference一致。
- [ ] 写失败测试 `test_multi_snr_one_draw_per_update`：列表均匀抽样；同一步所有块/replay同SNR，resume可复现。默认只10dB。
- [ ] 写失败测试 `test_cli_paths_and_outputs`：未提供GPU时脚本明确报错，CUDA_VISIBLE_DEVICES继承；输入路径存在性先检查；新out冲突拒绝，resume不覆盖已有记录。
- [ ] 确认失败，实现Truck-only调度及一目录布局：checkpoints、metrics、charts、images，保存latest/四阶段端点/final；progress.json与complete.json区别明确。
- [ ] 可选direct对照从头3000属性+2000渲染，使用相同decoder_geometry、场景/预算/相机划分；总5000更新，报告参数/耗时而不声称FLOP匹配。
- [ ] 运行 `test_modular_experiment.py` 与 `bash -n scripts/test_modular_codec_truck.sh`，通过后提交。

### Task 8: 图表、回归、审查与推送

**Files:** Create `modular_plots.py`；Test `tests/test_modular_plots.py`；补充 `docs/modular_codec_truck.md`。

**Interfaces:** `plot_run(out: Path)->list[Path]` 从metrics读取，不重新渲染/压缩场景。

- [ ] 写失败测试 `test_phase_separated_curves_and_snr_gaps`：四阶段目标分开，clean/有限无噪声/AWGN明确区分；各档位PSNR-SNR与当前clean参考正确；空阶段和缺最终评价不崩溃。
- [ ] 确认失败，实现绘图、各模块梯度与更新、损失来源和预算说明；文档给出nohup、日志、resume及语义分叉启动示例。
- [ ] 跑全部 `test_modular*.py`，然后完整 `python -m unittest discover -s tests -v`；记录已有失败与新失败，未执行CUDA验证明确披露。
- [ ] 使用CPU合成渲染完成全四阶段/精确续训/真实receiver packet小测试，检查保存图可读；不把此测试包装为真实Truck性能结果。
- [ ] 逐条检查spec覆盖：四模块边界、坐标辅助、阶段冻结、replay、接收独立性、RNG、计费、最终测试视角、日志与图片精简。
- [ ] 检查git diff，保留用户无关修改；只提交本实验相关文件。按用户既定要求推送GitHub，确认远端提交一致；若网络失败如实报告，不声称已推送。

## 执行方式与本轮状态

推荐由主代理在当前会话逐任务执行（superpowers:executing-plans），任务接口依赖较紧；不必为本次重构拆分多代理。实施时按using-git-worktrees确定隔离方式，执行后进行完整代码自审；需要独立review时另遵循review技能。

当前只更新设计并编写本计划，没有产品代码变更。根据brainstorming/writing-plans流程，用户审核本计划并确认执行方式后，才开始上述实现与测试。
