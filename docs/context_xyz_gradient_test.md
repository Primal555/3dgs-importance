# Context → XYZ 梯度短程对照

目的：定位属性 Context 的位置反向通路是否导致位置分支梯度过大或方向冲突。不是调参搜索，也不同时改变位置损失、坐标表示或裁剪阈值。

## 两组只差一个开关

- `attached`：现有行为，属性 Context 可以向 XYZ 反向传播。
- `detached`：只在属性解码器的 GridContext 输入处使用 `xyz.detach()`；返回位置、位置重建损失和渲染器的直接位置梯度均保留。
- 同一 checkpoint、重新建立的 Adam；每组 200 步局部属性训练 + 100 步全场景渲染训练，顺序执行，不是双卡并行。分别采用 5e-5 / 1e-5 学习率，全局裁剪阈值 1。
- 保留 checkpoint 的 loss profile、各分项权重、符号预算和网络结构；建议使用刚完成 4000 步的 `balanced_v2` checkpoint。若用其他 profile，`config.json` 会如实记录，不会自动迁移。
- 每步重设随机种子，保证块、档位、相机、SNR 和信道噪声抽样对应。完整档位散列写入日志，结束时检查配对。CUDA scatter 原子操作仍可能有微小非确定性；不宣称逐位确定。
- 不会覆盖初始化 checkpoint、改变 packet 格式或增加接收端坐标信息。此开关只影响反向，加载输出 checkpoint 做推理无需设置开关。继续训练若希望保留 detached 行为，必须显式设置；普通训练 CLI 不会自动采用本实验的开关。

## 运行

在 maskgs 环境和项目根目录中，用实际最新 checkpoint 替换下面路径。输出目录必须不存在；不需要预先 mkdir。

```bash
conda activate maskgs
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only origin main
INIT="/实际路径/刚完成4000步的训练目录/codec.pt"
OUT="$PWD/output/context_xyz_gradients_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/test_context_xyz_gradients.sh "$INIT" "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

GPU3 必须事先确认空闲。`Ctrl+C` 只退出 tail，不会停止后台脚本。若数据路径不同，用环境变量 PLY、SCENE 指定。全部完成会输出 `Finished paired diagnostic`。

## 输出与解释

- `config.json`：初始化文件 SHA256、完整实验参数、checkpoint 配置。
- `attached/loss.jsonl`、`detached/loss.jsonl`：每步 loss、SNR、块/相机、完整档位散列、各模块裁剪前后范数及最大绝对梯度、全局范数和裁剪因子。
- 每 50 步以及阶段首尾：`component_gradients` 记录加权位置项、其余加权属性项、渲染项的梯度范数，以及它们在全模型/几何编解码器上的两两余弦。负余弦表示该采样下方向冲突；零向量时为 null，不伪造为 0。
- `component_sum_relative_error`：分项梯度之和与正常总梯度的相对误差，应接近浮点计算误差。渲染阶段按全部 batch 累计；不是拿一个局部块冒充全场景梯度。
- `updates`：抽查步骤 Adam **实际**参数更新的 L2 范数及相对参数范数；不能用裁剪比例代替更新量。
- `largest_parameter_gradients`：抽查步骤梯度范数最大的8个具名参数，帮助从分支进一步定位到具体层。
- `fixed_evaluation.csv`、每组 `evaluation.json`：固定空间块，q1/q2/q3，无噪声及 AWGN 0/10/20 dB。固定种子，记录第0步/中途/末步位置 RMSE、bbox 对角线归一化 NRMSE、欧氏距离中位数/P95、重建 loss。RMSE 定义为 sqrt(mean(三个坐标分量的平方误差))。这些块来自训练场景，不是跨场景泛化证据；不可当成完整场景指标。
- `initial_test/`、两组 `test/`：固定两个测试视角，完整场景，q2、10 dB AWGN；同一噪声抽样。PNG 顺序记录在 `metrics.json` 的 panel_order：GT / 原始场景 / 仅位置恢复误差 / 仅属性恢复误差 / 完整恢复。指标同时包含相对真实图像和相对源 PLY 渲染，切勿混用。
- `diagnostics.png`：训练位置损失、裁剪前总梯度、固定块无噪声位置 RMSE、几何解码器实际更新曲线。训练曲线跨两阶段且随机条件不同，主要看固定评估而不是阶段交界跳变。
- `gradient_components.png`：两组几何解码器的分项梯度范数与两两余弦；零梯度的余弦不绘制，范数零值仅为对数坐标展示而绘制在1e-20。
- 每组 `codec_200.pt`、`codec_300.pt`、`codec.pt`：可独立继续评估的实验权重；summary 保存最终固定评估、测试渲染及配对验证。

需要回传：`summary.json`、`config.json`、`fixed_evaluation.csv`、两组 `loss.jsonl`，以及 `diagnostics.png` 和测试拼图。

## 判断顺序

1. 第0步两组应同输出；若不一致，先检查配对，不能解释为 detach 的训练收益。
2. detached 组的属性项 → geometry_decoder 梯度应接近 0，位置/渲染项梯度仍存在。geometry_encoder 仍与属性分支共享前端与功率归一化，不要求属性项对它为 0。
3. 对比相同 phase、SNR 和采样的模块梯度、分项余弦及实际更新，而不只数“裁剪触发几次”。高梯度本身不是梯度爆炸的充分证据。
4. 只有梯度行为改善且固定位置误差/测试渲染也改善，才能支持保留该切断；单个短程、单种子结果只是定位线索。阈值、loss 权重和位置表示本次不一起改。

## 开销与验证边界

分项诊断只在抽查步额外调用 autograd.grad，正常 backward/optimizer.step 仍各一次；渲染部分使用原 replay，最多保留一个 codec batch 的激活。诊断步显存和耗时会高于原性能基准，固定评估也有额外耗时，不能把这次每步时间当生产训练速度。

CPU 测试覆盖前向不变、切断范围、分项梯度重组、观察器不改变总梯度/随机状态及短程 CLI 输出；真实 CUDA rasterizer、显存峰值和质量收益需服务器运行确认。
