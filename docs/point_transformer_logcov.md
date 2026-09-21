# 编码端几何邻域 Transformer 实验

这是上一轮 `multiscale_self` 的**编码器单变量结构升级**，不是完整移植 PTv1/PTv2/PTv3，也不预先承诺画质提升。

## 改了什么

| 项目 | 上一轮编码端 | 本轮编码端 |
|---|---|---|
| 邻域 | Morton 序列固定/移位窗口 | 处理块内按源 XYZ 搜索最多 16 个有效近邻，包含自身 |
| 注意力 | QK 点积 + 相对位置分数偏置 | Q−K 与相对位置共同生成分组注意力权重 |
| 聚合内容 | V 特征 | V + 学习的相对位置特征 |
| 尺度 | 原始点、4 点池化、16 点池化 | 保留三个尺度，每个尺度都用几何近邻注意力 |

相对位置使用邻域半径归一化，附加归一化距离；这些都是发送端计算特征，不是额外发送的坐标。权重在邻居维归一化，各 head/group 分别计算。保留 pre-norm、残差和 FFN。

4/16 点池化仍按固定 Morton 槽位进行，并非新聚类；kNN 仅在对应处理块/池化块内搜索，不能跨处理块。细尺度单层对无距离并列的重排是等变的，但**整个带槽位池化的系统不宣称任意点排列不变**。并列距离时 kNN 可任选同距点。

自身符号路径、Context 门控、接收端全部结构、logcov 输出、`spatial_logcov_v1`、0/8/16/32 复符号预算和功率归一化均不改动。没有 XYZ 旁路，没有新增通信元数据，没有二阶段渲染训练。q0 和 padding 不参与邻域选择/聚合。

通过 `--encoder-attention geometric_point --encoder-neighbors 16` 显式启用，要求 `--context-mode multiscale_self --architecture learned_split_logcov`。原有 checkpoint 默认仍使用旧注意力；不能把旧窗口权重当作新编码器初始化，配置不匹配会拒绝。

## 成本与边界

默认 SH3 / hidden96 / depth2 参数量从 1,542,199 增至 1,698,615（约 +10.1%）。块内 kNN 距离矩阵为 O(B*N²)，注意力激活约 O(B*N*K*hidden)；N 默认 256，不会构造全场景两两距离。源邻接关系不依赖可训练参数，不对离散近邻搜索求导。

显存与运行速度需要服务器实测，不能沿用上一轮 64 blocks 的结论。默认先用 32 blocks；OOM 时降低到 16。局部自身路径、几何解码头和原目标仍在，因此本次**并未解决或保证解决 XYZ 梯度偏大和细粒度位置精度不足**，仅检验更直接的几何编码是否有效。

## 后台启动

确认 GPU 2 空闲后运行（否则修改卡号）。不使用旧 checkpoint：

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git pull --ff-only --no-recurse-submodules origin main
export PYTHON_BIN="$(command -v python)"
export PLY="$PWD/output/truck_mask_0005/point_cloud/iteration_30000/point_cloud.ply"
export SCENE="$PWD/data/tandt_db/tandt/truck"
export CUDA_VISIBLE_DEVICES=2
export BOOTSTRAP_STEPS=5000 SAVE_EVERY=500 BLOCKS_PER_BATCH=32
export ENCODER_NEIGHBORS=16 LR=0.0002 SNR=10 CHANNEL=awgn
export RENDER_HISTORY=1 RENDER_VIEWS=8 RESOLUTION=4 VALIDATION_TRIALS=2
mkdir -p output
OUT="$PWD/output/truck_point_transformer_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/test_point_transformer_logcov.sh "$OUT" > "${OUT}.log" 2>&1 < /dev/null &
echo $! > "${OUT}.pid"
printf '结果：%s\n日志：%s\n' "$OUT" "${OUT}.log"
tail -f "${OUT}.log"
```

训练完成后才自动渲染各 500 步 checkpoint，统一保存在 `$OUT/render_history`。Ctrl+C 退出 `tail` 不会停止后台训练。要做严格结构对比，应让旧/新实验采用相同 batch、训练步数和评估条件，多种子复查；本轮 CPU 检查仅证明功能和短拟合可运行，不是 Truck PSNR 改善的证据。

重点观察：相同训练步数的 XYZ 相对半径误差、q1/q2/q3/mixed 渲染 PSNR/SSIM、原生形状和外观误差、XYZ head 梯度与实际更新，以及训练时间/显存。不能只凭总 loss 下降判定成功。
