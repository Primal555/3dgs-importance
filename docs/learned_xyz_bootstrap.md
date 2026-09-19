# 无逐点坐标侧流的第一阶段测试

入口：`scripts/test_learned_xyz_bootstrap.sh`。它与已验证的量化坐标两阶段入口分开：
强制随机权重、`position-delivery=learned`、`spatial-response`，只执行5000步bootstrap；
不接收相机、不调用场景渲染、不执行render/joint，不读取INIT。
旧的 `test_local_response.sh` 和损失不变。

## 信息流与通信口径

发送端仍可使用原始XYZ做Morton排序、局部网格和特征编码。
接收端仍是既有 `learned_joint` decoder：只接收有噪声符号、档位、SNR和固定块内序号，
XYZ与其他属性都由网络输出。没有位置seed、手工位置波形、预测中心替换或逐点坐标侧流。
每Gaussian预算仍为0/8/16/32复符号；位置和属性共同使用这份预算。
全局bbox、属性统计/模型、档位和顺序协议仍沿用现有假设，不能称为“零辅助信息”。
teacher位置只用于训练目标，不进入decoder。全局bbox对坐标误差做等比例的对角线归一化。

## 新损失：保留中心位移的解析多尺度响应

旧local-response把两个中心对齐，不能监督XYZ。新目标在随机正交观察方向上，
将真实中心差投影为二维位移，同时投影Gaussian协方差；不把预测中心对齐到teacher。
对每个二维协方差加 `sigma^2 I`，sigma默认是场景bbox对角线的
`0.5, 0.125, 0.03125, 0.0078125`。各尺度同时参与，不做隐式阶段切换。

采用单位L2范数的Gaussian响应，避免小点的误差因积分面积小而消失。
两个响应的内积有解析式：

`K = 2 * (det(A)*det(B))^(1/4) / sqrt(det(A+B)) * exp(-delta^T(A+B)^(-1)delta/2)`。

每个通道的积分平方误差为 `ap^2 + at^2 - 2*ap*at*K`。
七个通道分别是：幅值恒为1的几何响应、黑背景RGB、白背景相对背景的RGB差值。
单位几何通道防止网络用降低透明度或颜色的方式掩盖位置误差。
损失对点、视角、尺度、七通道等权平均，不再逐属性配置0.25等权重。
**通道选择、单位L2归一化和带宽仍是人为的实验设计，不是无超参数或已验证最优的损失。**
它不是真实像素MSE，忽略遮挡、透视与场景合成。宽尺度能在常见冷启动偏差下提供
吸引梯度，但极端远离时仍会饱和；细尺度也不保证达到最终渲染所需的位置精度。
投影协方差沿用相对数值下限，解析2x2 Cholesky使用float64，codec仍为float32。

## 服务器启动

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance || exit 1
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main || exit 1
OUT="$PWD/output/truck_learned_xyz_bootstrap_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/output"
CUDA_VISIBLE_DEVICES=2 PYTHON_BIN="$(command -v python)" \
BOOTSTRAP_STEPS=5000 BLOCKS_PER_BATCH=32 LR=2e-4 \
nohup bash scripts/test_learned_xyz_bootstrap.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

GPU需选择空闲卡。可设置CHANNEL=none对照，但默认仍是10 dB AWGN。
带宽可通过CLI `--spatial-bandwidths ...`显式修改，脚本保留默认便于复现。

## 观察什么

- `training.json`：损失定义、带宽、通信假设、固定验证块、是否与训练块重叠。
- `loss.jsonl`：总损失、各尺度响应、geometry/appearance响应、XYZ世界单位RMSE、
  bbox对角线归一化RMSE、点距离中位数/P95，位置头梯度及参数更新。
- `bootstrap_validation.jsonl`：q1/q2/q3/mixed固定种子验证，每次默认16块、2次噪声。
  指标为各块/试验等权平均，RMSE不是全场景汇总；按点分位数也是块内分位数均值。
- `codec_best_bootstrap.pt`：按四布局平均空间响应损失选择，**不是最佳渲染模型**。
- `codec_end_bootstrap.pt` / `codec.pt`：最后一步；selection.json给出最佳步数。
- `charts/`：训练loss/梯度和位置验证曲线；没有场景恢复图是正常的，因为未加载相机。

判断时必须同时看held-out块的位置误差、局部外观响应和xyz_head梯度，不能用总loss
下降替代位置恢复证据，也不能由本测试宣称渲染质量、跨场景或多SNR能力已经解决。

## 本机检查

CPU下验证了相同输入近零损失、位移产生朝目标的梯度、低透明度不屏蔽位置梯度、
各输出头参与反传、AWGN短优化、无相机的CLI流程和独立脚本参数隔离。
另用真实Truck的883438点作为数据池，以hidden32/depth1、每步2个128点块、
4个固定验证块、10 dB AWGN运行120步；q1/q2/q3的验证XYZ NRMSE分别从
0.10149/0.12702/0.15936降到0.07510/0.07485/0.07369。
这是缩小网络的梯度/运行检查，剩余误差很大，不代表位置精度达标或已改善渲染。
本机无可用CUDA，默认服务器网络的显存、耗时与收敛需要实际验证。
