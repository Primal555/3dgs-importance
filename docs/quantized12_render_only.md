# 固定保留的纯渲染训练线

当前优先观察这条线；局部响应两阶段代码仍保留，暂不作为本次训练入口。

## 历史源码与实际实验不是同一份默认配置

- 历史提交：`9f2810e`，支持显式 XYZ 交付，尚未加入局部响应预训练。
- 两阶段实现提交：`37e4d35`；添加可选路径，没有替换 bootstrap=0 时的渲染损失。
- 有效实验：`output/truck_quantized12_render20000_20260918_143246`。
  目录虽含 20000，实际配置和日志为1000步；render_lr=1e-4；215训练视角；blocks=64。
- 历史提交的默认 render_lr 为1e-5，所以单独 checkout 旧提交、直接用默认值，
  并不等于重现这个有效实验。服务器当时的未提交改动无法仅凭Git记录完整还原。

本地归档位于 `output/code_snapshots/render_only_9f2810e/` 和同名 ZIP。
由 `git archive 9f2810e` 导出，未包含数据集、实验输出、环境和子模块的实际文件。
归档是历史代码的独立副本，不回退或覆盖当前工作区。旁边的 README 记录来源和校验信息。
模型权重和实验 training.json 仍保留在原实验目录，未复制进源码ZIP。

## 专用入口

`scripts/train_quantized12_render_only.sh` 独立调用 CLI，不经过两阶段脚本。
其参数兼容历史 `9f2810e` 的 CLI，也可在当前 main 上运行。锁定：

- bootstrap=0、joint=0：只优化源 PLY 多视角渲染 RGB MSE；
- 每轴12 bit XYZ 可靠侧流；属性仍用 JSCC；
- 固定10 dB AWGN，逐 Gaussian 8/16/32复符号档位及随机混合布局；
- 学习率1e-4，不裁剪梯度，无随机q0；
- 全部非验证训练视角，每步2视角，4验证视角×2固定噪声试验；
- 默认blocks=64、CPU数据缓存、resolution=2；
- 默认本次5000步，每500步验证及保存，不早停。

不受继承的 BOOTSTRAP_STEPS、BOOTSTRAP_OBJECTIVE、JOINT_STEPS、TRAIN_VIEWS、
RENDER_LR、POSITION_DELIVERY 等环境变量影响，避免误启动另一条训练线。
可覆盖 PLY、SCENE、PYTHON_BIN、RENDER_STEPS、BLOCKS_PER_BATCH、VALIDATE_EVERY、SAVE_EVERY。
必须显式选择空闲GPU；不要仅凭低利用率判断空闲，应同时看显存和进程。

## 从现有1000步权重继续4000步

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
conda activate maskgs
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main

# 2只是示例，先用nvidia-smi确认该卡空闲。
OUT="$PWD/output/truck_quantized12_render_continue_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 INITIALIZATION=checkpoint \
INIT="$PWD/output/truck_quantized12_render20000_20260918_143246/codec.pt" \
RENDER_STEPS=4000 nohup bash scripts/train_quantized12_render_only.sh "$OUT" \
  > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

这是**权重续训，不是严格断点恢复**。旧checkpoint不含Adam状态，--init保留模型权重/统计量，
新建Adam，重新从日志step=0进行验证并计数。新4000步是在旧1000步权重上继续学习，
但不能冒称与连续随机初始化训练5000步完全相同。新输出不覆盖原实验。
初始化检查要求position-delivery=quantized、position-bits=12；模型结构和rates来自检查点，
因此续训请使用上面这次有效实验的checkpoint，而非任意文件。

## 从随机权重重新训练更长时间

```bash
OUT="$PWD/output/truck_quantized12_render_random_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES=2 INITIALIZATION=random RENDER_STEPS=5000 \
nohup bash scripts/train_quantized12_render_only.sh "$OUT" > "${OUT}.log" 2>&1 &
echo $! > "${OUT}.pid"
tail -f "${OUT}.log"
```

修改RENDER_STEPS即可增加预算；5000只是观察预算，不是必要收敛量。
检查 `validation.jsonl`、`validation_images/`、`charts/`；主选模指标仍为四布局平均源渲染MSE。
`codec_best_render.pt` 为已有验证记录里最优；`codec.pt` 是最后执行步。

## 安全提取历史代码（可选）

不需要为了运行纯渲染线在现有服务器目录执行checkout/reset。若需审计历史源码，
可在一个全新目录中提取 `git archive 9f2810e`，或另外添加detached worktree。
数据、编译扩展和依赖不包含在源码归档里，需要使用现有环境或单独准备。
