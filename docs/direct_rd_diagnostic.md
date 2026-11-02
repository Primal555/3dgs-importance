# 停滞提案的一次性归因测试

本测试读取 `truck_direct_rd` 最新原子 checkpoint，固定地图与两个网络，不继续训练，不修改原实验目录。沿用源模型、MSE＋结构项、β、归一化、候选采样和数据划分。

历史拒绝提案未保存点编号。本测试在固定地图上延续调度器，重新收集默认 10 窗口提案，记录新的点编号、动作、排序、视角及噪声种子。

## 比较内容

1. **同条件预测误差**：原候选窗口的网络预测与精确单点干预标签，包含整个编码块的恢复响应。
2. **视角差异**：保留同一基线、四档候选和噪声，只将训练窗口改成原校准视角。
3. **噪声差异**：校准视角保持不变，使用原验收的固定噪声试验。
4. **成本代理误差**：用完整传输编码分别测量单点动作的实际通信开销，对比原代理。
5. **多点交互**：同条件单点损失之和与同组完整场景实测比较，分列图像交互和通信成本非可加性。
6. **排序前缀盲区**：整批／半数／四分之一提案、高分样本及其互不重叠小组。

默认从高分、中分、低分已选动作及普通候选各抽 16 个。普通候选采用当前档位以外预测最好的档，保留预测不利动作。各层分别统计，不能当作全场景随机误判率。

局部标签会抽取默认 4 个动作，在原校准噪声试验上对照完整渲染器。另核对重算校准 RD 与原 checkpoint。若一致性未通过，报告先指向执行问题，暂停解释其他归因。

## 服务器启动

将 RUN 设置为本轮 **truck_direct_rd** 目录。以下 PAUSE_SOURCE=1 会核对同 UID、同模块和同绝对 OUT，向匹配进程发送 SIGTERM，等待退出；使用最近已保存 checkpoint，未保存的几个窗口可在以后恢复时重算。原编解码器、场景和收益权重均不修改。

```bash
cd /data/home/zhangyueheng/projects/3dgs-importance
git -c submodule.recurse=false pull --ff-only --no-recurse-submodules origin main
conda activate maskgs
RUN="$OUT"
GPU=2
CHECK_OUT="$PWD/output/truck_direct_diagnostic_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" PAUSE_SOURCE=1 nohup bash scripts/test_direct_rd_diagnostic.sh \
  "$RUN" "$CHECK_OUT" >"${CHECK_OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${CHECK_OUT}.pid"
tail -n 40 -F "${CHECK_OUT}.log"
```

已有原进程停止时可省略 PAUSE_SOURCE=1。GPU 应为空闲卡；暂停源任务会释放其显存。只读预览可加 `--dry-run`。

## 输出

- `summary.json`：分层预测／真实收益、方向误判、误差分解、联合交互、执行一致性和修正重点。
- `actions.csv`、`actions.json`：逐动作、逐条件明细和原行号。
- `actions.json` 的 `condition_samples` 保留每个 context、seed、trial 的原始收益与视角名称，便于检查噪声下的符号翻转。
- `groups.json`、`proposals.npz`、`contexts.json`：可追溯动作集合和采样条件。
- `review.tar.gz`：精简查看包。不含 checkpoint、replay、模型副本或批量图片。

本地定向验证覆盖 CPU 配对噪声、候选缓存复用、局部标签／完整渲染差分、只读输出和进程路径匹配。CUDA 在服务器运行既有预检及真实抽样一致性检查。

输入图像通过 COLMAP 元数据先筛选原训练名字后再打开，不加载测试图像；原测试划分只保留 checkpoint 中的身份元数据。`sample_coverage` 报告每层请求数、实际数和有效数；某层少于 8 个有效动作时标记 `insufficient`，不生成机制主因排名。

## 结果之后的修改依据

- 同条件高分误判明显：针对被选择的动作修正收益学习和筛选，增加高分困难负样本。
- 视角／噪声差异主导：先对齐收益定义和候选评价条件，再调整采样平均。
- 单点正确、联合变差：修改相互作用动作的组织、分组验收和失败集合拆分。
- 成本代理差异主导：修正边际通信成本，保留实测成本作为最终依据。
- 局部标签／完整渲染不一致：优先修正执行路径。

上述是结果对应的代码方向。实际选择依据各项原始差值、动作类型和分层数量，修正后仍需同条件前后对照确认收益。
