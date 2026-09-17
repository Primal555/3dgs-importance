# Gaussian JSCC 使用入口

当前只维护 `learned_joint`：XYZ 与属性共享学习得到的信道符号，接收端基于带噪特征做局部联合解码。每个 Gaussian 独立选择 q0/q1/q2/q3。

完整架构、损失和运行说明见 [learned_joint_jscc.md](learned_joint_jscc.md)。

## 常用入口

- `python -m gaussian_jscc train-learned`：codec 预训练、渲染训练和可选 mask 联合训练。
- `python -m gaussian_jscc train`：上述同一入口的别名，不再执行旧训练方案。
- `python benchmark_codec.py`：固定档位下的独立通信损失、属性误差、渲染质量评估。
- `python benchmark_training.py`：完整场景前后向耗时、显存测试，不更新权重。
- `python -m gaussian_jscc transmit` / `decode`：实际打包及独立接收端恢复。
- `python -m gaussian_jscc export-route2`：导出匹配模型的四档概率与硬档位。
- `python -m gaussian_jscc plot-stats`：重新绘制已有日志。

各入口支持 `--help`。启动脚本为 `scripts/train_codec_learned.sh`，不要复用已移除的旧版参数。

## 文件与版本

主干只加载 learned_joint 的 version 4 检查点。手工位置传输及旧 geometry-first 模型需使用对应历史 Git 提交，主干不再承担其兼容实现。当前 learned-v4 的权重结构和包标识保持不变。

旧实验代码已记录在 `a545aa5`，清理不删除 `output/` 中的实验数据。需要复现时，可在独立目录检出历史版本，避免覆盖当前工作目录。

元数据可靠交付、共享模型权重和全局 bbox 归一化等假设仍需在通信开销报告中明确；不发送逐点或逐块粗坐标。
