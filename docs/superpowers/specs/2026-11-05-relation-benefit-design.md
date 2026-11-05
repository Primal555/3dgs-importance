# 关系收益网络独立实验

## 目标与边界

服务器完成一次源场景及全 q1/q2/q3、mixed 无噪声有限符号属性重建，构建固定输入缓存。关系网络分别进行四背景条件前向，等权聚合有符号失真分量，输出原 PLY 顺序的 q0–q3 价值及档位。编解码器固定，保持 10 dB、0/8/24/48 和可靠 16bit XYZ。保留已有收益网络实验。多背景细节见 `2026-11-05-relation-context-design.md`。

用户已批准关系网络路线及本轮实现，工作在当前仓库新增独立文件。

## 与原实验对齐

- 使用 load_inputs/make_scene，严格核对源 codec、PLY、相机身份。
- 内部所有缓存及教师索引统一为原 PLY 行号；光栅器使用 Morton 顺序，转换必须显式。
- 预处理相机从训练相机中确定性选择，测试照片不进入特征或监督。源 PLY 渲染提供图像参考。
- 候选特征采用 finite-noiseless 重建，噪声实际采样仅用于教师；发送端不读取接收端的真实噪声。
- 教师轮换全 q1、全 q2、全 q3、mixed 背景：每次仅改变一个点，但重算其所在整个编码块，包含 q0 引起的语义上下文、解码邻域和块归一化变化。标签为同背景 D(qk)-D(q0)，分量为 [ΔMSE, lambda_s*Δstructure]/d_ref。画质参考固定源 PLY 渲染。
- 首版评估固定预处理背景下的四档价值，最终混合场景另做真实通信恢复评价。单点标签的相加仅为代理，报告真实混合 RD。
- 同一教师组、四个档位共用固定槽位噪声；默认每组两次噪声试验取均值。
- 校准 D_ref 仅用独立校准相机、全 q3；C_ref 为统一数字文件侧信息口径下的全 q3 总复信道使用。
- 逐点成本采用明确的边际代理，最终通信成本由 ExactDeploymentCost.comparison_accounting 测量；不将旧 2bit/use 账本混入新 RD。

## 网络

宽度 128、8 组局部关系注意力；三维邻域 16，投影邻域 16；两层三维残差注意力，一层投影关系注意力，一层四候选交互；LayerNorm。节点编码共享，跨编码块构建三维邻域。投影邻居来自覆盖重叠的前后景，保留有效性掩码。

共同头 B 输出两分量；候选头输出三个 A，每分量按三档减均值。v0=0，vk=B+Ak-mean(A)。输出无 Sigmoid，不强制档位单调。监督分量保留 float64，网络浮点输出。

同一前馈图可以分批计算；每个根节点训练使用完整两跳空间邻域和一跳投影邻域。分批和整图结果须一致，不能将批次边界当成邻域边界。

## bank 接口（原 PLY 顺序，CPU 张量）

字典字段：
- node: float32 [N,Dn]，源属性、候选差异及静态／块统计。
- candidate: float32 [N,4,Dc]，原始参考+候选属性有符号差、存在性、符号数；q0 保留原始覆盖描述。
- xyz: float32 [N,3]，等比例归一化坐标。
- spatial_index: int64 [N,Ks]，不含自身，尾部 -1。
- view_node: float32 [V,N,Dv]，源投影、可见性、源图及 q2 残差 profile、候选投影变化。
- view_geom: float32 [V,N,7]，归一化 u,v、log depth、归一化 radius、opacity、log potential mass、log actual mass。
- view_index: int64 [V,N,Kv]，不含自身，尾部 -1。
- view_weights: float32 [V]，固定均匀权重，教师使用相同视角及权重。
- metadata: JSON 类型字典，版本、codec、场景、相机、噪声输入协议、构建参数、尺寸。

公共函数 prepare_bank(model,scene,cameras,args) 返回上述字典。纯函数 build_spatial_graph(xyz,k,workers)、build_projection_graph(projected,k,workers) 供测试。

## model 接口

RelationBenefitNetwork(node_dim,candidate_dim,view_dim,width=128,groups=8,dropout=0.,context_node_dim=...,context_view_dim=...)。
fit_normalizers(bank,training_ids) 仅使用指定训练行拟合输入尺度并设置下限。
forward(bank,root_ids,context=...) 返回 per_view [B,V,3,2]、values [B,3,2]、common [B,2]、advantage [B,3,2]。
output_scale 注册 buffer [2]，由训练标签拟合，推理无重新拟合。支持 CPU bank + GPU network，邻域闭包按批传输。predict_all(bank,batch_size) 返回四条件等权平均的 CPU values [N,3,2]，不保留全图 autograd；条件间共享静态节点和候选编码。

## 训练与存储

默认 700 窗口，前 50 窗口拟合标签尺度，64 个完整标签组／窗口，64 次网络更新／窗口，batch 8。标签只存 context、source、step、values [V,3,2]、noise trial count，输入从 bank 索引，复用已有精确局部渲染教师。四背景各175窗口，更新共41,600次。

按原行号固定训练／验证点划分。验证组从首轮固定建立，绝不训练；所有阶段的验证组保持一致。标签覆盖使用无放回轮转，记录真实访问次数。活跃标签池同时保留初始稳定标签与近期标签，分层均匀抽取并明确配额；不按难例无限放大权重。

回归用固定尺度的 SmoothL1，分别监督共同项和中心化档间项，辅以旧版实测使用的 RD-gap Bernoulli KL。通信成本以同一单位进入训练和选档。日志聚合整个窗口所有更新，记录平均／最大梯度和裁剪比例，分别记录共同／档间／方向损失。

只保存 latest 与 final 网络检查点、标签与输入缓存供精确续训。review.tar.gz 仅包含计划、指标、图像及档位数组，不含 bank、codec 权重、优化器或大 replay。正常完成后默认用末步网络部署，不按测试集选模。定期验证不生成图片，最终仅保存两视角 source/received 和档位图。

## 验证重点

CPU：节点置换、有效性掩码、q0 存在性、邻居改变响应、输出分解、分批等价、正负标签、共同偏置与方向梯度、标签隔离、精确续训、边际成本与统一最终计费。
CUDA：服务器启动时检查局部教师与完整光栅器、块级反事实与完整 codec 噪声槽位一致；本地不宣称真实 CUDA 渲染已验证。

## 文献依据

RAP 属性与邻域统计；Point Transformer V2 分组向量注意力；FCGS 共享场景上下文；Dueling Networks 共同价值与中心化优势分解。当前模型使用真实渲染监督，本文不引入 TD 自举。
