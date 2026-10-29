# 像素级收益学习与缓存局部干预

用户已批准包含独立 CUDA 局部干预接口的重构。保留冻结编解码器、10 dB、0/8/24/48、既有可靠 XYZ/尺度交付、归一化 MSE＋SSIM、十次部署与概率表接口。历史实验入口保持可用。

## 方法

每窗口重建实际混合场景一次。渲染器缓存各相机的投影、完整 tile 高斯列表和基线图像；列表包含基线提前终止之后的高斯。候选改变整个编解码块，局部渲染合并未变化列表与替换块，使用当前候选的投影范围、透明度和深度顺序。图像分辨率、alpha 截断与透明合成提前终止规则对齐原 MaskGaussian。

收益网络读取原像素 tile（16×16 内区及5像素上下文）、当前 RGB、参考残差、深度顺序场景 token 和四状态候选 token。不得把干预图像输入网络；不得固定截断深度 token。输出 q1–q3 相对 q0 的两分量失真变化图，最后按全图面积和 D_ref 积分。SSIM 监督包含变化区域之外的受影响窗口及其读取 halo。真实标签按视角保存。

采用完整干预组，map MSE＋value MSE；训练标签拟合固定 RMS 尺度，尺度不按档位分别改变。每窗口默认64组；覆盖、几何影响和训练困难度组成有非零均匀下限的抽样分布，记录概率并校正总体监督。固定空间留出；验证样本库在归一化阶段结束时冻结，后续评估同一批样本。latest/final 保存末步网络与对应优化器，不回载历史窗口。

## 效率与状态

预训练先抽监督点再仅解码需要的块；配对噪声保留原始批量噪声槽位。q0 改变存在集合时重新语义编码。分配阶段对轮转候选预测并只给部分候选采集标签。几何投影、tile 列表、基线 RGB 和局部索引窗口内复用；学习特征在网络更新后失效。原始回放和固定验证集以磁盘分片存储，检查点保存索引及抽样 RNG；分片是续训依赖，单列下载说明。

局部 CUDA 为独立扩展，不覆盖现有渲染安装。CPU 参照用于数学及接口测试；服务器启动前必须检查真实 CUDA 与全图结果。记录投影、候选解码、teacher 渲染、特征构造、网络训练/预测、缓存命中、像素量和显存峰值。大区域分块，不降分辨率或丢弃高斯。

## 公共接口

`pixel_local_renderer.capture(raw,camera,degree,existence,white_background=False,backend='cuda') -> TileScene`。
TileScene: `image[3,H,W]`, `projected[N,12]` 列为 x,y,z,conicA,conicB,conicC,opacity,existence,R,G,B,radius；`tile_ranges[M,2]`、`tile_ids[L]`；`height,width,tile_size=16`。
`project(raw,existence)->[K,12]`、`affected_tiles(ids,candidates[4,K,12],halo_tiles=0)->[T]`、`render_tiles(ids,projected[K,12],tile_ids)->[T,3,16,16]`。修改 ID 为当前渲染场景的行号；替换整个块，返回新图像，不返回 delta。

特征构造接口返回 dict：`image[T,6,26,26]` 为 RGB 与 residual；`context` 和 `context_mask` 按 tile 存为变长 tuple；`candidate[T,4,K,12]`；`candidate_mask[T,4,K]`；`valid[T,1,16,16]`。学习器只对当前 tile 批次填充，形成网络输入 `context[B,L,13]` 和 `context_mask[B,L]`。上下文13列为上述12投影量＋changed标记。投影 x/y 以 tile 左上为原点，深度正数采用 log1p，conic 和 radius 等保留连续数值。掩码依据原渲染器的 tile 覆盖范围，仅去除 padding 或不相交的 token，不能按数量截断。
`PixelBenefitNetwork.forward(inputs)->[T,3,2,16,16]`：第一分量 RGB MSE 差（RGB均值），第二分量 lambda_s 加权结构差。输出物理单位。
`pixel_density_labels(images[4,3,26,26],target[3,26,26],lambda_s)->[3,2,16,16]` 以 q0 为锚，SSIM 使用已有11×11核，图像边界零填充与原实现一致。

## 验证

独立参照覆盖删除前景显露背景、深层点提前终止解除、扩大的投影、跨 tile、深度并列、离屏、白背景、q0 重编码、同噪声槽位、SSIM halo、无干预图像输入、所有深度 token 保留、完整组尺度、抽样校正、固定验证与精确恢复。仅本轮及直接关联回归，沿用用户已明确的测试范围。

本地定向验证：137项测试及1项 subtest 通过，7项 CUDA 测试跳过。GPU 编译、数值一致性和实际吞吐仍由服务器验证。
