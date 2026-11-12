# 侧信息数字链路交付验证

## 后续实验采用的统一计费标准

自2026-11-12起，新实验的正式通信成本按实际复信道使用次数统计。设属性发送量为N_attr个复符号，三路实际序列化文件大小分别为B_xyz、B_metadata、B_scales字节，则默认协议下：

```text
N_file(B) = ceil((B + 48) / 508) * 1536
C = N_attr + N_file(B_xyz) + N_file(B_metadata) + N_file(B_scales)
```

LDPC的4096个信息位含每码字4字节CRC，因此每码字可承载508字节文件数据。6144个编码位经16QAM占1536次复信道使用。48字节文件头在每个独立文件中计入一次，末码字补齐完整计费。文件本身已有的GXYZ、元数据及SCL1帧头均包含在B中。

代码复用`comparison_baselines.digital_link.file_channel_cost`及`comparison_baselines.side_information_accounting.side_information_cost`。新实验的C_ref、基线、RD评价和选模使用相同口径；训练中的成本代理需标明。历史实验保留原始RD与参考常数，统一口径的重算结果单独列出。共享模型预装，默认无ARQ；实际链路失败仍计入发送成本，成功率与成功条件下画质同时报告。

服务器实测记录`truck_actual_side_delivery_gpu2_20261112_114308`的selected每次成本：属性14366104、XYZ7277568、元数据536064、尺度52224，合计22231960次复信道使用。CUDA打包解码预检最大绝对误差为0。本地已同步的前两次完整试验画质与原实验同噪声试次一致，第三次已有三路文件接收成功记录；该本地副本尚无完整四次试验及uniform_q2汇总。

入口：`scripts/test_digital_side_delivery.sh COMPLETED_ALLOCATION NEW_OUT`。

对已完成的分配结果进行独立最终交付测试，默认评价 selected 和全 q2。模型与档位保持固定；继承源实验的 SNR、测试相机、试验次数及图像指标。当前 Truck 为10 dB、32个测试相机、4次试验。可用`--labels selected latest uniform_q2 uniform_q3`选择已有档位图。

发送端按原始编解码块、批量分组及补齐规则编码。XYZ为量化后压缩字节，元数据包含完整档位序列、边界、配置、模型身份及接收端重建分组所需的blocks_per_batch；块尺度为原SCL1帧串联。接收端可从收到的元数据推导每帧长度，无须场景原始属性、latent、分组索引或发送端尺度张量。新增协议字段的实际元数据字节完整计费，因此数字链路成本可能与历史静态账本存在少量差别。

三路文件分别经过现有`comparison_baselines.digital_link`：48字节文件头、CRC32、LDPC4096/6144、16QAM、公开加扰、AWGN、软解调、20次LDPC迭代、文件SHA256检查。默认无ARQ。可用`--baseline BASELINE_RUN`继承已完成数字基线的LinkConfig，并核对相机名单、分辨率、图像目录与背景配置。数字环境由`DIGITAL_PYTHON`指定，默认`$HOME/miniforge3/envs/gs_digital/bin/python`。

属性以实际复符号经过AWGN；`received.npy`是信道输出的仿真容器，未再次数字编码，其磁盘大小不作为通信开销。三路侧信息与属性使用相同Es/N0条件、独立噪声。保留原始逐点固定符号槽的属性噪声抽样，便于和历史理想侧信息结果配对检查。

在第一次试验中，服务器先核对packet接收端与原始批量编解码器的一致性，写入execution_check.json。通过后再开展数字链路测试。只有三路文件全部通过接收检查，才从接收文件解码并渲染。发送端侧信息文件会在实际接收端调用前删除；失败时记录交付失败、完整成本和各文件CRC/误码信息。失败试验的画质为null，汇总报告成功率及成功交付条件下的PSNR、SSIM、LPIPS，与数字基线的失败统计方式一致。

每次试验的全部文件发送都执行，即使其中一路失败；成本为属性复符号数加三路实际数字信道使用之和，不进行重传。试验结束删除临时发送文件、接收文件及浮点信号容器。`review.tar.gz`只保留配置、结果、数字链路日志和少量对照图片。原分配实验的训练成本、RD参考和选模规则保留在原目录；这次交付测试产生独立结果。

本地验证覆盖部分块、非原始顺序、混合档位、全删除、噪声／无噪声、禁止接收端调用编码器、缺失及多余尺度字节，以及成功／失败汇总。真实Sionna链路测试在依赖存在时执行；CUDA全量路径由服务器预检确认。

```bash
conda activate maskgs
GPU=2
SOURCE="$PWD/output/truck_first_order_views32_cap2_beta0p5_gpu2_20261111_213320"
OUT="$PWD/output/truck_actual_side_delivery_gpu${GPU}_$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="$GPU" nohup bash scripts/test_digital_side_delivery.sh \
  "$SOURCE" "$OUT" >"${OUT}.log" 2>&1 &
printf '%s\n' "$!" >"${OUT}.pid"
tail -n 40 -F "${OUT}.log"
```
