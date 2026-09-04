# MaskGaussian + JSCC：路线一与路线二流程图

## 统一符号与边界

- $G_i$：第 $i$ 个 Gaussian 的位置、尺度、旋转、不透明度和 SH/颜色等属性。
- $p_i^{exist}$：MaskGaussian 学到的存在概率。
- $q_i\in\{0,1,2,3\}$：资源档位；0 表示不传，1/2/3 表示低/中/高码率。
- $k_i$：档位对应的信道符号数，例如 $\{0,8,16,32\}$。
- $\gamma$：当前信道状态/SNR；$B$：总信道资源预算。
- $h_i$：结合邻域上下文后得到的 Gaussian 潜在特征。
- JSCC 输出的每个信道符号同时承担信息表达和抗噪保护，不预先硬拆成“信息符号”和“校验符号”。
- “端到端”指渲染损失能够穿过信道仿真回传到通信编码器和资源策略，不要求对全场景做全局注意力。

---

## 路线一：MaskGaussian 重要性先验 + 分离式资源分配

```mermaid
flowchart LR
    subgraph S1[① 离线重要性学习]
        direction TB
        A0[多视角图像<br/>相机参数与初始点云]
        A1[MaskGaussian 训练]
        A2[紧凑 Gaussian 场景 G<br/>存在概率 p_exist]
        A3[[冻结 p_exist<br/>不受后续信道反向影响]]
        A0 --> A1 --> A2 --> A3
    end

    subgraph S2[② 信道感知码率分配]
        direction TB
        B0[当前 SNR γ<br/>总资源预算 B]
        B1[轻量 Rate Allocator]
        B2[每个 Gaussian 的档位 q_i<br/>0:不传 · 1:低 · 2:中 · 3:高]
        B3[符号预算 k_i]
        B0 --> B1 --> B2 --> B3
    end

    subgraph S3[③ 上下文感知源表征]
        direction TB
        C0[q_i=0：删除/不发送<br/>q_i>0：进入编码]
        C1[Morton/Hilbert 排序<br/>固定数量局部分块]
        C2[局部 Gaussian Transformer]
        C3[上下文潜在特征 h_i<br/>消除空间与属性冗余]
        C0 --> C1 --> C2 --> C3
    end

    subgraph S4[④ 可变码率 JSCC]
        direction TB
        D0[条件化 JSCC Encoder<br/>输入 h_i, k_i, γ]
        D1[k_i 个连续信道符号 z_i<br/>功率归一化]
        D2[[信息表达与抗噪保护<br/>由 JSCC 隐式联合学习]]
        D0 --> D1 --> D2
    end

    subgraph S5[⑤ 信道与接收端]
        direction TB
        E0[AWGN / Rayleigh 信道]
        E1[条件化 JSCC Decoder]
        E2[局部上下文重建]
        E3[恢复 Gaussian 场景 G_hat]
        E0 --> E1 --> E2 --> E3
    end

    subgraph S6[⑥ 渲染监督]
        direction TB
        F0[3DGS 可微渲染]
        F1[率失真目标<br/>L = D_render + βΣk_i]
        F0 --> F1
    end

    A3 --> B1
    A2 --> C0
    B2 --> C0
    B3 --> D0
    B0 --> D0
    C3 --> D0
    D2 --> E0
    E3 --> F0

    F1 -.反向更新.-> E1
    F1 -.反向更新.-> D0
    F1 -.反向更新.-> C2
    F1 -.反向更新.-> B1
    F1 -.梯度在此截止.-> A3

    classDef source fill:#EAF2FF,stroke:#3569B7,color:#14213D,stroke-width:1.5px;
    classDef rate fill:#FFF4D8,stroke:#C68A00,color:#4A3400,stroke-width:1.5px;
    classDef context fill:#E9F8EF,stroke:#27864D,color:#123B22,stroke-width:1.5px;
    classDef jscc fill:#EDE9FE,stroke:#7650B5,color:#2D1744,stroke-width:1.5px;
    classDef channel fill:#E8F7FA,stroke:#258092,color:#12383F,stroke-width:1.5px;
    classDef loss fill:#FFE9E9,stroke:#B84242,color:#4A1616,stroke-width:1.5px;
    class A0,A1,A2,A3 source;
    class B0,B1,B2,B3 rate;
    class C0,C1,C2,C3 context;
    class D0,D1,D2 jscc;
    class E0,E1,E2,E3 channel;
    class F0,F1 loss;
```

### 路线一的梯度边界

```text
D_render → JSCC Decoder/Encoder → Gaussian Transformer → Rate Allocator
梯度在冻结的 p_exist 处截止，不更新已经训练完成的 MaskGaussian
```

路线一回答的是：**如何把已有的“存在必要性”转化为通信资源，并在给定资源内完成抗噪传输。**

---

## 路线二：信道—渲染闭环中的端到端可学习档位

```mermaid
flowchart TB
    subgraph INPUT[阶段 A：源场景与条件]
        A1[预训练 3DGS / 紧凑 Gaussian 场景 G]
        A2[Gaussian 属性 G_i]
        A3[可选：MaskGaussian p_exist<br/>只作为先验特征或正则]
        A4[信道状态/SNR γ<br/>总预算 B]
        A1 --> A2
    end

    subgraph CONTEXT[阶段 B：可扩展上下文编码]
        A2 --> B1[Morton/Hilbert 排序<br/>固定数量局部分块]
        B1 --> B2[局部 Gaussian Transformer]
        B2 --> B3[局部上下文特征 h_i]
        B3 --> B4[可选：区域 Anchor 汇聚<br/>获得低分辨率全局摘要]
    end

    subgraph POLICY[阶段 C：可学习资源策略]
        B3 --> C1[Rate Policy Network]
        B4 --> C1
        A3 --> C1
        A4 --> C1
        C1 --> C2[档位概率 π_i^0, π_i^1, π_i^2, π_i^3]
        C2 --> C3[可微离散选择<br/>Gumbel-Softmax / Straight-Through]
        C3 --> C4[档位 q_i 与符号数 k_i]
        C4 --> C5{q_i = 0?}
        C5 -->|是| C6[不发送该 Gaussian]
        C5 -->|否| C7[进入信道编码]
    end

    subgraph JSCC[阶段 D：档位与信道条件化 JSCC]
        B3 --> D1[JSCC Channel Mapper]
        C7 --> D1
        A4 --> D1
        D1 --> D2[k_i 个连续信道符号 z_i<br/>隐式联合学习信息与保护]
        D2 --> D3[功率归一化]
    end

    subgraph RECEIVE[阶段 E：信道、恢复与渲染]
        D3 --> E1[AWGN / Rayleigh 信道]
        E1 --> E2[条件化 JSCC Decoder]
        C4 --> E2
        A4 --> E2
        E2 --> E3[局部上下文重建]
        C6 --> E3
        E3 --> E4[恢复 Gaussian 场景 G_hat]
        E4 --> E5[3DGS 可微渲染]
        E5 --> E6[总目标 L = D_render + βR<br/>R = Σk_i，可附加档位正则]
    end

    E6 -.更新内容编码.-> B2
    E6 -.更新档位策略.-> C1
    E6 -.更新信道映射.-> D1
    E6 -.更新接收端.-> E2

    classDef source fill:#E8F1FF,stroke:#3B6FB6,color:#14213D;
    classDef context fill:#E8F8EF,stroke:#27864D,color:#123B22;
    classDef policy fill:#FFF3D6,stroke:#C58A00,color:#4B3500;
    classDef channel fill:#F2EAFE,stroke:#7C4DAD,color:#2D1744;
    classDef loss fill:#FFE8E8,stroke:#B84242,color:#4A1616;
    class A1,A2,A3,A4 source;
    class B1,B2,B3,B4 context;
    class C1,C2,C3,C4,C5,C6,C7 policy;
    class D1,D2,D3,E1,E2,E3,E4 channel;
    class E5,E6 loss;
```

路线二回答的是：**在当前内容、邻域冗余、总预算和信道条件下，每个 Gaussian 应该处于哪个资源档位，才能使最终渲染率失真最优。**

---

## 两条路线的公平对比边界

```mermaid
flowchart LR
    X[相同 Gaussian 输入<br/>相同训练/测试视角] --> R1[路线一<br/>冻结 p_exist 后分档]
    X --> R2[路线二<br/>端到端学习档位]

    R1 --> S1[相同局部上下文编码器]
    R2 --> S2[相同局部上下文编码器]
    S1 --> J1[相同 JSCC 主干与功率约束]
    S2 --> J2[相同 JSCC 主干与功率约束]
    J1 --> C1[相同信道模型/SNR]
    J2 --> C2[相同信道模型/SNR]
    C1 --> O1[恢复场景与渲染结果]
    C2 --> O2[恢复场景与渲染结果]

    P1[主要变量：<br/>p_exist, γ, B → Rate Allocator → q_i] -.决定.-> R1
    P2[主要变量：<br/>h_i, γ, B → π_i → q_i] -.决定.-> R2

    classDef shared fill:#E8F8EF,stroke:#27864D,color:#123B22;
    classDef route1 fill:#E8F1FF,stroke:#3B6FB6,color:#14213D;
    classDef route2 fill:#FFF3D6,stroke:#C58A00,color:#4B3500;
    classDef output fill:#F2EAFE,stroke:#7C4DAD,color:#2D1744;
    class X,S1,S2,J1,J2,C1,C2 shared;
    class R1,P1 route1;
    class R2,P2 route2;
    class O1,O2 output;
```

## 核心区别摘要

| 项目 | 路线一 | 路线二 |
|---|---|---|
| 重要性来源 | 已训练完成的 MaskGaussian $p_i^{exist}$ | 上下文特征、SNR、预算；可选使用 $p_i^{exist}$ 作为先验 |
| 档位产生方式 | 轻量 Rate Allocator 根据冻结的 $p_i^{exist}$、SNR 和预算分档 | 可微离散策略网络端到端学习 |
| 信道是否改变原始重要性 | 不改变 $p_i^{exist}$ | 可以改变最终档位 $q_i$ |
| 上下文 Transformer 的职责 | 提取待传内容、消除源冗余 | 同时支撑内容编码与档位决策 |
| JSCC 的职责 | 在既定 $k_i$ 内隐式平衡信息与保护 | 在联合学习的 $k_i$ 内隐式平衡信息与保护 |
| 主要研究问题 | 已有存在重要性是否能有效指导通信资源 | 渲染感知的边际通信收益能否优于静态存在重要性 |
| 复杂度与稳定性 | 较低，模块边界清晰 | 较高，涉及离散决策和率失真联合优化 |

## 需要避免的歧义

1. $p_i^{exist}$ 表示 Gaussian 的存在必要性，不天然等于最佳通信码率。
2. $q_i/k_i$ 表示总信道资源，不直接规定多少符号是“信息”、多少是“保护”。
3. Transformer 用于上下文表征，不把原始 Attention 权重直接解释成重要性。
4. 两条路线都可采用局部窗口、序列化或层次化 Transformer，无需全场景 $O(N^2)$ 注意力。
5. 路线二的“端到端”可以只更新通信系统；是否同时更新原始 3DGS 参数应作为独立的系统边界说明。
