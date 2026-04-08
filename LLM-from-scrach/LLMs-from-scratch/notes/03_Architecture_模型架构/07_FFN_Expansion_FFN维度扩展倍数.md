# FFN 维度扩展倍数详解

> 核心问题：为什么 FeedForward 的中间层维度是 embedding 的 4 倍？不同模型的选择是什么？

---

## FFN 的"扩展-压缩"结构

```
输入 x: [batch, seq_len, emb_dim=768]
         │
         ▼
┌─────────────────────────────────────┐
│  Linear(768 → 3072)   ← 扩展 4 倍   │
│         │                           │
│      GELU()           ← 非线性激活  │
│         │                           │
│  Linear(3072 → 768)   ← 压缩回来    │
└─────────────────────────────────────┘
         │
         ▼
输出: [batch, seq_len, emb_dim=768]
```

### GPT-2/3 的标准实现

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768 → 3072
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 3072 → 768
        )

    def forward(self, x):
        return self.layers(x)
```

---

## 为什么是 4 倍？

### 1. 源自原论文 "Attention is All You Need"

```
Transformer (2017):
  d_model = 512   (embedding 维度)
  d_ff = 2048     (FFN 隐藏层维度)
  比例 = 2048/512 = 4
```

### 2. 信息瓶颈理论

```
      768 维                    3072 维                   768 维
   ┌─────────┐              ┌───────────────┐          ┌─────────┐
   │ 压缩的  │  ──扩展──▶   │  更高维空间    │ ──压缩──▶ │ 精炼的  │
   │ 表示    │              │  学习复杂模式  │          │ 表示    │
   └─────────┘              └───────────────┘          └─────────┘
```

**类比**：
- 像是把一幅画**放大 4 倍**仔细观察细节
- 然后再**缩小回去**，但保留了发现的重要信息

### 3. 计算量与表达能力的平衡

| 扩展因子 | 参数量 | 表达能力 | 实际应用 |
|---------|--------|---------|---------|
| 2x | 较少 | 受限 | 轻量模型 |
| **4x** | 适中 | **足够强** | **GPT-2/3, BERT** |
| 8x | 很大 | 更强但收益递减 | 少见 |

---

## 不同模型的 FFN 设计

### 原始 Transformer (2017)

```python
class OriginalFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # 512 → 2048
        self.linear2 = nn.Linear(d_ff, d_model)    # 2048 → 512
        self.relu = nn.ReLU()                       # ReLU 激活
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
```

### LLaMA SwiGLU (2023)

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, dim=4096, hidden_dim=11008):  # ≈ 2.67 倍
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up

    def forward(self, x):
        # SwiGLU: Swish(xW1) ⊙ (xW3) 然后 W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

---

## 主流模型 FFN 扩展比例调研

### Qwen 系列

| 模型 | hidden_size | intermediate_size | **扩展倍数** | 激活函数 |
|------|-------------|-------------------|-------------|----------|
| **Qwen 1.0** (7B) | 4096 | 22016 | **5.38x** | SiLU |
| **Qwen 1.5** (7B) | 4096 | 11008 | **2.69x** | SiLU |
| **Qwen 2.0** (7B) | 3584 | 18944 | **5.29x** | SiLU |
| **Qwen 2.5** (7B) | 3584 | 18944 | **5.29x** | SiLU |

```
Qwen FFN 演进:

Qwen 1.0:  ████████████████████████████████  5.38x  (激进扩展)
Qwen 1.5:  ████████████                      2.69x  (收敛到 LLaMA)
Qwen 2.0:  ███████████████████████████       5.29x  (回归大扩展)
Qwen 2.5:  ███████████████████████████       5.29x  (保持)

对比:
GPT-2/3:  ████████████████                  4.00x
LLaMA:    ████████████                      2.67x
```

### MiniCPM 系列

| 模型 | hidden_size | intermediate_size | **扩展倍数** | 特点 |
|------|-------------|-------------------|-------------|------|
| **MiniCPM 1** (1B) | 1536 | 3840 | **2.50x** | 小模型高效设计 |
| **MiniCPM 2** (2B) | 2304 | 5760 | **2.50x** | 保持一致 |
| **MiniCPM 3** (4B) | 2560 | 6400 | **2.50x** | 统一风格 |

```
MiniCPM FFN 设计:

MiniCPM 1:  ██████████              2.50x
MiniCPM 2:  ██████████              2.50x
MiniCPM 3:  ██████████              2.50x  (一致的设计哲学)

特点: 小于 LLaMA 的 2.67x，追求极致效率
```

---

## 不同扩展倍数的权衡

```
┌────────────────────────────────────────────────────────────────┐
│                    FFN 扩展倍数权衡                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   2.5x (MiniCPM)                                               │
│    │   └─ 小模型优先：参数效率 > 表达能力                        │
│    │   └─ 使用 SwiGLU，门控本身提供额外容量                      │
│    │                                                           │
│   2.67x (LLaMA)                                                │
│    │   └─ SwiGLU 需要 3 个矩阵 (W1,W2,W3)                       │
│    │   └─ 8/3 ≈ 2.67 让总参数量≈传统 4x 的参数量                 │
│    │                                                           │
│   4x (GPT-2/3, BERT)                                           │
│    │   └─ 经典设计，简单 GELU，无门控                            │
│    │                                                           │
│   5.3x (Qwen)                                                  │
│    │   └─ 激进策略：更大 FFN = 更多知识存储容量                   │
│    │   └─ 代价是更多参数，但 Qwen 追求模型能力上限                │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 参数量计算

### 传统 FFN (GELU)

```
参数 = d × 4d + 4d × d = 8d²

例如 GPT-2 (d=768):
  8 × 768² = 4,718,592 参数/层
```

### SwiGLU FFN

```
参数 = d × h + h × d + d × h = 3dh

若要参数量相当: 3dh ≈ 8d²
→ h ≈ 8d/3 ≈ 2.67d  ← LLaMA 的选择

Qwen 的选择:
  h = 5.3d
  → 参数 = 3 × d × 5.3d = 15.9d²  (几乎是传统的 2 倍！)
```

---

## 激活函数演进

```
ReLU (2017)           GELU (2018-2020)         SwiGLU (2022+)
     │                      │                        │
     ▼                      ▼                        ▼
  ___/                   _____                    门控机制
 /                      /     \                  更强表达
简单高效              更平滑                    额外参数但效果更好
Transformer           GPT-2/3, BERT            LLaMA, Mistral
```

---

## FFN 在 Transformer 中的作用

Transformer 两大核心组件分工：

| 组件 | 作用 | 类比 | 存储内容 |
|------|------|------|---------|
| **Multi-Head Attention** | Token 之间交互 | **阅读**：看整段话 | 语法知识、结构关系 |
| **FFN** | 单个 Token 的特征变换 | **思考**：消化理解 | **事实知识、世界信息** |

**研究表明**：
- **注意力层**存储"语法知识"（结构关系）
- **FFN 层**存储"事实知识"（实际信息）

例如 "Paris is the capital of France" 这个知识，主要存储在 **FFN 的参数**中。

---

## 设计哲学总结

| 模型系列 | 扩展倍数 | 设计哲学 | 代表模型 |
|---------|---------|---------|---------|
| **经典平衡** | 4x | 经典经验值，平衡参数和能力 | GPT-2/3, BERT |
| **参数等效** | 2.67x | SwiGLU 参数量等效于传统 4x | LLaMA, Mistral |
| **极致效率** | 2.5x | 小模型追求参数效率 | MiniCPM |
| **激进容量** | 5.3x | 追求知识存储上限 | Qwen |

---

## 一句话总结

> **4 倍扩展**是 Transformer 的"经验黄金比例"——在不过度增加参数的前提下，给模型足够的空间学习复杂的非线性变换。不同模型根据自己的定位（效率 vs 能力）选择不同的扩展倍数，但核心思想都是"扩展-压缩"的信息瓶颈设计。

---

> 最后更新: 2026-04-07
