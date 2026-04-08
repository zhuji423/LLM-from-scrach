# Multi-Head Attention 实现细节

> 核心问题：为什么 QKV 要转置？为什么最后需要 out_proj？

---

## 问题 1：为什么 QKV 要转置？

### 转置操作

```python
# 转置前的形状
(batch, num_tokens, num_heads, head_dim)
 dim0   dim1        dim2       dim3

# transpose(1, 2) - 交换 dim1 和 dim2
(batch, num_heads, num_tokens, head_dim)
 dim0   dim1       dim2        dim3

# 代码示例
keys = keys.transpose(1, 2)
queries = queries.transpose(1, 2)
values = values.transpose(1, 2)
```

### 具体数值例子

```python
import torch

# 创建测试数据
x = torch.randn(2, 10, 8, 64)
print("转置前:", x.shape)  # torch.Size([2, 10, 8, 64])
#                               batch=2, tokens=10, heads=8, head_dim=64

x_transposed = x.transpose(1, 2)
print("转置后:", x_transposed.shape)  # torch.Size([2, 8, 10, 64])
#                                       batch=2, heads=8, tokens=10, head_dim=64
```

---

## 转置的三大原因

### 1. 让每个 Head 独立并行计算

```
转置前: (batch, num_tokens, num_heads, head_dim)
┌──────────────────────────────────────────────────────────────┐
│  数据布局:                                                    │
│  batch 0:                                                    │
│    token 0: [head0_data, head1_data, ..., head7_data]       │
│    token 1: [head0_data, head1_data, ..., head7_data]       │
│    token 2: [head0_data, head1_data, ..., head7_data]       │
│    token 3: [head0_data, head1_data, ..., head7_data]       │
│                                                              │
│  ❌ head 数据是交错的，无法直接批量计算                       │
└──────────────────────────────────────────────────────────────┘

                         ↓ transpose(1, 2)

转置后: (batch, num_heads, num_tokens, head_dim)
┌──────────────────────────────────────────────────────────────┐
│  数据布局:                                                    │
│  batch 0:                                                    │
│    head 0: [token0_data, token1_data, token2_data, token3_data] │
│    head 1: [token0_data, token1_data, token2_data, token3_data] │
│    ...                                                       │
│    head 7: [token0_data, token1_data, token2_data, token3_data] │
│                                                              │
│  ✅ 每个 head 的数据是连续的，可以并行计算！                  │
└──────────────────────────────────────────────────────────────┘
```

### 2. 符合批量矩阵乘法的维度约定

```python
# Attention 核心计算
scores = queries @ keys.transpose(-2, -1)

# queries: (batch, num_heads, num_tokens, head_dim)
# keys.T:  (batch, num_heads, head_dim, num_tokens)
# ────────────────────────────────────────────────
# result:  (batch, num_heads, num_tokens, num_tokens)  ← Attention Matrix
```

**PyTorch 批量矩阵乘法规则**：

```
A @ B, 其中 A: (*, m, n), B: (*, n, p)
         → 结果: (*, m, p)

* 代表任意数量的"批量维度"（batch dimensions）
只有最后两个维度参与矩阵乘法

转置后:
- batch 和 num_heads 都是批量维度
- num_tokens 和 head_dim 是矩阵维度
- PyTorch 自动对每个 (batch, head) 组合并行计算 attention
```

**如果不转置会怎样？**

```python
# 假设保持 (batch, num_tokens, num_heads, head_dim)
queries @ keys.transpose(-2, -1)
# (b, num_tokens, num_heads, head_dim) @ (b, num_tokens, head_dim, num_heads)
#                ↑        ↑    ↑                    ↑        ↑    ↑
#          这三个维度无法对齐做矩阵乘法！
```

### 3. 性能：完全并行 vs 串行循环

| 实现方式 | 是否转置 | GPU 利用率 | 原因 |
|---------|---------|-----------|------|
| ✅ **标准实现** | 转置 | 高 | 所有 head 完全并行 |
| ❌ 不转置 | 不转置 | 低 | 需要循环处理每个 head |

```python
# ❌ 如果不转置，只能这样写（串行）
for i in range(num_heads):
    q_i = queries[:, :, i, :]  # 提取第 i 个头
    k_i = keys[:, :, i, :]
    scores_i = q_i @ k_i.transpose(-2, -1)
    # ... 重复 8 次

# ✅ 转置后（并行）
scores = queries @ keys.transpose(-2, -1)  # 一次性计算所有 head！
```

---

## 类比：并行处理流水线

```
不转置的情况（低效）:
工人1: 处理 产品A的零件1,2,3 → 产品B的零件1,2,3 → 产品C的零件1,2,3
工人2: 处理 产品A的零件4,5,6 → 产品B的零件4,5,6 → 产品C的零件4,5,6
       ↑ 产品数据交错，工人之间需要频繁协调

转置后的情况（高效）:
工人1: 专门处理 产品A的所有零件
工人2: 专门处理 产品B的所有零件
工人3: 专门处理 产品C的所有零件
       ↑ 每个工人独立工作，完全并行
```

---

## 问题 2：为什么需要 out_proj？

### reshape 只是"拼接"，out_proj 才是"融合"

```python
# 假设 num_heads=8, head_dim=64, d_out=512

# Step 1: 转置回来
context_vec = context_vec.transpose(1, 2)  # (b, num_tokens, num_heads, head_dim)
                                            # (b, 10, 8, 64)

# Step 2: Reshape - 只是物理拼接
context_vec = context_vec.reshape(b, num_tokens, self.d_out)
# (b, 10, 8, 64) → (b, 10, 512)
#        ↓
# [head0_64维 | head1_64维 | head2_64维 | ... | head7_64维]
#  只是把 8 个头的数据排成一行，彼此没有交互！

# Step 3: Output Projection - 融合不同 head 的信息
context_vec = self.out_proj(context_vec)
# Linear(512 → 512)
# 让不同 head 学到的信息可以相互融合
```

---

## 可视化对比

```
┌─────────────────────────────────────────────────────────────┐
│  仅 Reshape（没有 out_proj）                                │
│  ═══════════════════════════════════════════════════════     │
│                                                             │
│  Head 0 输出: [0.1, 0.2, ..., 0.9]  (64维)                 │
│  Head 1 输出: [0.3, 0.1, ..., 0.7]  (64维)                 │
│  ...                                                        │
│  Head 7 输出: [0.5, 0.8, ..., 0.4]  (64维)                 │
│       ↓ reshape                                             │
│  拼接结果: [0.1,0.2,...,0.9, 0.3,0.1,...,0.7, ..., 0.5,0.8,...,0.4] │
│            └─ head0 ─┘ └─ head1 ─┘       └─ head7 ─┘       │
│                                                             │
│  ❌ 问题：8个头各自为政，信息没有交互！                      │
│           就像8个专家各说各话，没有综合讨论                  │
└─────────────────────────────────────────────────────────────┘

                         ↓ + out_proj

┌─────────────────────────────────────────────────────────────┐
│  Reshape + Output Projection                                │
│  ═══════════════════════════════════════════════════════     │
│                                                             │
│  拼接结果: [h0_data | h1_data | ... | h7_data]              │
│       ↓ W_out @ concat                                      │
│       每个输出维度都是所有头的加权组合：                       │
│                                                             │
│  out[0] = w₀₀·h0[0] + w₀₁·h0[1] + ... + w₀,₅₁₁·h7[63]      │
│  out[1] = w₁₀·h0[0] + w₁₁·h0[1] + ... + w₁,₅₁₁·h7[63]      │
│  ...                                                        │
│                                                             │
│  ✅ 不同 head 的信息融合在一起！                             │
│     8个专家的观点经过综合讨论，得出最终结论                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 类比：多专家会议

```
场景：8位专家分析同一个问题

只有 Reshape（没有 out_proj）:
┌────────────────────────────────────────┐
│  专家1: "我觉得是语法问题"              │
│  专家2: "我觉得是语义问题"              │
│  专家3: "我觉得是上下文问题"            │
│  ...                                  │
│  专家8: "我觉得是逻辑问题"              │
│       ↓                               │
│  报告: 把8个人的话依次抄下来             │
│  [专家1的话][专家2的话]...[专家8的话]    │
└────────────────────────────────────────┘
❌ 没有综合，只是并列

Reshape + out_proj:
┌────────────────────────────────────────┐
│  专家1-8: 各自分析                      │
│       ↓                               │
│  主持人（out_proj）: 综合所有观点        │
│  "根据专家1的语法分析和专家2的语义分析..." │
│  "结合专家3提到的上下文..."            │
│  "最终结论是..."                       │
└────────────────────────────────────────┘
✅ 有机融合，形成统一观点
```

---

## out_proj 的三大作用

### 1. 融合不同表示子空间

```
Head 0: 关注句法结构（主谓宾）
Head 1: 关注语义关系（因果、对比）
Head 2: 关注距离关系（远近词汇）
...
Head 7: 关注情感色彩

out_proj: 把这8个视角融合成一个统一的表示
```

### 2. 增加模型容量

```
参数量:
- 没有 out_proj: 0 个可学习参数（reshape 是固定操作）
- 有 out_proj: 512×512 = 262,144 个参数

表达能力:
- 没有 out_proj: 线性组合受限于原始的 head 输出
- 有 out_proj: 可以学习任意的线性变换
```

### 3. 符合残差连接的设计

```python
# Transformer Block 中的残差连接
x = x + multi_head_attention(x)
#   ↑                         ↑
#  原始输入              需要和输入维度相同

如果没有 out_proj:
- 输出是 8 个 head 的简单拼接
- 每个维度的含义不清晰

有 out_proj:
- 输出经过变换，可以更好地与残差相加
- 相当于给模型一个"调节阀"
```

---

## 原始论文的说明

> **"Attention is All You Need" (2017)**
>
> Multi-head attention allows the model to jointly attend to information from **different representation subspaces** at different positions.
>
> The outputs are **concatenated and once again projected**, resulting in the final values.

关键点：
1. 每个 head 关注**不同的表示子空间**
2. 拼接后需要**再次投影（once again projected）**
3. 目的是融合不同子空间的信息

---

## 完整的 Attention 计算流程

```python
# 假设 num_heads=8, head_dim=64, seq_len=10

# 1. 线性投影得到 Q, K, V
Q = self.W_query(x)  # (b, 10, 512)

# 2. 拆分成多个头
Q = Q.view(b, 10, 8, 64)  # (b, num_tokens, num_heads, head_dim)

# 3. 转置：把 head 维度移到前面（让 head 并行）
Q = Q.transpose(1, 2)  # (b, 8, 10, 64)

# 4. 计算 Attention Scores (每个 head 独立计算)
scores = Q @ K.transpose(-2, -1)  # (b, 8, 10, 10)
attn_weights = F.softmax(scores / sqrt(head_dim), dim=-1)

# 5. 加权求和
context = attn_weights @ V  # (b, 8, 10, 64)

# 6. 转置回来并拼接
context = context.transpose(1, 2)  # (b, 10, 8, 64)
context = context.contiguous().view(b, 10, 512)  # 合并 heads

# 7. Output Projection（融合不同 head 的信息）
output = self.out_proj(context)  # (b, 10, 512)
```

---

## 对比总结

| 特性 | 只有 Reshape | Reshape + out_proj |
|------|-------------|-------------------|
| **操作性质** | 确定性拼接 | 可学习的融合 |
| **head 之间** | 各自独立 | 相互交互 |
| **参数量** | 0 | d_model² |
| **表达能力** | 受限 | 更强 |
| **是否标准** | ❌ 非标准 | ✅ 标准实现 |

---

## 一句话总结

> **QKV 转置**是为了让所有 attention head 能够完全并行计算（把 num_heads 变成批量维度）。**out_proj** 不是修正 reshape，而是让不同 head 的信息有机融合——就像把多个专家的独立观点综合成一个统一结论。

---

> 最后更新: 2026-04-07
