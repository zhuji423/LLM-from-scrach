# head_dim：每个注意力头的维度

> 深入理解多头注意力中的 head_dim 参数及 Qwen 的 GQA 设计

---

## 问题：head_dim 是每一个 attention 头的维度吗？

### 快速答案

**✅ 完全正确！**

**标准关系**：
$$\text{emb\_dim} = \text{n\_heads} \times \text{head\_dim}$$

**Qwen 0.6B 配置验证**：
```python
emb_dim = 1024
n_heads = 16
head_dim = 128

# 验证
1024 == 16 × 128  ✓
```

---

## Multi-Head Attention 的维度分解

### 完整的维度流动

```python
# 输入
x: [batch=2, seq_len=512, emb_dim=1024]

# === Query 投影 ===
queries = W_query(x)
# Shape: [2, 512, num_heads × head_dim]
#      = [2, 512, 16 × 128]
#      = [2, 512, 2048]  ← 注意：可能 > emb_dim！

# === 重塑为多头 ===
queries = queries.view(batch, seq_len, n_heads, head_dim)
#       = [2, 512, 16, 128]
queries = queries.transpose(1, 2)
#       = [2, 16, 512, 128]
#         ↑  ↑   ↑    ↑
#         |  |   |    └─ head_dim (每个头的维度)
#         |  |   └────── seq_len
#         |  └────────── n_heads (头数)
#         └───────────── batch

# === 每个头独立计算注意力 ===
# 在 128 维空间中进行 QK^T 运算
attn_scores = queries @ keys.transpose(2, 3)
# Shape: [2, 16, 512, 512]
#         ↑  ↑   ↑    ↑
#         |  |   |    └─ seq_len (keys)
#         |  |   └────── seq_len (queries)
#         |  └────────── 16 个头独立计算
#         └───────────── batch

# === 加权求和 ===
context = attn_weights @ values
# Shape: [2, 16, 512, 128]
#            ↑        ↑
#            |        └─ 每个头输出 128 维
#            └────────── 16 个头

# === 合并多头 ===
context = context.transpose(1, 2).reshape(batch, seq_len, n_heads * head_dim)
# Shape: [2, 512, 2048]

# === 输出投影 ===
output = W_out(context)
# Shape: [2, 512, 1024]  ← 投影回 emb_dim
```

---

## 从代码中看 head_dim 的使用

### 1. 投影矩阵的维度

```python
class GroupedQueryAttention:
    def __init__(self, d_in, num_heads, head_dim, num_kv_groups):
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim  # 输出维度

        # Query: 所有头的投影
        self.W_query = nn.Linear(
            d_in,                        # 1024 (emb_dim)
            num_heads * head_dim,        # 16 × 128 = 2048
            bias=False
        )

        # Key/Value: 只有 n_kv_groups 个头（GQA 优化）
        self.W_key = nn.Linear(
            d_in,                        # 1024
            num_kv_groups * head_dim,    # 8 × 128 = 1024
            bias=False
        )

        self.W_value = nn.Linear(
            d_in,                        # 1024
            num_kv_groups * head_dim,    # 8 × 128 = 1024
            bias=False
        )
```

### 2. 重塑操作

```python
def forward(self, x):
    # x: [batch, seq_len, 1024]

    queries = self.W_query(x)  # [batch, seq_len, 2048]

    # 重塑为多头格式
    queries = queries.view(
        batch,
        seq_len,
        self.num_heads,    # 16
        self.head_dim      # 128  ← 每个头的维度
    )
    # Shape: [batch, seq_len, 16, 128]

    queries = queries.transpose(1, 2)
    # Shape: [batch, 16, seq_len, 128]
    #              ↑           ↑
    #              |           └─ 每个头操作 128 维向量
    #              └─ 16 个独立的头
```

### 3. 缩放因子

```python
# 注意力分数的缩放
attn_scores = queries @ keys.transpose(2, 3)
attn_weights = torch.softmax(
    attn_scores / self.head_dim**0.5,  # ← 按 head_dim 缩放
    dim=-1
)
# 缩放因子: √128 ≈ 11.31

# 为什么要缩放？
# - 防止 softmax 饱和
# - 稳定梯度
# - 缩放因子与维度的平方根成正比
```

### 4. RoPE 位置编码

```python
def compute_rope_params(head_dim, theta_base, context_length):
    # RoPE 在每个头的维度上操作
    inv_freq = 1.0 / (
        theta_base ** (
            torch.arange(0, head_dim, 2) / head_dim
            #              ↑        ↑
            #              |        └─ 步长为 2（成对旋转）
            #              └─ 0 到 head_dim
        )
    )
    # 返回: [head_dim] 大小的 cos/sin 矩阵

def apply_rope(x, cos, sin):
    # x: [batch, num_heads, seq_len, head_dim]
    batch, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # 前半部分
    x2 = x[..., head_dim // 2:]   # 后半部分

    # Apply 2D rotation
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return rotated
```

---

## 为什么是 128？

### 常见的 head_dim 选择

| 模型 | emb_dim | n_heads | head_dim | 计算方式 |
|------|---------|---------|---------|---------|
| GPT-2 Small | 768 | 12 | **64** | 768 / 12 |
| GPT-2 Medium | 1024 | 16 | **64** | 1024 / 16 |
| GPT-3 | 12288 | 96 | **128** | 12288 / 96 |
| LLaMA 7B | 4096 | 32 | **128** | 4096 / 32 |
| **Qwen 0.6B** | **1024** | **16** | **128** | 1024 ÷ 16 ≠ 128! |

**等等！发现问题了！**

---

## Qwen 的特殊设计：Grouped Query Attention (GQA)

### 标准 MHA vs Qwen 的 GQA

#### 标准 Multi-Head Attention（MHA）

```python
# 标准配置：head_dim = emb_dim / n_heads
emb_dim = 1024
n_heads = 16
head_dim = 1024 / 16 = 64  # 自动计算

# 投影维度
Q_proj: 1024 → 1024  (16 × 64)
K_proj: 1024 → 1024  (16 × 64)
V_proj: 1024 → 1024  (16 × 64)
```

#### Qwen 的 Grouped Query Attention（GQA）

```python
# Qwen 配置：head_dim 显式指定为 128
emb_dim = 1024
n_heads = 16
head_dim = 128  # 显式设置（而不是 64！）
n_kv_groups = 8

# 投影维度
Q_proj: 1024 → 2048  (16 × 128)  ← Q 扩展！
K_proj: 1024 → 1024  (8 × 128)   ← 只有 8 组
V_proj: 1024 → 1024  (8 × 128)   ← 只有 8 组

# 每 2 个 Q 头共享 1 对 KV（16 / 8 = 2）
```

### GQA 的工作机制

```python
# Q: 16 个头，每个 128 维
Q_heads = [Q_0, Q_1, Q_2, Q_3, ..., Q_15]
        128维 128维 128维 128维     128维

# KV: 8 组，每组 128 维
KV_groups = [(K_0, V_0), (K_1, V_1), ..., (K_7, V_7)]
              128维      128维              128维

# 共享关系
Q_0, Q_1  → 共享 (K_0, V_0)
Q_2, Q_3  → 共享 (K_1, V_1)
Q_4, Q_5  → 共享 (K_2, V_2)
...
Q_14, Q_15 → 共享 (K_7, V_7)
```

### 为什么选择 head_dim=128？

#### 1. 更丰富的表征空间

```python
# 标准 GPT-2 风格
head_dim = 64
# 每个头只能表示 64 维的信息

# Qwen 风格
head_dim = 128
# 每个头可以表示 128 维的信息 → 2 倍表达能力！
```

#### 2. GQA 的权衡

Qwen 通过 **Grouped Query Attention** 平衡了效率和性能：

```python
# 标准 MHA 的 KV 缓存
KV_cache_size = 2 × seq_len × n_heads × head_dim
              = 2 × 512 × 16 × 64
              = 1,048,576

# Qwen GQA 的 KV 缓存
KV_cache_size = 2 × seq_len × n_kv_groups × head_dim
              = 2 × 512 × 8 × 128
              = 1,048,576  # 相同！

# 但 Qwen 的每个头有更大的维度（128 vs 64）
```

**优势**：
- KV 缓存大小不变（推理内存优化）
- 每个 Q 头有更大的表征空间（128 维）
- 性能接近标准 MHA，但内存效率更高

#### 3. 硬件对齐

```python
# GPU Tensor Core 的最优维度
128 是 2 的幂次
128 对齐到 CUDA warp 大小（32 线程）
128 × 128 的矩阵乘法在现代 GPU 上非常高效
```

---

## 不同模型规模的 head_dim

从 Qwen 代码中观察到的趋势：

| 模型 | emb_dim | n_heads | head_dim | n_kv_groups | 备注 |
|------|---------|---------|---------|-------------|------|
| Qwen 0.6B | 1024 | 16 | **128** | 8 | Q 扩展 2x |
| Qwen 1.7B | 2048 | 16 | **128** | 8 | 保持 128 |
| Qwen 4B | 2560 | 32 | **128** | 8 | 保持 128 |
| Qwen 7B | 3584 | 32 | **128** | 8 | 保持 128 |
| Qwen 14B | 5120 | 40 | **128** | 8 | 保持 128 |
| Qwen 32B | 5120 | 64 | **128** | 8 | 保持 128 |

**设计哲学**：
- **head_dim 固定为 128**（所有规模模型）
- 通过增加 `n_heads` 和 `emb_dim` 来扩展模型容量
- 保持 `n_kv_groups = 8`（或 4）以控制 KV 缓存大小

---

## 直观类比

### 类比1：餐厅的服务员和厨师

```
标准 MHA（16 个服务员，16 个厨师）：
- 每个服务员（Q 头）有自己专属的厨师（KV 对）
- 服务员记事本：64 页（head_dim=64）

Qwen GQA（16 个服务员，8 个厨师）：
- 每 2 个服务员共享 1 个厨师（节省人力）
- 服务员记事本：128 页（head_dim=128，更详细！）
- 厨师效率不变，但服务员记录更详细
```

### 类比2：侦察小队

```
标准配置：
- 16 个侦察兵，每人携带 64 页情报本

Qwen 配置：
- 16 个侦察兵，每人携带 128 页情报本
- 但情报分析组只有 8 个（每组服务 2 个侦察兵）
- 侦察兵能记录更详细，但共享分析资源
```

---

## head_dim 为什么必须是偶数？

### RoPE 的成对旋转

从代码中看到：
```python
def compute_rope_params(head_dim, ...):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # RoPE 成对旋转维度
    inv_freq = 1.0 / (theta_base ** (
        torch.arange(0, head_dim, 2) / head_dim
        #                          ↑
        #                          步长为 2
    ))
```

### 2D 旋转的数学原理

RoPE（Rotary Position Embedding）需要将维度**成对旋转**：

```python
# 维度划分（head_dim=128）
x = [x_0, x_1, x_2, x_3, ..., x_126, x_127]
     └─┬─┘ └─┬─┘            └────┬────┘
     pair0  pair1               pair63

# 每对应用 2D 旋转
[x_0']   = [cos θ  -sin θ] [x_0]
[x_1']     [sin θ   cos θ] [x_1]

# 总共 head_dim / 2 = 64 对
```

**类比**：就像时钟的时针和分针，必须成对旋转才能表示位置信息。

---

## 实际代码运行示例

```python
import torch
import torch.nn as nn

# Qwen 配置
emb_dim = 1024
n_heads = 16
head_dim = 128
n_kv_groups = 8

# 输入
x = torch.randn(2, 512, emb_dim)  # [batch, seq_len, emb_dim]

# Query 投影
W_query = nn.Linear(emb_dim, n_heads * head_dim, bias=False)
queries = W_query(x)
print(f"Queries shape: {queries.shape}")
# Output: [2, 512, 2048]  ← 注意：2048 = 16 × 128

# 重塑为多头
queries = queries.view(2, 512, n_heads, head_dim)
print(f"Queries (reshaped): {queries.shape}")
# Output: [2, 512, 16, 128]
#                   ↑   ↑
#                   |   └─ head_dim (每个头 128 维)
#                   └───── 16 个头

queries = queries.transpose(1, 2)
print(f"Queries (transposed): {queries.shape}")
# Output: [2, 16, 512, 128]
#            ↑        ↑
#            |        └─ 每个头操作 128 维向量
#            └────────── 16 个独立的注意力头

# Key/Value 投影（GQA）
W_key = nn.Linear(emb_dim, n_kv_groups * head_dim, bias=False)
keys = W_key(x)
print(f"Keys shape: {keys.shape}")
# Output: [2, 512, 1024]  ← 1024 = 8 × 128（只有 8 组）

keys = keys.view(2, 512, n_kv_groups, head_dim).transpose(1, 2)
print(f"Keys (final): {keys.shape}")
# Output: [2, 8, 512, 128]
#            ↑        ↑
#            |        └─ 同样 128 维
#            └────────── 只有 8 组 KV（16 个 Q 共享）

# 注意力计算（需要扩展 KV 以匹配 Q）
# keys: [2, 8, 512, 128] → repeat → [2, 16, 512, 128]
keys_expanded = keys.repeat_interleave(n_heads // n_kv_groups, dim=1)
print(f"Keys (expanded): {keys_expanded.shape}")
# Output: [2, 16, 512, 128]

# 现在可以计算注意力
attn_scores = queries @ keys_expanded.transpose(2, 3)
print(f"Attention scores shape: {attn_scores.shape}")
# Output: [2, 16, 512, 512]
```

---

## 为什么所有 Qwen 模型都用 head_dim=128？

### 1. 跨模型迁移学习

**好处**：
- 位置编码参数（RoPE）可以在不同规模模型间共享
- head_dim 固定 → RoPE 的 `cos/sin` 表可以预计算和复用
- 简化模型设计和调试

### 2. qk_norm 在 head_dim 级别操作

```python
if qk_norm:
    self.q_norm = RMSNorm(head_dim, eps=1e-6)  # ← 注意：head_dim
    self.k_norm = RMSNorm(head_dim, eps=1e-6)

# 归一化每个头的 128 维向量
queries = self.q_norm(queries)  # [batch, 16, seq_len, 128]
keys = self.k_norm(keys)        # [batch, 8, seq_len, 128]
```

### 3. 统一的工程实践

**优势**：
- 代码复用（RoPE、归一化等模块）
- 性能优化（针对 128 维优化一次，所有模型受益）
- 超参数调优（只需针对 head_dim=128 调优）

---

## 总结

### 核心要点

| 概念 | 要点 |
|------|------|
| **head_dim** | 每个注意力头独立处理的向量维度 |
| **标准关系** | emb_dim = n_heads × head_dim（标准 MHA） |
| **Qwen 设计** | head_dim=128，固定不变（所有规模） |
| **GQA 优化** | Q 头独立，KV 头共享，节省内存 |
| **RoPE 要求** | head_dim 必须是偶数（成对旋转） |

### 记忆口诀

> "head_dim 是头的眼界，128 维看世界"
> "16 个头各自看，合起来才完整"
> "Q 头各自记，KV 组里配"

### Qwen 的 GQA 优势

```
标准 MHA:
- 16 个 Q 头 × 64 维 = 1024 维
- 16 个 KV 对 × 64 维 = 1024 维 KV 缓存

Qwen GQA:
- 16 个 Q 头 × 128 维 = 2048 维（更强表达力）
- 8 个 KV 组 × 128 维 = 1024 维 KV 缓存（内存不变）

结果：性能↑，内存→
```

---

> 最后更新: 2026-04-08
