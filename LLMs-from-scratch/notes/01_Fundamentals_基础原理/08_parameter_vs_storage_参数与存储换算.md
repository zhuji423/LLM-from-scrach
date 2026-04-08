# 参数数量与存储大小的换算

> 深入理解"124M 参数"和"文件大小"的关系

---

## 问题：GPT-2 124M 参数为什么文件是 652.9MB 而不是 496MB？

### 快速验证

**标准 GPT-2 Small 应该是**：
```python
124.4M 参数 × 4 bytes (float32) = 497.6 MB ≈ 500 MB
```

**如果文件是 652.9MB**：
```python
652.9 MB ÷ 4 bytes = 163M 参数
```

**结论**：这不是标准的 124M 模型！

---

## 参数数量 vs 存储大小

### 核心换算公式

$$\text{存储大小 (bytes)} = \text{参数数量} \times \text{每参数字节数}$$

$$\text{存储大小 (MB)} = \frac{\text{参数数量} \times \text{字节数}}{1{,}048{,}576}$$

### 不同精度的存储大小

| 数据类型 | 每参数字节数 | 124M 参数占用 | 163M 参数占用 |
|---------|-------------|--------------|---------------|
| **float64** | 8 bytes | 992 MB | 1304 MB |
| **float32** | 4 bytes | **496 MB** | **652 MB** ✓ |
| **float16** | 2 bytes | 248 MB | 326 MB |
| **bfloat16** | 2 bytes | 248 MB | 326 MB |
| **int8** | 1 byte | 124 MB | 163 MB |

**从 652.9 MB 反推**：最接近 **163M × 4 bytes (float32) ≈ 652 MB**

---

## 标准 GPT-2 Small 的参数计算

### 配置

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

### 手动计算参数量

#### 1. 嵌入层

```python
# Token Embedding
token_emb = vocab_size × emb_dim
         = 50257 × 768
         = 38,597,376  (~38.6M)

# Position Embedding
pos_emb = context_length × emb_dim
        = 1024 × 768
        = 786,432  (~0.79M)
```

#### 2. 单个 Transformer 层

```python
# Multi-Head Attention (无 bias)
qkv_proj = 3 × (emb_dim × emb_dim)  # Q, K, V 投影
         = 3 × (768 × 768)
         = 1,769,472

out_proj = emb_dim × emb_dim        # 输出投影
         = 768 × 768
         = 589,824

# Feed-Forward Network (通常是 4 × emb_dim)
ffn_dim = 4 × emb_dim = 3072
fc1 = emb_dim × ffn_dim
    = 768 × 3072
    = 2,359,296

fc2 = ffn_dim × emb_dim
    = 3072 × 768
    = 2,359,296

# Layer Norm (两个)
ln = 2 × emb_dim × 2  # gamma 和 beta
   = 2 × 768 × 2
   = 3,072

# 单层总参数
per_layer = qkv_proj + out_proj + fc1 + fc2 + ln
          = 1,769,472 + 589,824 + 2,359,296 + 2,359,296 + 3,072
          = 7,080,960  (~7.08M)
```

#### 3. 总参数量

```python
total = token_emb + pos_emb + (n_layers × per_layer) + final_ln
      = 38,597,376 + 786,432 + (12 × 7,080,960) + 1,536
      = 38,597,376 + 786,432 + 84,971,520 + 1,536
      = 124,356,864  (~124.4M) ✓
```

#### 4. 存储大小（float32）

```python
storage = 124,356,864 × 4 bytes
        = 497,427,456 bytes
        = 497,427,456 / (1024 × 1024) MB
        = 474.3 MB ≈ 496 MB
```

---

## 为什么实际文件可能是 652.9MB（163M 参数）？

### 可能性 1：配置不同

**标准 GPT-2 Small**：
```python
emb_dim = 768
n_heads = 12
head_dim = 768 / 12 = 64
```

**可能的修改配置**：
```python
# 方案 A：增大 emb_dim
emb_dim = 1024  # 而不是 768
n_heads = 16
head_dim = 1024 / 16 = 64

# 重新计算参数量
token_emb = 50257 × 1024 = 51,463,168  (~51.5M)
per_layer ≈ 12.5M
total ≈ 51.5M + (12 × 12.5M) ≈ 201M  # 太大了

# 方案 B：使用 Qwen 风格的 head_dim
emb_dim = 1024
n_heads = 16
head_dim = 128  # 显式指定（而不是 64）

# Query 投影维度扩展
Q_dim = n_heads × head_dim = 16 × 128 = 2048  # > emb_dim
# 这会增加参数量
```

### 可能性 2：context_length 不同

**训练配置 vs 完整配置**：
```python
# 训练时的配置（你的 notebook）
"context_length": 256  # 缩短以减少计算

# 加载预训练权重时的配置
"context_length": 1024  # 恢复完整大小

# Position Embedding 的差异
pos_emb_train = 256 × 768 = 196,608     (~0.2M)
pos_emb_full = 1024 × 768 = 786,432     (~0.79M)
# 差异: ~0.6M 参数
```

但这个差异不足以解释 163M vs 124M 的差距。

### 可能性 3：包含额外的层或模块

```python
# 可能增加的组件
- 额外的输出投影层
- 分类头（classification head）
- 额外的 LayerNorm
- Value head（用于强化学习）
```

### 可能性 4：文件包含了优化器状态

```python
# 如果是完整 checkpoint
model_size = 124M × 4 bytes = 496 MB
optimizer_size = 124M × 4 × 2 = 992 MB  # Adam 的两份动量
total = 496 + 992 = 1488 MB ≈ 1.5 GB

# 但你说的是 652.9MB，不是 1.5GB
# 所以可能不包含优化器
```

---

## 实际验证方法

### 方法 1：代码计算

```python
import torch
from gpt_model import GPTModel

# 加载模型
config = GPT_CONFIG_124M
model = GPTModel(config)

# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# 计算存储大小
def get_model_size_mb(model):
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024**2)

param_count = count_parameters(model)
storage_mb = get_model_size_mb(model)

print(f"参数数量: {param_count:,} ({param_count/1e6:.1f}M)")
print(f"存储大小: {storage_mb:.1f} MB")
print(f"数据类型: {next(model.parameters()).dtype}")

# 验证文件大小
import os
checkpoint_path = "model_and_optimizer.pth"
file_size_mb = os.path.getsize(checkpoint_path) / (1024**2)
print(f"文件大小: {file_size_mb:.1f} MB")
```

### 方法 2：检查配置

```python
# 从 checkpoint 加载配置
checkpoint = torch.load("model_and_optimizer.pth")

# 检查模型状态字典
model_state = checkpoint["model_state_dict"]

# 打印关键参数的形状
for name, param in model_state.items():
    if "tok_emb" in name or "pos_emb" in name:
        print(f"{name}: {param.shape}")

# 输出示例
# tok_emb.weight: torch.Size([50257, 768])  → 标准
# pos_emb.weight: torch.Size([1024, 768])   → 标准

# 或者
# tok_emb.weight: torch.Size([50257, 1024]) → 扩展了！
# pos_emb.weight: torch.Size([1024, 1024])  → 扩展了！
```

### 方法 3：逐层统计

```python
def detailed_param_count(model):
    """详细统计每层参数"""
    total = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        param_size_mb = param_count * param.element_size() / (1024**2)
        total += param_count
        print(f"{name:50s} | {param_count:>12,} | {param_size_mb:>8.2f} MB")

    print("-" * 80)
    print(f"{'Total':50s} | {total:>12,} | {total * 4 / (1024**2):>8.2f} MB")

detailed_param_count(model)
```

**输出示例**：
```
tok_emb.weight                                     |   38,597,376 |   147.25 MB
pos_emb.weight                                     |      786,432 |     3.00 MB
trf_blocks.0.attn.W_query.weight                   |      589,824 |     2.25 MB
trf_blocks.0.attn.W_key.weight                     |      589,824 |     2.25 MB
...
--------------------------------------------------------------------------------
Total                                              |  124,356,864 |   474.30 MB
```

---

## 不同 GPT 模型的参数量对比

| 模型 | emb_dim | n_layers | n_heads | 参数量 | 存储大小 (FP32) |
|------|---------|---------|---------|--------|----------------|
| GPT-2 Small | 768 | 12 | 12 | **124M** | 496 MB |
| GPT-2 Medium | 1024 | 24 | 16 | **355M** | 1.36 GB |
| GPT-2 Large | 1280 | 36 | 20 | **774M** | 2.96 GB |
| GPT-2 XL | 1600 | 48 | 25 | **1.5B** | 5.72 GB |

---

## 实际观察：你的文件

你提到的文件大小：
```bash
model_and_optimizer.pth: 1.8GB
```

**拆解**：
```python
# 假设包含优化器
模型参数: ~600 MB (约 150M 参数)
优化器状态: ~1.2 GB (2 × 模型大小)
总计: ~1.8 GB ✓

# 参数量估算
600 MB ÷ 4 bytes = 150M 参数

# 这介于 GPT-2 Small (124M) 和 Medium (355M) 之间
```

**可能是自定义配置**：
```python
# 猜测的配置
emb_dim = 1024  # 比 Small 大
n_layers = 12   # 和 Small 相同
n_heads = 16    # 比 Small 多

# 或者
emb_dim = 896   # 介于 768 和 1024 之间
n_layers = 16   # 比 Small 多
```

---

## 关键区别总结

### 1. 参数量（Parameter Count）

**定义**：神经网络中可训练的数值**个数**

```python
param_count = 124,356,864  # 1.24 亿个浮点数
```

### 2. 存储大小（Storage Size）

**定义**：这些参数占用的**字节数**

```python
storage_size = param_count × bytes_per_param
             = 124,356,864 × 4 bytes
             = 497,427,456 bytes
             = 474.3 MB
```

### 3. 文件大小（File Size）

**定义**：保存到磁盘的实际大小

```python
file_size = model_size + optimizer_size + metadata
          = 496 MB + 992 MB + 几MB
          ≈ 1.5 GB (如果包含优化器)
          ≈ 500 MB (如果只有模型)
```

---

## 记忆口诀

> **"参数量看个数，文件大小看字节"**
>
> - 124M = 1.24亿个浮点数（参数个数）
> - 496MB = 存储空间（字节数）
> - 1.8GB = 模型 + 优化器

### 换算速记

```
参数量 (M) × 4 = 存储大小 (MB)  [float32]
参数量 (M) × 2 = 存储大小 (MB)  [float16]
参数量 (M) × 1 = 存储大小 (MB)  [int8]
```

---

## 总结

| 概念 | 值 | 说明 |
|------|-----|------|
| **标准 GPT-2 Small** | 124.4M 参数 | 标准配置 |
| **标准存储大小** | 496 MB | float32 |
| **你的文件（模型部分）** | ~653 MB | 约 163M 参数 |
| **你的文件（总大小）** | 1.8 GB | 包含优化器 |

**结论**：
- 你的模型**不是标准 GPT-2 Small**
- 参数量约为 **163M**（介于 Small 和 Medium 之间）
- 可能是**自定义配置**或**包含额外组件**

---

> 最后更新: 2026-04-08
