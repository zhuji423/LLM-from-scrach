# Chapter 02: Working with Text Data - 学习笔记

## 📋 核心概念速览

本章聚焦于 **文本数据的预处理与表示**，将原始文本转换为大语言模型可处理的数值张量。

---

## 🔄 完整数据流 Pipeline

```
原始文本 (Raw Text)
    ↓ Tokenization
Token 序列 (Tokens)
    ↓ Encoding
Token IDs 序列 (Integer IDs)
    ↓ Embedding Lookup
Token Embeddings (Dense Vectors)
    ↓ Position Encoding
Input Embeddings (Token + Position)
    ↓
送入 LLM
```

---

## 1️⃣ Tokenization（分词）

### 1.1 定义与作用

**Tokenizer** 负责将字符串映射到离散符号空间（Token ID 序列）

```python
# 示例
"Hello, world!" → ["Hello", ",", "world", "!"] → [15496, 11, 995, 0]
```

### 1.2 核心概念

| 术语 | 定义 | 示例 |
|------|------|------|
| **Token** | 文本的基本单元（词/子词/字符） | "Hello", ",", "world" |
| **Token ID** | Token 在词表中的索引（整数） | 15496, 11, 995 |
| **Vocabulary** | 所有 Token 到 ID 的映射字典 | {"Hello": 15496, ",": 11, ...} |

### 1.3 特殊标记（Special Tokens）

GPT-2 的设计哲学：**极简主义**

```python
vocab = {
    "normal_tokens": {...},  # 普通词汇
    "<|endoftext|>": 50256   # 唯一特殊标记
}
```

**`<|endoftext|>` 的三重角色：**
- 📄 文本分隔符（区分不同文档）
- 🛑 序列结束标记（[EOS]）
- 📦 填充标记（[PAD]，配合 attention mask）

**对比其他模型：**
- BERT 使用：`[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, `[UNK]`
- GPT-2 理念：用 BPE 处理未登录词，无需 `[UNK]`

---

## 2️⃣ Embedding（嵌入层）

### 2.1 Token Embedding

**核心原理：** 查表操作（Lookup Table），非 one-hot 矩阵乘法

```python
# 嵌入层实现
embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)

# 输入: Token IDs [40, 367, 2885]
# 输出: 3D Tensor [3, embed_dim]
embeddings = embedding_layer(token_ids)
```

### 2.2 参数量对比

| 模型 | 词表大小 | 嵌入维度 | 参数量 | 占比 |
|------|----------|----------|--------|------|
| **BERT-base** | 30,522 | 768 | 23.4M | 21% / 110M |
| **GPT-2-small** | 50,257 | 768 | 38.8M | 31% / 124M |

**计算公式：** `参数量 = vocab_size × embed_dim`

### 2.3 与 One-Hot 的关系

虽然数学上等价，但实现不同：

```
传统方法（低效）：
Token ID → One-Hot Vector → 矩阵乘法 → Embedding

现代方法（高效）：
Token ID → 直接索引 → Embedding
```

**类比：** 不是先打印书单再找书，而是直接根据编号取书 📚

---

## 3️⃣ Position Encoding（位置编码）

### 3.1 为什么需要位置信息？

**问题根源：** Self-Attention 天生对位置不敏感

```python
# 对于 Attention 来说，以下两句是等价的：
"I love you"  ≈  "you love I"  # ❌ 语义完全不同！
```

### 3.2 GPT-2 的方案：可学习的绝对位置编码

```python
# 位置编码层（与 token embedding 维度相同）
pos_embedding_layer = torch.nn.Embedding(max_length, embed_dim)

# 为每个位置生成编码
pos_embeddings = pos_embedding_layer(torch.arange(max_length))

# 最终输入表示
input_embeddings = token_embeddings + pos_embeddings
```

**数学表达：**

$$
\mathbf{h}_i^{(0)} = \mathbf{E}[x_i] + \mathbf{P}[i]
$$

其中：
- $\mathbf{E}$：Token Embedding 矩阵
- $\mathbf{P}$：Position Embedding 矩阵
- $x_i$：第 $i$ 个 Token ID
- $i$：位置索引

### 3.3 对比不同方案

| 方案 | 优点 | 缺点 | 使用模型 |
|------|------|------|----------|
| **可学习绝对位置** | 简单、有效 | 受限于预训练的 max_length | GPT-2, BERT |
| **正弦编码** | 可推广到更长序列 | 不可学习 | Transformer 原论文 |
| **相对位置编码** | 泛化性更好 | 实现复杂 | T5, DeBERTa |

---

## 4️⃣ Data Loading（数据加载）

### 4.1 滑动窗口机制

**目标：** 训练模型预测下一个词

```
原始序列: [40, 367, 2885, 1464, 1807, 3619, ...]
              ↓ 滑动窗口 (max_length=4)
Input:  [40, 367, 2885, 1464]
Output: [   367, 2885, 1464, 1807]  # 向右偏移 1 位
```

### 4.2 Stride 参数的权衡

| Stride 设置 | 训练速度 | 数据利用 | 过拟合风险 | 适用场景 |
|-------------|----------|----------|------------|----------|
| **stride = max_length** | ⚡ 快 | 低（无重叠） | 低 | 大数据集、快速实验 |
| **stride < max_length** | 🐢 慢 | 高（有重叠） | 高 | 小数据集、充分训练 |

**可视化示例：**

```
stride = max_length (无重叠)
[A B C D] [E F G H] [I J K L] ...
  ↑不重叠     ↑不重叠

stride = 2 (50% 重叠)
[A B C D]
    [C D E F]
        [E F G H] ...
        ↑重叠区域
```

### 4.3 Batch 维度

```python
# DataLoader 输出
batch = {
    'input_ids':  torch.Size([Batch, Length]),      # [8, 4]
    'embeddings': torch.Size([Batch, Length, Dim])  # [8, 4, 256]
}
```

**维度解读：**
- **Batch**：并行处理的样本数（如 8）
- **Length**：序列长度 / 上下文窗口（如 4）
- **Dim**：嵌入维度（如 256 或 768）

---

## 5️⃣ 关键代码片段

### 5.1 完整流程示例

```python
import torch
import tiktoken

# 1. 初始化 Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# 2. 文本编码
text = "Hello, world! <|endoftext|> New document."
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# 3. Token Embedding
vocab_size = 50257
embed_dim = 768
token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
token_vecs = token_embedding(torch.tensor(token_ids))

# 4. Position Embedding
max_length = len(token_ids)
pos_embedding = torch.nn.Embedding(max_length, embed_dim)
pos_vecs = pos_embedding(torch.arange(max_length))

# 5. 最终输入表示
input_embeddings = token_vecs + pos_vecs
# Shape: [seq_len, embed_dim]
```

### 5.2 DataLoader 构建

```python
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(token_ids[i:i + max_length])
            self.target_ids.append(token_ids[i + 1:i + max_length + 1])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.input_ids[idx]),
            torch.tensor(self.target_ids[idx])
        )

# 使用示例
dataloader = DataLoader(
    GPTDatasetV1(raw_text, tokenizer, max_length=256, stride=128),
    batch_size=8,
    shuffle=True
)
```

---

## 🎯 本章核心要点总结

### ✅ 必须掌握

1. **Tokenization 三概念**：Token、Token ID、Vocabulary
2. **Embedding 本质**：查表操作，非矩阵乘法
3. **Position Encoding**：弥补 Attention 的位置盲区
4. **数据流维度变化**：
   ```
   文本 → [seq_len] → [seq_len, embed_dim] → [batch, seq_len, embed_dim]
   ```

### 💡 设计哲学

- **GPT-2 的极简主义**：单一特殊标记 `<|endoftext|>`
- **BPE 的优势**：无需 `[UNK]`，子词级别处理未登录词
- **Stride 的折衷**：速度 vs 数据利用率

### 🔗 与下一章的连接

本章准备的 `input_embeddings` 将作为 **Self-Attention** 的输入，进入 Transformer 的核心：

```
input_embeddings → Multi-Head Attention → Feed Forward → Output
```

---

## 📚 延伸阅读

- **BPE 原理**：[Byte Pair Encoding 详解](../02_bonus_bytepair-encoder/)
- **Embedding vs Matrix Multiplication**：[数学等价性证明](../03_bonus_embedding-vs-matmul/)
- **相对位置编码**：T5、DeBERTa 等模型的改进方案

---

## 🤔 自我检验

完成本章后，你应该能回答：

1. 为什么 GPT-2 不需要 `[PAD]` 和 `[UNK]` 标记？
2. Embedding 层的参数量如何计算？在整个模型中占比多少？
3. 如果不加位置编码，模型会出现什么问题？
4. Stride=1 和 Stride=max_length 分别适用于什么场景？
5. `input_embeddings` 的三个维度分别代表什么？

---

**最后更新：** 2026-04-01
**下一章预告：** Chapter 03 - Attention Mechanisms 🔍
