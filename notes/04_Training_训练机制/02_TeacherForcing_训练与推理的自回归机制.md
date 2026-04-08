# 训练 vs 推理：自回归机制详解

> 核心问题：训练时如何处理自回归？每次输出不是要变成输入再继续吗？

---

## 根本区别

```
┌─────────────────────────────────────────────────────────────────────┐
│                         训练阶段                                     │
│                    (Teacher Forcing)                                │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  输入:  [The]  [cat]  [sat]  [on]   [the]                          │
│  目标:  [cat]  [sat]  [on]   [the]  [mat]    ← 右移一位             │
│                                                                     │
│  ⚡ 并行计算！一次前向传播，5个位置同时预测                           │
│  ⚡ 不需要等上一个输出！用 Causal Mask 模拟自回归                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         推理阶段                                     │
│                   (Auto-regressive)                                 │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Step 1: [The] → 预测 → [cat]                                       │
│  Step 2: [The][cat] → 预测 → [sat]                                  │
│  Step 3: [The][cat][sat] → 预测 → [on]                              │
│  Step 4: [The][cat][sat][on] → 预测 → [the]                         │
│  Step 5: [The][cat][sat][on][the] → 预测 → [mat]                    │
│                                                                     │
│  🐢 串行计算！必须等上一步输出，才能进行下一步                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 训练阶段：Teacher Forcing

### 数据构造

```python
# 训练数据构造
text = "The cat sat on the mat"
tokens = [The, cat, sat, on, the, mat]  # 假设已分词

# 构造输入和目标（右移一位）
input_ids  = [The, cat, sat, on,  the]   # 去掉最后一个
target_ids = [cat, sat, on,  the, mat]   # 去掉第一个

# 一次前向传播
logits = model(input_ids)  # [batch, seq_len, vocab_size]
loss = cross_entropy(logits, target_ids)  # 5个位置的loss平均
```

### Causal Mask（因果掩码）

```
                    Query 位置
                The   cat   sat   on   the
              ┌─────┬─────┬─────┬─────┬─────┐
        The   │  ✓  │  ✗  │  ✗  │  ✗  │  ✗  │
Key          ├─────┼─────┼─────┼─────┼─────┤
位置    cat   │  ✓  │  ✓  │  ✗  │  ✗  │  ✗  │
              ├─────┼─────┼─────┼─────┼─────┤
        sat   │  ✓  │  ✓  │  ✓  │  ✗  │  ✗  │
              ├─────┼─────┼─────┼─────┼─────┤
        on    │  ✓  │  ✓  │  ✓  │  ✓  │  ✗  │
              ├─────┼─────┼─────┼─────┼─────┤
        the   │  ✓  │  ✓  │  ✓  │  ✓  │  ✓  │
              └─────┴─────┴─────┴─────┴─────┘

✓ = 可以看到    ✗ = 看不到（mask 掉）

位置 3 (sat) 只能看到 [The, cat, sat] → 预测 "on"
位置 5 (the) 可以看到全部 → 预测 "mat"
```

### 为什么高效？

| 对比项 | Teacher Forcing | 真正自回归 |
|-------|-----------------|-----------|
| 一次前向传播 | 5 个预测任务 | 1 个预测任务 |
| 生成 5 个词 | 1 次前向 | 5 次前向 |
| GPU 利用率 | 高（并行） | 低（串行） |
| 训练速度 | 快 100x+ | 极慢 |

---

## 推理阶段：真正的 Auto-regressive

```python
def generate(model, prompt_ids, max_new_tokens=50):
    generated = prompt_ids.clone()  # [1, prompt_len]

    for _ in range(max_new_tokens):
        # 每次都要重新计算整个序列
        logits = model(generated)           # [1, current_len, vocab]
        next_logits = logits[:, -1, :]      # 只取最后一个位置
        next_token = torch.argmax(next_logits, dim=-1)

        # 拼接到输入后面
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        if next_token == eos_token_id:
            break

    return generated
```

### 推理循环图示

```
Step 1:
输入: [BOS]
      ↓ 模型
输出: logits → sample → "The"

Step 2:
输入: [BOS, The]
      ↓ 模型（重新计算整个序列！）
输出: logits[-1] → sample → "cat"

Step 3:
输入: [BOS, The, cat]
      ↓ 模型（又重新计算整个序列！）
输出: logits[-1] → sample → "sat"

... 循环直到 [EOS] 或 max_length
```

---

## Padding 处理

### 训练时的 Padding（右 padding）

```
Batch 中不同长度的序列:
序列1: [The, cat, sat]           长度 3
序列2: [I, love, deep, learning] 长度 4
序列3: [Hi]                      长度 1

Padding 对齐到 max_length=4:
序列1: [The, cat, sat, PAD]      + attention_mask: [1,1,1,0]
序列2: [I, love, deep, learning] + attention_mask: [1,1,1,1]
序列3: [Hi, PAD, PAD, PAD]       + attention_mask: [1,0,0,0]

attention_mask 确保:
1. PAD 位置不参与 attention 计算
2. PAD 位置的 loss 不计入总 loss
```

### 推理时的 Padding（左 padding）

```
右 padding（训练常用）:
[The, cat, sat, PAD, PAD]
                ↑ 最后一个有效位置在 index 2
预测时需要找到正确位置

左 padding（推理常用）:
[PAD, PAD, The, cat, sat]
                      ↑ 统一取 logits[-1]
更简单！
```

---

## KV Cache：推理加速

```
没有 KV Cache（朴素版）:
Step 3: [The, cat, sat]
    重新计算 The 的 K,V  ← 重复！
    重新计算 cat 的 K,V  ← 重复！
    重新计算 sat 的 K,V

有 KV Cache（优化版）:
Step 1: [The] → 计算并缓存 K₁,V₁
Step 2: [cat] → 计算 K₂,V₂，复用 K₁,V₁
Step 3: [sat] → 计算 K₃,V₃，复用 K₁,V₁,K₂,V₂

只计算新 token 的 K,V！
时间复杂度从 O(n²) 降到 O(n)
```

---

## 对比总结表

| 维度 | 训练 | 推理 |
|------|------|------|
| **机制名称** | Teacher Forcing | Auto-regressive |
| **计算方式** | 并行所有位置 | 串行逐个生成 |
| **自回归模拟** | Causal Mask | 真正的循环 |
| **一次前向** | N 个预测任务 | 1 个 token |
| **输入来源** | 真实 target | 模型自己的输出 |
| **Padding** | 右 padding + mask | 左 padding 或 KV Cache |
| **效率** | 极高 | 较低（推理瓶颈） |

---

## 关键洞察

1. **训练时不需要真正自回归**
   - 用 Causal Mask "假装"是自回归
   - 并行计算，高效利用 GPU

2. **推理时必须真正自回归**
   - 因为不知道下一个词是什么
   - 必须等模型输出后才能继续
   - 这就是 LLM 推理慢的根本原因

3. **KV Cache 是推理加速的关键**
   - 避免重复计算历史 token 的 K,V
   - 是工程优化，不改变结果

---

> 最后更新: 2026-04-07
