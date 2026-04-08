# 采样机制与温度参数详解

> 深入理解 LLM 文本生成中的采样策略、温度参数、top-k/top-p 过滤

---

## 问题1：为什么 torch.multinomial 采样只输出 3 种 token？

### 现象

```python
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
              for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))

    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

# 结果：只有 3 种不同的 token
```

### 根本原因

**`probas` 概率分布中只有 3 个位置是非零概率**。

```python
probas = [0.3, 0.5, 0.2, 0.0, 0.0, 0.0, ...]
         └────有效支撑集────┘  └───概率为0────┘
```

### torch.multinomial 的工作机制

`torch.multinomial` 是**离散概率分布的采样器**。

**轮盘赌类比**：
- 轮盘被分成若干扇区，每个扇区对应一个 token
- 扇区的面积大小由 `probas` 中对应位置的概率值决定
- **概率为 0 的 token 对应的扇区面积为 0** → 指针永远不会停在那里

### 数学表达

假设 `probas` 的形式是：

$$\text{probas} = [p_0, p_1, p_2, 0, 0, \ldots, 0]$$

其中 $\sum_{i=0}^{2} p_i = 1.0$，而其余位置概率均为 0。

那么：
```python
torch.multinomial(probas, num_samples=1)
```
**只能从索引 $\{0, 1, 2\}$ 中采样**，永远不会采样到索引 $\geq 3$ 的位置。

### 实际代码验证

```python
import torch

# 情况1：只有3个非零概率
probas = torch.tensor([0.3, 0.5, 0.2, 0.0, 0.0])
print(probas)  # tensor([0.3000, 0.5000, 0.2000, 0.0000, 0.0000])

# 采样1000次
torch.manual_seed(123)
samples = [torch.multinomial(probas, num_samples=1).item()
           for _ in range(1000)]
print(f"唯一值: {set(samples)}")  # 输出: {0, 1, 2}

# 统计分布
print(f"token 0: {samples.count(0)} 次")  # 约 300 次
print(f"token 1: {samples.count(1)} 次")  # 约 500 次
print(f"token 2: {samples.count(2)} 次")  # 约 200 次
```

---

## 为什么会这样设置？Top-k 和 Top-p 采样

在 LLM 的生成过程中，通常会对原始 logits 进行**过滤**：

### 1. Top-k 采样

**只保留概率最高的 k 个 token，其余设为 0**

```python
def top_k_sampling(logits, k=3):
    # 原始概率分布（假设）
    probas = torch.softmax(logits, dim=-1)
    # tensor([0.05, 0.30, 0.50, 0.10, 0.02, 0.01, 0.02])

    # Top-k 过滤：只保留前 k=3 名
    topk_values, topk_indices = torch.topk(probas, k)
    # topk_values: [0.50, 0.30, 0.10]
    # topk_indices: [2, 1, 3]

    # 创建新的概率分布（其余位置为0）
    filtered_probas = torch.zeros_like(probas)
    filtered_probas[topk_indices] = topk_values

    # 重新归一化
    filtered_probas = filtered_probas / filtered_probas.sum()
    # tensor([0.0, 0.33, 0.56, 0.11, 0.0, 0.0, 0.0])
    #              └─────只有3个非零─────┘

    return filtered_probas

# 采样：只会从这3个位置选择
token = torch.multinomial(filtered_probas, num_samples=1)
```

**流程图**：
```
原始 Softmax 输出
[0.05, 0.30, 0.50, 0.10, 0.02, 0.01, 0.02]
        ↓
Top-k=3 过滤（保留前3名）
[0.0, 0.33, 0.56, 0.11, 0.0, 0.0, 0.0]
        ↓
采样 → 只从 {1, 2, 3} 中选择
```

### 2. Top-p (Nucleus) 采样

**保留累积概率达到 p 的最小 token 集合**

```python
def top_p_sampling(logits, p=0.9):
    probas = torch.softmax(logits, dim=-1)
    # 排序（降序）
    sorted_probas, sorted_indices = torch.sort(probas, descending=True)

    # 计算累积概率
    cumulative_probas = torch.cumsum(sorted_probas, dim=-1)

    # 找到累积概率首次超过 p 的位置
    cutoff_index = torch.where(cumulative_probas > p)[0][0] + 1

    # 保留前 cutoff_index 个 token
    selected_probas = sorted_probas[:cutoff_index]
    selected_indices = sorted_indices[:cutoff_index]

    # 重新归一化
    selected_probas = selected_probas / selected_probas.sum()

    # 创建过滤后的分布
    filtered_probas = torch.zeros_like(probas)
    filtered_probas[selected_indices] = selected_probas

    return filtered_probas
```

**示例**：
```
原始概率（降序）: [0.50, 0.30, 0.10, 0.05, 0.02, 0.01, 0.01, 0.01]
累积概率:         [0.50, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 1.00]
                                  ↑
                          首次超过 p=0.9

保留前 3 个 token → 只有 3 个非零概率
```

### 为什么要过滤？

| 策略 | 候选数量 | 输出质量 | 多样性 |
|------|---------|---------|--------|
| **不过滤** | 50257（全部） | 低（包含大量低质量token） | 极高 |
| **Top-k=3** | 3 | 高（只选择高概率token） | 低 |
| **Top-p=0.9** | 动态（3-10） | 高 | 中等 |

---

## 问题2：为什么 logits/0.1 会让概率变得极小？

### 困惑点

```python
temperature = 0.1
scaled_logits = logits / temperature  # 等于 logits × 10

# 放大了10倍，为什么概率反而变小了？
```

### Softmax 的指数放大效应

Softmax 的数学定义：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

**当输入放大时，指数函数会产生"赢家通吃"效应。**

### 数值示例

假设原始 logits = `[2, 4, 6]`（简化版）

| Temperature | 实际输入 | 计算过程 | 结果概率分布 |
|-------------|----------|----------|--------------|
| **T=1** (原始) | `[2, 4, 6]` | $\frac{[e^2, e^4, e^6]}{e^2+e^4+e^6}$ | `[0.015, 0.117, 0.868]` |
| **T=0.1** (×10) | `[20, 40, 60]` | $\frac{[e^{20}, e^{40}, e^{60}]}{e^{20}+e^{40}+e^{60}}$ | `[2e-18, 4e-9, ~1.0]` |
| **T=5** (÷5) | `[0.4, 0.8, 1.2]` | $\frac{[e^{0.4}, e^{0.8}, e^{1.2}]}{...}$ | `[0.256, 0.316, 0.428]` |

### 为什么 $e^{60}$ 会"吃掉"所有概率？

指数函数的增长速度**极其恐怖**：

$$
\begin{align}
e^{60} &\approx 10^{26} \\
e^{40} &\approx 10^{17} \\
e^{20} &\approx 10^{9}
\end{align}
$$

所以：

$$\frac{e^{60}}{e^{20} + e^{40} + e^{60}} \approx \frac{10^{26}}{10^9 + 10^{17} + 10^{26}} \approx \frac{10^{26}}{10^{26}} \approx 1.0$$

**最大值的指数项完全主导了分母**，其他项可以忽略不计！

### 实际数据验证

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

next_token_logits = torch.tensor([...])  # 实际 logits

# Temperature values
temperatures = [1, 0.1, 5]

# Calculate scaled probabilities
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                 for T in temperatures]

print("T=1 (原始):")
print(scaled_probas[0])
# tensor([6.09e-02, 1.63e-03, ..., 5.72e-01, ..., 3.58e-01, ...])
# 最大值 0.572，次大值 0.358 - 分布相对平滑

print("\nT=0.1 (÷0.1 = ×10):")
print(scaled_probas[1])
# tensor([1.85e-10, 3.52e-26, ..., 9.91e-01, ..., 9.01e-03, ...])
# 最大值 0.991 ≈ 1，其他几乎为0 - 极度尖锐

print("\nT=5 (÷5):")
print(scaled_probas[2])
# tensor([0.1546, 0.0750, ..., 0.2421, ..., 0.2203, ...])
# 最大值 0.242，分布更均匀 - 更平滑
```

### 数学证明：Softmax 的极限行为

当 $\alpha \to \infty$（即 logits 无限放大）：

$$\lim_{\alpha \to \infty} \text{softmax}(\alpha \cdot \mathbf{x}) = \mathbf{one\_hot}(\arg\max(\mathbf{x}))$$

这意味着 **T→0 时，Softmax 退化为 one-hot 向量**（只有最大值处为1，其他全0）。

---

## 问题3：温度参数的顺逻辑记忆方法

### 最佳记忆法：物理温度类比

| 温度值 | 物理现象 | 分布形状 | 模型行为 |
|--------|----------|----------|----------|
| **T 大（高温）** | 分子到处乱跑，活动范围大 | **宽** | 不确定，探索多样性 |
| **T 小（低温）** | 分子冻结，聚集在最低点 | **窄** | 确定，专注高概率 |

### 顺逻辑关系（同向变化）

$$T \uparrow \Rightarrow \text{分布宽度} \uparrow$$

**记忆口诀**：
> "高温乱飞四处跑，低温冻僵缩一团"
> "温度跟着宽度走"
> "热胀冷缩"

### 多种类比帮助记忆

#### 类比1：图像处理滤镜

| 温度值 | 图像效果 | 分布形状 |
|--------|----------|----------|
| **T 大** | 模糊滤镜 - 边界柔和，细节丢失 | **宽** |
| **T 小** | 锐化滤镜 - 边界清晰，对比强烈 | **窄** |

#### 类比2：赌博奖金分配

| 温度值 | 奖金分配 | 中奖概率 |
|--------|----------|----------|
| **T 大** | 均分奖金 - 每人都有份 | 人人有机会 |
| **T 小** | 赢家通吃 - 冠军拿走99% | 只有最强者 |

#### 类比3：确定性/信心视角

| 温度值 | 心理状态 | 行为 |
|--------|----------|------|
| **T 大** | "我不太确定...各种可能都试试吧" | 探索、多样化 |
| **T 小** | "我非常确定就是这个！" | 利用、专注 |

### 实际应用场景

```python
# 创意写作（高多样性）
temperature = 1.5  # 高温 → 宽分布 → 更随机

# 代码生成（高准确性）
temperature = 0.3  # 低温 → 窄分布 → 更确定

# 聊天对话（平衡）
temperature = 0.7  # 中温 → 中等分布 → 平衡
```

---

## 问题4：如何增加采样的多样性？

### 方法1：提高温度参数

```python
# 原始温度
temperature = 1.0
probas = softmax(logits / temperature)
# 结果：[0.3, 0.5, 0.2]  # 3种token

# 提高温度
temperature = 2.0
probas = softmax(logits / temperature)
# 结果：[0.28, 0.42, 0.22, 0.05, 0.03]  # 5种token（更多选择）
```

### 方法2：调整 top-k 参数

```python
# 限制性强
top_k = 3
# 只能输出 3 种 token

# 增加多样性
top_k = 10
# 可以输出 10 种 token
```

### 方法3：使用 top-p (nucleus) 采样

```python
# 动态候选数量
top_p = 0.9
# 根据概率分布动态选择 token 数量（通常 5-15 个）
```

### 方法4：组合策略

```python
def advanced_sampling(logits, temperature=1.0, top_k=50, top_p=0.9):
    # 1. 温度缩放
    scaled_logits = logits / temperature
    probas = torch.softmax(scaled_logits, dim=-1)

    # 2. Top-k 过滤
    if top_k > 0:
        topk_values, topk_indices = torch.topk(probas, top_k)
        probas = torch.zeros_like(probas)
        probas[topk_indices] = topk_values

    # 3. Top-p 过滤
    if top_p < 1.0:
        sorted_probas, sorted_indices = torch.sort(probas, descending=True)
        cumulative_probas = torch.cumsum(sorted_probas, dim=-1)

        remove_indices = cumulative_probas > top_p
        remove_indices[0] = False  # 至少保留1个

        sorted_probas[remove_indices] = 0
        probas = torch.zeros_like(probas)
        probas[sorted_indices] = sorted_probas

    # 4. 重新归一化
    probas = probas / probas.sum()

    # 5. 采样
    return torch.multinomial(probas, num_samples=1)
```

---

## 问题5：优化器参数需要保存吗？

### 快速判断

| 场景 | 是否保存优化器 | 原因 |
|------|---------------|------|
| **中断后继续训练** | ✅ **必须** | 需要恢复动量状态 |
| **仅推理/部署** | ❌ 不需要 | 只需模型权重 |
| **微调新任务** | ❌ 不需要 | 创建新优化器 |

### Adam 优化器的内部状态

Adam 不只是"学习率"，它维护着**复杂的历史信息**：

```python
# Adam 的内部状态（每个参数都有）
state = {
    "step": t,                    # 步数计数器
    "exp_avg": m_t,              # 一阶动量（梯度的移动平均）
    "exp_avg_sq": v_t,           # 二阶动量（梯度平方的移动平均）
}
```

### 开车类比

| 元素 | 对应优化器 |
|------|-----------|
| **当前位置** | 模型参数 $\theta$ |
| **车速和方向** | 一阶动量 $m_t$ |
| **加速度历史** | 二阶动量 $v_t$ |
| **行驶里程** | 步数 $t$ |

**场景对比**：

**保存优化器**：
- 记住了"车速+方向+加速度"
- 重启后平滑接续
- 训练曲线稳定

**不保存优化器**：
- 重启后"车速归零"
- 需要重新加速
- 训练曲线会抖动，浪费几百到几千步

### 最佳实践

```python
# ✅ 训练中定期保存（完整 checkpoint）
if step % 1000 == 0:
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),  # 保存
        "loss": loss.item(),
    }, f"checkpoint_step_{step}.pth")

# ✅ 最终部署（只保存模型）
torch.save(model.state_dict(), "model_final.pth")  # 不保存优化器

# ✅ 恢复训练
checkpoint = torch.load("checkpoint.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 恢复状态
```

---

## 信息熵视角理解温度

### 信息熵定义

$$H(p) = -\sum_i p_i \log p_i$$

熵值越大，分布越"不确定"。

### 温度对熵的影响

| Temperature | 熵值 | 解释 |
|-------------|------|------|
| **T=0.1** | 接近0 | 几乎确定性（低多样性） |
| **T=1** | 中等 | 平衡状态 |
| **T=5** | 接近最大 | 高度不确定（高多样性） |

### 可视化

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

logits = torch.tensor([2.0, 4.0, 6.0, 3.0, 1.0])
temperatures = np.linspace(0.1, 5.0, 50)

entropies = []
for T in temperatures:
    probas = torch.softmax(logits / T, dim=-1)
    entropy = -(probas * torch.log(probas + 1e-10)).sum().item()
    entropies.append(entropy)

plt.plot(temperatures, entropies)
plt.xlabel('Temperature')
plt.ylabel('Entropy (bits)')
plt.title('Temperature vs Information Entropy')
plt.grid(True)
plt.show()
```

---

## 总结

### 核心概念

| 概念 | 要点 |
|------|------|
| **采样机制** | multinomial 只从非零概率位置采样 |
| **Top-k 过滤** | 保留前 k 个高概率 token |
| **Top-p 过滤** | 保留累积概率达到 p 的 token 集合 |
| **温度参数** | 控制分布的"宽度"，低温→尖锐，高温→平滑 |
| **Softmax 效应** | 指数放大导致"赢家通吃" |
| **优化器状态** | 训练必须保存，推理不需要 |

### 实用技巧

1. **调试采样问题**：检查概率分布的非零元素个数
2. **温度选择**：创意任务用高温(1.5-2.0)，精确任务用低温(0.3-0.7)
3. **组合策略**：temperature + top-p 是最常用的组合
4. **保存策略**：训练中保存完整 checkpoint，部署只保存模型

---

> 最后更新: 2026-04-08
