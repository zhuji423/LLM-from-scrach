# 反向传播中的链式法则 - 可视化详解

## 🎯 核心思想

**链式法则 (Chain Rule)** 是微积分中的基本规则，它让我们能够计算**复合函数**的导数。

**简单表达：**
```
如果 z = f(y) 且 y = g(x)，那么：
∂z/∂x = (∂z/∂y) × (∂y/∂x)
```

---

## 📊 案例 1：最简单的单层网络

### 计算图 (Computational Graph)

```
     x                w                b
     │                │                │
     └────────►[×]◄───┘                │
              │                       │
              └───────►[+]◄───────────┘
                       │
                       ▼
                     output
                       │
                       ▼
                  [(output-target)²]
                       │
                       ▼
                     loss
```

### 具体数值示例

```python
# 给定数值
x = 2.0
w = 0.5
b = 0.3
target = 0.0

# 前向传播
z = w * x        # z = 0.5 × 2.0 = 1.0
output = z + b   # output = 1.0 + 0.3 = 1.3
loss = (output - target)²  # loss = (1.3 - 0)² = 1.69
```

### 反向传播：逐步计算梯度

#### 步骤 1：计算 ∂loss/∂output

```python
# loss = (output - target)²
# ∂loss/∂output = 2(output - target)

∂loss/∂output = 2 × (1.3 - 0) = 2.6
```

#### 步骤 2：计算 ∂loss/∂b（偏置的梯度）

```python
# 应用链式法则
# output = z + b
# ∂output/∂b = 1

∂loss/∂b = ∂loss/∂output × ∂output/∂b
         = 2.6 × 1
         = 2.6
```

**可视化：**
```
loss ──2.6──> output ──1──> b
              ↑
         (∂loss/∂output) (∂output/∂b)
```

#### 步骤 3：计算 ∂loss/∂z（中间变量的梯度）

```python
# 应用链式法则
# output = z + b
# ∂output/∂z = 1

∂loss/∂z = ∂loss/∂output × ∂output/∂z
         = 2.6 × 1
         = 2.6
```

#### 步骤 4：计算 ∂loss/∂w（权重的梯度）

```python
# 应用链式法则
# z = w × x
# ∂z/∂w = x = 2.0

∂loss/∂w = ∂loss/∂z × ∂z/∂w
         = 2.6 × 2.0
         = 5.2
```

**完整链式展开：**
```
∂loss/∂w = (∂loss/∂output) × (∂output/∂z) × (∂z/∂w)
         = 2.6 × 1 × 2.0
         = 5.2
```

**可视化：**
```
loss ──2.6──> output ──1──> z ──2.0──> w
              ↑            ↑          ↑
         (∂loss/∂output) (∂output/∂z) (∂z/∂w)
```

---

## 📊 案例 2：两层神经网络

### 网络结构

```
Input → [Linear1] → [ReLU] → [Linear2] → Output → Loss
```

### 详细计算图

```
       x          w1         b1
       │           │          │
       └───►[×]◄───┘          │
            │                 │
            └─────►[+]◄───────┘
                    │
                    ▼
                   h1 (hidden)
                    │
                    ▼
               [ReLU(h1)]
                    │
                    ▼
                   a1 (activation)
                    │          w2         b2
                    │           │          │
                    └───►[×]◄───┘          │
                         │                 │
                         └─────►[+]◄───────┘
                                 │
                                 ▼
                              output
                                 │
                                 ▼
                         [(output-target)²]
                                 │
                                 ▼
                               loss
```

### 具体数值示例

```python
# 给定数值
x = 2.0
w1 = 0.5
b1 = -0.3
w2 = 1.2
b2 = 0.1
target = 0.0

# 前向传播
# Layer 1
h1 = w1 * x + b1          # h1 = 0.5×2.0 + (-0.3) = 0.7
a1 = max(0, h1)           # a1 = ReLU(0.7) = 0.7

# Layer 2
h2 = w2 * a1 + b2         # h2 = 1.2×0.7 + 0.1 = 0.94
output = h2               # output = 0.94

# Loss
loss = (output - target)² # loss = (0.94 - 0)² = 0.8836
```

### 反向传播：完整链式推导

#### 第 1 步：∂loss/∂output

```python
∂loss/∂output = 2(output - target)
              = 2 × (0.94 - 0)
              = 1.88
```

#### 第 2 步：∂loss/∂w2（第二层权重）

```python
# 链式法则：
# loss → output → h2 → w2

∂loss/∂w2 = (∂loss/∂output) × (∂output/∂h2) × (∂h2/∂w2)
          = 1.88 × 1 × a1
          = 1.88 × 1 × 0.7
          = 1.316
```

**详细分解：**
```
∂output/∂h2 = 1           (因为 output = h2)
∂h2/∂w2 = a1 = 0.7        (因为 h2 = w2×a1 + b2)
```

#### 第 3 步：∂loss/∂b2（第二层偏置）

```python
∂loss/∂b2 = (∂loss/∂output) × (∂output/∂h2) × (∂h2/∂b2)
          = 1.88 × 1 × 1
          = 1.88
```

#### 第 4 步：∂loss/∂a1（第一层激活值）

```python
# 这是关键的中间梯度！
∂loss/∂a1 = (∂loss/∂output) × (∂output/∂h2) × (∂h2/∂a1)
          = 1.88 × 1 × w2
          = 1.88 × 1.2
          = 2.256
```

**为什么重要？** 这个梯度会继续传播到第一层的参数。

#### 第 5 步：∂loss/∂h1（通过 ReLU）

```python
# ReLU 的导数：
# d(ReLU(x))/dx = { 1  if x > 0
#                 { 0  if x ≤ 0

# 因为 h1 = 0.7 > 0，所以：
∂a1/∂h1 = 1

∂loss/∂h1 = (∂loss/∂a1) × (∂a1/∂h1)
          = 2.256 × 1
          = 2.256
```

**注意：** 如果 h1 < 0，这里的梯度会变成 0（梯度消失）！

#### 第 6 步：∂loss/∂w1（第一层权重）

```python
# 完整的链式法则：
∂loss/∂w1 = (∂loss/∂output) × (∂output/∂h2) × (∂h2/∂a1) ×
            (∂a1/∂h1) × (∂h1/∂w1)
          = 1.88 × 1 × 1.2 × 1 × x
          = 2.256 × 2.0
          = 4.512
```

**详细分解：**
```
∂h1/∂w1 = x = 2.0        (因为 h1 = w1×x + b1)
```

#### 第 7 步：∂loss/∂b1（第一层偏置）

```python
∂loss/∂b1 = (∂loss/∂output) × (∂output/∂h2) × (∂h2/∂a1) ×
            (∂a1/∂h1) × (∂h1/∂b1)
          = 1.88 × 1 × 1.2 × 1 × 1
          = 2.256
```

---

## 🎨 可视化：梯度流动图

### 前向传播（从左到右）

```
x=2.0 ──┐
        ├──[×0.5]──> h1=0.7 ──[ReLU]──> a1=0.7 ──┐
b1=-0.3─┘                                        ├──[×1.2]──> h2=0.94 ──> output=0.94
                                       b2=0.1 ───┘
                                                          │
                                                          ▼
                                                      loss=0.8836
```

### 反向传播（从右到左）

```
                                    ┌─────────────────────────────────────┐
                                    │        ∂loss/∂output = 1.88         │
                                    └─────────────────────────────────────┘
                                                        │
                                                        ▼
┌──────────────┐                           ┌─────────────────────┐
│∂loss/∂w2     │                           │   ∂loss/∂a1 = 2.256 │
│= 1.88×1×0.7  │◄──────────────────────────┤                     │
│= 1.316       │                           └─────────────────────┘
└──────────────┘                                       │
                                                       ▼ (通过 ReLU)
┌──────────────┐                           ┌─────────────────────┐
│∂loss/∂b2     │                           │   ∂loss/∂h1 = 2.256 │
│= 1.88        │                           └─────────────────────┘
└──────────────┘                                   │         │
                                                   ▼         ▼
                                        ┌──────────────┐  ┌──────────────┐
                                        │∂loss/∂w1     │  │∂loss/∂b1     │
                                        │= 2.256×2.0   │  │= 2.256       │
                                        │= 4.512       │  └──────────────┘
                                        └──────────────┘
```

---

## 💻 Python 验证代码

让我们用 PyTorch 验证上面的手算结果：

```python
import torch
import torch.nn as nn

# 创建一个简单的两层网络
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1, bias=True)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(1, 1, bias=True)

        # 手动设置权重为我们的示例值
        with torch.no_grad():
            self.layer1.weight.fill_(0.5)
            self.layer1.bias.fill_(-0.3)
            self.layer2.weight.fill_(1.2)
            self.layer2.bias.fill_(0.1)

    def forward(self, x):
        h1 = self.layer1(x)
        a1 = self.relu(h1)
        output = self.layer2(a1)
        return output

# 创建模型和输入
model = TwoLayerNet()
x = torch.tensor([[2.0]], requires_grad=True)
target = torch.tensor([[0.0]])

# 前向传播
output = model(x)
loss = nn.MSELoss()(output, target)

# 反向传播
loss.backward()

# 打印结果
print("=" * 60)
print("前向传播结果：")
print("=" * 60)
print(f"Output: {output.item():.4f}")
print(f"Loss: {loss.item():.4f}")

print("\n" + "=" * 60)
print("反向传播梯度（自动计算）：")
print("=" * 60)
print(f"∂loss/∂w1 (layer1.weight.grad): {model.layer1.weight.grad.item():.4f}")
print(f"∂loss/∂b1 (layer1.bias.grad): {model.layer1.bias.grad.item():.4f}")
print(f"∂loss/∂w2 (layer2.weight.grad): {model.layer2.weight.grad.item():.4f}")
print(f"∂loss/∂b2 (layer2.bias.grad): {model.layer2.bias.grad.item():.4f}")

print("\n" + "=" * 60)
print("手算结果对比：")
print("=" * 60)
print(f"∂loss/∂w1 (手算): 4.512")
print(f"∂loss/∂b1 (手算): 2.256")
print(f"∂loss/∂w2 (手算): 1.316")
print(f"∂loss/∂b2 (手算): 1.88")
```

**运行结果：**
```
============================================================
前向传播结果：
============================================================
Output: 0.9400
Loss: 0.8836

============================================================
反向传播梯度（自动计算）：
============================================================
∂loss/∂w1 (layer1.weight.grad): 4.5120
∂loss/∂b1 (layer1.bias.grad): 2.2560
∂loss/∂w2 (layer2.weight.grad): 1.3160
∂loss/∂b2 (layer2.bias.grad): 1.8800

============================================================
手算结果对比：
============================================================
∂loss/∂w1 (手算): 4.512
∂loss/∂b1 (手算): 2.256
∂loss/∂w2 (手算): 1.316
∂loss/∂b2 (手算): 1.88
```

✅ **完全匹配！** 证明我们的手算正确。

---

## 🎓 链式法则的通用模式

### 多层网络的梯度传播公式

对于一个 L 层神经网络：

```
Input → Layer1 → Layer2 → ... → LayerL → Output → Loss
```

**任意参数 θ 的梯度：**

$$
\frac{\partial \text{Loss}}{\partial \theta} = \frac{\partial \text{Loss}}{\partial \text{Output}} \times \frac{\partial \text{Output}}{\partial z_L} \times \frac{\partial z_L}{\partial a_{L-1}} \times \cdots \times \frac{\partial z_2}{\partial a_1} \times \frac{\partial a_1}{\partial z_1} \times \frac{\partial z_1}{\partial \theta}
$$

**关键点：**
1. 从 **loss** 开始往回计算
2. 每一步乘以 **当前层输出对输入的导数**
3. 一直乘到 **参数 θ**

---

## 🔥 常见激活函数的导数

### 1. ReLU

```python
# 前向
f(x) = max(0, x)

# 导数
f'(x) = { 1  if x > 0
        { 0  if x ≤ 0
```

**影响：** 负数区域梯度为 0，导致"死亡 ReLU"问题。

### 2. Sigmoid

```python
# 前向
f(x) = 1 / (1 + e^(-x))

# 导数
f'(x) = f(x) × (1 - f(x))
```

**影响：** 当 x 很大或很小时，f'(x) → 0，导致梯度消失。

### 3. Tanh

```python
# 前向
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

# 导数
f'(x) = 1 - f(x)²
```

**影响：** 比 sigmoid 好，但仍可能梯度消失。

---

## 💡 关键洞察

### 1. 梯度消失

```
如果链式乘积中有很多 < 1 的导数：
∂loss/∂w1 = 0.9 × 0.8 × 0.7 × 0.6 × ... → 接近 0
```

**解决方案：**
- 使用 ReLU（导数恒为 1 或 0，不是小数）
- 残差连接（ResNet）
- 批归一化（Batch Normalization）

### 2. 梯度爆炸

```
如果链式乘积中有很多 > 1 的导数：
∂loss/∂w1 = 1.5 × 1.8 × 2.1 × 1.7 × ... → 爆炸
```

**解决方案：**
- 梯度裁剪（Gradient Clipping）
- 权重初始化（Xavier/He initialization）
- 较小的学习率

---

## 🎯 总结

### 链式法则的本质

**三步走：**
1. **前向传播**：计算输出，记录中间值
2. **计算 loss 对输出的梯度**：∂loss/∂output
3. **反向传播**：从右到左，依次乘以局部导数

**关键公式：**
```
∂loss/∂w = (∂loss/∂output) × (∂output/∂hidden) × (∂hidden/∂w)
                ↑                  ↑                    ↑
            从 loss 开始        通过激活函数         到达参数
```

### 记忆技巧

想象梯度是一条**水流**：
- **前向传播**：水从山顶（输入）流到山谷（输出）
- **反向传播**：水从山谷（loss）往回流到山顶（参数）
- **链式法则**：每经过一个"水坝"（层），水流都会被放大或缩小

---

## 📚 扩展阅读

- **自动微分**：PyTorch 如何自动计算梯度（Autograd）
- **计算图**：TensorFlow 的静态图 vs PyTorch 的动态图
- **高阶导数**：二阶优化方法（牛顿法、L-BFGS）

---

**创建日期：** 2026-04-03
**适用于：** 深度学习初学者理解反向传播原理
