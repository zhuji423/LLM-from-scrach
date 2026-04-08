# 优化器状态存储详解

> 深入理解为什么模型文件会从 653MB 膨胀到 1.95GB

---

## 问题：为什么保存优化器后文件从 653MB 变成 1.95GB？

### 快速答案

**Adam 优化器需要存储 2 份动量**，每份大小和模型参数相同。

```
总大小 = 模型参数 + 优化器状态
1.95GB ≈ 0.65GB  + 1.3GB
```

---

## 文件大小的数学拆解

### 以 163M 参数模型为例

```python
模型参数 θ:     163M × 4 bytes = 652 MB
一阶动量 m_t:   163M × 4 bytes = 652 MB  ← Adam 状态
二阶动量 v_t:   163M × 4 bytes = 652 MB  ← Adam 状态
元数据:                           ~1 MB
--------------------------------------------
总计:           163M × 12 bytes ≈ 1.96 GB (约3倍)
```

### 验证公式

$$\text{文件大小} = \text{参数数量} \times \text{字节数} \times \text{份数}$$

$$1.96 \text{ GB} = 163M \times 4 \text{ bytes} \times 3 \text{ 份} / 1024^3$$

---

## Adam 优化器的内部状态

### Adam 的更新公式

$$
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(一阶动量)} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(二阶动量)} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \quad \text{(偏差修正)} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align}
$$

### 优化器内部需要记住的状态

```python
# 对于每个参数，Adam 都维护：
state = {
    "step": t,                      # 步数计数器（用于偏差修正）
    "exp_avg": m_t,                # 一阶动量（梯度的指数移动平均）
    "exp_avg_sq": v_t,             # 二阶动量（梯度平方的指数移动平均）
}

# 这些张量的大小和参数本身相同！
```

### 为什么需要两份动量？

#### 1. **一阶动量 $m_t$（方向和速度）**

记录梯度的"方向和速度"，提供**惯性**：

```python
# 类比：小球滚下山坡
m_t = 0.9 * m_{t-1} + 0.1 * gradient  # 保留90%的历史方向

# 作用：
# - 平滑梯度波动
# - 加速下降（类似动量）
# - 穿越局部平坦区域
```

#### 2. **二阶动量 $v_t$（波动程度）**

记录梯度的"波动程度"，用于**自适应学习率**：

```python
# 类比：路况检测器
v_t = 0.999 * v_{t-1} + 0.001 * gradient²  # 记录梯度方差

# 作用：
# - 梯度稳定的参数 → 大步前进（学习率高）
# - 梯度波动的参数 → 小步谨慎（学习率低）
```

---

## 实际验证：查看 checkpoint 内部

### 代码示例

```python
import torch

# 加载 checkpoint
checkpoint = torch.load("model_and_optimizer.pth")

# 查看各部分大小
def get_size_mb(state_dict):
    """计算 state_dict 的内存大小（MB）"""
    total_bytes = 0
    for param in state_dict.values():
        if isinstance(param, torch.Tensor):
            total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024**2)

# 1. 模型参数大小
model_size = get_size_mb(checkpoint["model_state_dict"])
print(f"模型参数: {model_size:.1f} MB")

# 2. 优化器状态大小
opt_state = checkpoint["optimizer_state_dict"]["state"]
momentum_size = 0
for param_state in opt_state.values():
    if "exp_avg" in param_state:  # 一阶动量
        momentum_size += param_state["exp_avg"].numel() * \
                        param_state["exp_avg"].element_size()
    if "exp_avg_sq" in param_state:  # 二阶动量
        momentum_size += param_state["exp_avg_sq"].numel() * \
                        param_state["exp_avg_sq"].element_size()

print(f"优化器动量: {momentum_size / (1024**2):.1f} MB")
print(f"总大小: {model_size + momentum_size / (1024**2):.1f} MB")
```

**预期输出**：
```
模型参数: 652.9 MB
优化器动量: 1305.8 MB  (约2倍模型大小)
总大小: 1958.7 MB  (约3倍模型大小)
```

---

## Load 到内存的大小：取决于加载内容

### 场景 1：仅推理（只加载模型）

```python
# 只加载模型参数
model = GPTModel(config)
checkpoint = torch.load("model_and_optimizer.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 内存占用: 652.9 MB
```

### 场景 2：继续训练（加载模型 + 优化器）

```python
# 加载模型和优化器
model = GPTModel(config)
optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load("model_and_optimizer.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# 内存占用: 1.95 GB
```

### 场景 3：训练时的实际内存占用（更大！）

```python
# 训练时的内存占用
内存 = 模型参数 + 优化器状态 + 梯度 + 激活值 + 临时缓存
```

| 组件 | 大小 | 说明 |
|------|------|------|
| 模型参数 | 652.9 MB | 权重 $\theta$ |
| 优化器状态 | 1305.8 MB | $m_t$ + $v_t$ |
| **梯度** | 652.9 MB | $\nabla_\theta L$ |
| **激活值** | ~1-3 GB | 前向传播的中间结果（取决于batch size） |
| **总计** | ~4-6 GB | 训练时的峰值内存 |

**训练内存是推理的 6-10 倍！**

---

## 开车类比

想象你在山路上开车：

| 元素 | 对应优化器 | 存储内容 |
|------|-----------|---------|
| **当前位置** | 模型参数 $\theta$ | 652.9 MB |
| **车速和方向** | 一阶动量 $m_t$ | 652.9 MB |
| **加速度历史** | 二阶动量 $v_t$ | 652.9 MB |
| **行驶里程** | 步数 $t$ | 几 KB |

### 场景对比

**场景1：中断后继续（保存了优化器）**
```
训练曲线：
Loss
  |     中断↓ 恢复后平滑继续
  |   ╱╲    ╱╲
  |  ╱  ╲  ╱  ╲___
  | ╱    ╲╱       ╲___
  |╱                  ╲___
  +------------------------> Steps
     10k  10k+1
```
- 记住了"车速+方向+加速度"
- 重启后平滑接续，训练曲线稳定

**场景2：没保存优化器**
```
训练曲线：
Loss
  |     中断↓ 恢复后出现波动
  |   ╱╲      ╱╲
  |  ╱  ╲    ╱  ╲  ╱╲
  | ╱    ╲  ╱    ╲╱  ╲___
  |╱      ╲╱            ╲___
  +------------------------> Steps
     10k  10k+1  需要重新积累动量
```
- 重启后"车速归零"
- 需要几百到几千步重新找到最优路径
- 浪费计算资源和时间

---

## 不同优化器的存储复杂度

| 优化器 | 状态大小 | 内存倍数 | 备注 |
|--------|---------|---------|------|
| **SGD（无动量）** | 0 | 1x | 无额外状态 |
| **SGD + Momentum** | 1× 参数量 | 2x | 一阶动量 |
| **Adam / AdamW** | 2× 参数量 | 3x | 一阶+二阶动量 |
| **Adafactor** | 行列分解的 $v_t$ | ~1.5x | 节省内存 |
| **8-bit Adam** | 量化的 $m_t + v_t$ | ~1.5x | 量化优化 |

*以模型参数为基准*

### 示例（163M 参数模型，float32）

```python
# 只保存模型
torch.save(model.state_dict(), "model.pth")
# 文件大小: 652.9 MB (163M × 4 bytes)

# 保存模型 + Adam 优化器
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "checkpoint.pth")
# 文件大小: 1958.7 MB (163M × 4 bytes × 3)
```

---

## 优化存储空间的方法

### 方法 1：训练时分离保存

```python
# 定期保存完整 checkpoint（用于恢复训练）
if step % 5000 == 0:
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),  # 保存
        "loss": loss.item(),
    }, f"checkpoint_step_{step}.pth")  # 1.95 GB

# 只保存最佳模型（用于部署）
if val_loss < best_loss:
    torch.save(model.state_dict(), "best_model.pth")  # 652.9 MB
```

### 方法 2：使用更轻量的优化器

```python
# 从 Adam 切换到 SGD + Momentum
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9  # 只需要一阶动量
)
# 存储大小: 652.9 MB × 2 = 1.31 GB (节省 33%)
```

### 方法 3：使用 8-bit 优化器

```python
import bitsandbytes as bnb

# 8-bit Adam：动量用 8-bit 存储
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.001)

# 存储大小: 652.9 MB × 1.5 ≈ 0.98 GB (节省 50%)
```

### 方法 4：只保留最近的 N 个 checkpoint

```python
import os
from collections import deque

checkpoints = deque(maxlen=3)  # 只保留最近 3 个

def save_checkpoint(step):
    ckpt_path = f"checkpoint_{step}.pth"
    torch.save({...}, ckpt_path)

    # 删除旧的 checkpoint
    checkpoints.append(ckpt_path)
    if len(checkpoints) > 3:
        old_ckpt = checkpoints.popleft()
        os.remove(old_ckpt)
```

---

## 混合精度训练的影响

### FP16 vs FP32

```python
# 模型参数：FP16 → 652.9 MB / 2 = 326.5 MB
# 但优化器状态仍然是 FP32 → 1305.8 MB
# 总计: 326.5 + 1305.8 ≈ 1.63 GB

# 为什么优化器保持 FP32？
# → 数值稳定性！一阶和二阶动量需要高精度累积
```

**关键点**：优化器通常保持 FP32 精度以保证数值稳定性！

---

## 实践建议

### 1. 训练中必须保存优化器

```python
# ✅ 正确做法
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),  # 必须保存
    "scheduler_state_dict": scheduler.state_dict(),  # 如果使用了调度器
}, "checkpoint.pth")
```

### 2. 部署时只保存模型

```python
# ✅ 最终部署
torch.save(model.state_dict(), "model_final.pth")

# 推理时加载
model.load_state_dict(torch.load("model_final.pth"))
model.eval()
```

### 3. 学习率调度器也需要保存

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 保存
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),  # ← 调度器状态
}, "full_checkpoint.pth")

# 恢复
checkpoint = torch.load("full_checkpoint.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
```

---

## 分布式训练的优化器状态分片

在大模型训练（如 GPT-3）中，优化器状态会占用巨大内存。**ZeRO 优化**（Zero Redundancy Optimizer）将优化器状态分片到多个 GPU：

```
传统 DDP:  每个 GPU 存储完整的优化器状态
ZeRO-1:    优化器状态分片（省内存 N 倍）
ZeRO-2:    优化器状态 + 梯度分片
ZeRO-3:    优化器状态 + 梯度 + 模型参数分片
```

**示例**：
```python
# 8 GPU 训练
模型大小: 652.9 MB
优化器状态（单 GPU）: 1305.8 MB

# 传统 DDP
每个 GPU: 652.9 + 1305.8 = 1958.7 MB
总内存: 1958.7 × 8 ≈ 15.7 GB

# ZeRO-1（优化器状态分片）
每个 GPU: 652.9 + 1305.8/8 = 816.1 MB
总内存: 816.1 × 8 ≈ 6.5 GB  (节省 58%)
```

---

## 总结

### 关键要点

| 概念 | 要点 |
|------|------|
| **Adam 状态** | 需要 2× 模型大小（一阶+二阶动量） |
| **文件大小** | 模型 + 优化器 ≈ 3× 模型参数 |
| **内存占用** | 取决于加载内容（推理 vs 训练） |
| **训练内存** | 4-6× 模型大小（参数+优化器+梯度+激活） |
| **保存策略** | 训练保存完整，部署只保存模型 |

### 记忆口诀

> "Adam 是个记账狂，一阶二阶都要藏"
> "存下来是 3 倍大，加载啥就占用啥"

### 判断树

```
需要从中断点继续训练？
├─ 是 → ✅ 必须保存优化器状态
└─ 否 → 是否需要保持训练动态？
    ├─ 是（如定期 checkpoint）→ ✅ 保存
    └─ 否（仅推理/微调）→ ❌ 不需要保存
```

---

> 最后更新: 2026-04-08
