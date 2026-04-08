"""
反向传播链式法则 - 可视化演示脚本

这个脚本演示：
1. 手动计算梯度（使用链式法则）
2. PyTorch 自动计算梯度
3. 对比两者结果，验证链式法则的正确性
"""

import torch
import torch.nn as nn


def manual_gradient_calculation():
    """手动计算梯度（使用链式法则）"""
    print("=" * 70)
    print("手动计算梯度（链式法则）")
    print("=" * 70)

    # 前向传播的数值
    x = 2.0
    w1 = 0.5
    b1 = -0.3
    w2 = 1.2
    b2 = 0.1
    target = 0.0

    # Layer 1
    h1 = w1 * x + b1
    a1 = max(0, h1)  # ReLU

    # Layer 2
    h2 = w2 * a1 + b2
    output = h2

    # Loss
    loss = (output - target) ** 2

    print(f"\n前向传播:")
    print(f"  x = {x}")
    print(f"  h1 = w1×x + b1 = {w1}×{x} + {b1} = {h1}")
    print(f"  a1 = ReLU(h1) = max(0, {h1}) = {a1}")
    print(f"  h2 = w2×a1 + b2 = {w2}×{a1} + {b2} = {h2}")
    print(f"  output = {output}")
    print(f"  loss = (output - target)² = ({output} - {target})² = {loss}")

    # 反向传播
    print(f"\n反向传播（链式法则）:")

    # Step 1: ∂loss/∂output
    dloss_doutput = 2 * (output - target)
    print(f"\n  Step 1: ∂loss/∂output")
    print(f"    = 2×(output - target)")
    print(f"    = 2×({output} - {target})")
    print(f"    = {dloss_doutput}")

    # Step 2: ∂loss/∂w2
    doutput_dh2 = 1  # output = h2
    dh2_dw2 = a1     # h2 = w2×a1 + b2
    dloss_dw2 = dloss_doutput * doutput_dh2 * dh2_dw2
    print(f"\n  Step 2: ∂loss/∂w2")
    print(f"    = ∂loss/∂output × ∂output/∂h2 × ∂h2/∂w2")
    print(f"    = {dloss_doutput} × {doutput_dh2} × {dh2_dw2}")
    print(f"    = {dloss_dw2}")

    # Step 3: ∂loss/∂b2
    dh2_db2 = 1      # h2 = w2×a1 + b2
    dloss_db2 = dloss_doutput * doutput_dh2 * dh2_db2
    print(f"\n  Step 3: ∂loss/∂b2")
    print(f"    = ∂loss/∂output × ∂output/∂h2 × ∂h2/∂b2")
    print(f"    = {dloss_doutput} × {doutput_dh2} × {dh2_db2}")
    print(f"    = {dloss_db2}")

    # Step 4: ∂loss/∂a1
    dh2_da1 = w2     # h2 = w2×a1 + b2
    dloss_da1 = dloss_doutput * doutput_dh2 * dh2_da1
    print(f"\n  Step 4: ∂loss/∂a1")
    print(f"    = ∂loss/∂output × ∂output/∂h2 × ∂h2/∂a1")
    print(f"    = {dloss_doutput} × {doutput_dh2} × {dh2_da1}")
    print(f"    = {dloss_da1}")

    # Step 5: ∂loss/∂h1 (通过 ReLU)
    da1_dh1 = 1 if h1 > 0 else 0  # ReLU 的导数
    dloss_dh1 = dloss_da1 * da1_dh1
    print(f"\n  Step 5: ∂loss/∂h1 (通过 ReLU)")
    print(f"    = ∂loss/∂a1 × ∂a1/∂h1")
    print(f"    = {dloss_da1} × {da1_dh1}  (ReLU导数: h1={h1}>0 → 1)")
    print(f"    = {dloss_dh1}")

    # Step 6: ∂loss/∂w1
    dh1_dw1 = x      # h1 = w1×x + b1
    dloss_dw1 = dloss_dh1 * dh1_dw1
    print(f"\n  Step 6: ∂loss/∂w1")
    print(f"    = ∂loss/∂h1 × ∂h1/∂w1")
    print(f"    = {dloss_dh1} × {dh1_dw1}")
    print(f"    = {dloss_dw1}")

    # Step 7: ∂loss/∂b1
    dh1_db1 = 1      # h1 = w1×x + b1
    dloss_db1 = dloss_dh1 * dh1_db1
    print(f"\n  Step 7: ∂loss/∂b1")
    print(f"    = ∂loss/∂h1 × ∂h1/∂b1")
    print(f"    = {dloss_dh1} × {dh1_db1}")
    print(f"    = {dloss_db1}")

    return {
        'loss': loss,
        'output': output,
        'dw1': dloss_dw1,
        'db1': dloss_db1,
        'dw2': dloss_dw2,
        'db2': dloss_db2
    }


def pytorch_automatic_gradient():
    """使用 PyTorch 自动计算梯度"""
    print("\n" + "=" * 70)
    print("PyTorch 自动计算梯度")
    print("=" * 70)

    # 创建两层网络
    class TwoLayerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(1, 1, bias=True)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(1, 1, bias=True)

            # 手动设置权重
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

    # 创建模型
    model = TwoLayerNet()
    x = torch.tensor([[2.0]])
    target = torch.tensor([[0.0]])

    # 前向传播
    output = model(x)
    loss = nn.MSELoss()(output, target)

    print(f"\n前向传播:")
    print(f"  Output: {output.item():.4f}")
    print(f"  Loss: {loss.item():.4f}")

    # 反向传播
    loss.backward()

    print(f"\n反向传播（自动计算）:")
    print(f"  ∂loss/∂w1 = {model.layer1.weight.grad.item():.4f}")
    print(f"  ∂loss/∂b1 = {model.layer1.bias.grad.item():.4f}")
    print(f"  ∂loss/∂w2 = {model.layer2.weight.grad.item():.4f}")
    print(f"  ∂loss/∂b2 = {model.layer2.bias.grad.item():.4f}")

    return {
        'loss': loss.item(),
        'output': output.item(),
        'dw1': model.layer1.weight.grad.item(),
        'db1': model.layer1.bias.grad.item(),
        'dw2': model.layer2.weight.grad.item(),
        'db2': model.layer2.bias.grad.item()
    }


def compare_results(manual, pytorch):
    """比较手算和自动计算的结果"""
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)

    print(f"\n前向传播:")
    print(f"  {'指标':<15} {'手算':<15} {'PyTorch':<15} {'误差':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'Output':<15} {manual['output']:<15.6f} {pytorch['output']:<15.6f} {abs(manual['output']-pytorch['output']):<15.8f}")
    print(f"  {'Loss':<15} {manual['loss']:<15.6f} {pytorch['loss']:<15.6f} {abs(manual['loss']-pytorch['loss']):<15.8f}")

    print(f"\n反向传播:")
    print(f"  {'梯度':<15} {'手算':<15} {'PyTorch':<15} {'误差':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'∂loss/∂w1':<15} {manual['dw1']:<15.6f} {pytorch['dw1']:<15.6f} {abs(manual['dw1']-pytorch['dw1']):<15.8f}")
    print(f"  {'∂loss/∂b1':<15} {manual['db1']:<15.6f} {pytorch['db1']:<15.6f} {abs(manual['db1']-pytorch['db1']):<15.8f}")
    print(f"  {'∂loss/∂w2':<15} {manual['dw2']:<15.6f} {pytorch['dw2']:<15.6f} {abs(manual['dw2']-pytorch['dw2']):<15.8f}")
    print(f"  {'∂loss/∂b2':<15} {manual['db2']:<15.6f} {pytorch['db2']:<15.6f} {abs(manual['db2']-pytorch['db2']):<15.8f}")

    # 检查是否匹配
    tolerance = 1e-6
    all_match = all([
        abs(manual['dw1'] - pytorch['dw1']) < tolerance,
        abs(manual['db1'] - pytorch['db1']) < tolerance,
        abs(manual['dw2'] - pytorch['dw2']) < tolerance,
        abs(manual['db2'] - pytorch['db2']) < tolerance,
    ])

    print("\n" + "=" * 70)
    if all_match:
        print("✅ 验证成功！手算结果与 PyTorch 自动计算完全一致！")
    else:
        print("❌ 验证失败！存在不匹配的结果。")
    print("=" * 70)


def visualize_gradient_flow():
    """可视化梯度流动"""
    print("\n" + "=" * 70)
    print("梯度流动可视化")
    print("=" * 70)

    print("""
前向传播（从左到右）：

    x=2.0 ──┐
            ├──[×0.5]──> h1=0.7 ──[ReLU]──> a1=0.7 ──┐
    b1=-0.3─┘                                        ├──[×1.2]──> h2=0.94
                                           b2=0.1 ───┘
                                                          │
                                                          ▼
                                                      output=0.94
                                                          │
                                                          ▼
                                                      loss=0.8836

反向传播（从右到左）：

                            ┌─────────────────────────────────┐
                            │   ∂loss/∂output = 1.88          │
                            └─────────────────────────────────┘
                                           │
                       ┌───────────────────┴───────────────────┐
                       ▼                                       ▼
            ┌──────────────────┐                   ┌──────────────────┐
            │ ∂loss/∂w2 = 1.316│                   │ ∂loss/∂b2 = 1.88 │
            └──────────────────┘                   └──────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────────┐
                            │   ∂loss/∂a1 = 2.256            │
                            └─────────────────────────────────┘
                                           │
                                           ▼ (通过 ReLU)
                            ┌─────────────────────────────────┐
                            │   ∂loss/∂h1 = 2.256            │
                            └─────────────────────────────────┘
                                   │               │
                       ┌───────────┴───┐       ┌───┴──────────────┐
                       ▼               ▼       ▼                  ▼
            ┌──────────────────┐   ┌──────────────────┐
            │ ∂loss/∂w1 = 4.512│   │ ∂loss/∂b1 = 2.256│
            └──────────────────┘   └──────────────────┘
    """)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" 反向传播链式法则 - 完整演示 ".center(70, "="))
    print("=" * 70)

    # 1. 手动计算
    manual_results = manual_gradient_calculation()

    # 2. PyTorch 自动计算
    pytorch_results = pytorch_automatic_gradient()

    # 3. 对比结果
    compare_results(manual_results, pytorch_results)

    # 4. 可视化
    visualize_gradient_flow()

    print("\n" + "=" * 70)
    print("关键要点:")
    print("=" * 70)
    print("""
1. 链式法则本质：将复杂的求导问题分解为多个简单的局部求导

   ∂loss/∂w = (∂loss/∂output) × (∂output/∂hidden) × (∂hidden/∂w)

2. 梯度从 loss 向后传播，每经过一层就乘以该层的局部导数

3. PyTorch 的 autograd 自动跟踪计算图并应用链式法则

4. 激活函数（如 ReLU）的导数会影响梯度传播：
   - ReLU: 导数为 0 或 1
   - Sigmoid/Tanh: 可能导致梯度消失

5. 梯度消失/爆炸：多层网络中，梯度可能因连续相乘而变得极小或极大
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
