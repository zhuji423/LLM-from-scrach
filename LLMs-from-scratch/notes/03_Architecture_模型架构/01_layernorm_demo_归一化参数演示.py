"""
Layer Normalization 参数演示

展示 scale 和 shift 参数如何参与训练
"""

import torch
import torch.nn as nn


def manual_layer_norm_demo():
    """手动实现 Layer Normalization，展示每个步骤"""
    print("=" * 70)
    print("Layer Normalization 完整流程")
    print("=" * 70)

    # 模拟输入
    batch_size = 2
    seq_len = 3
    emb_dim = 4  # 简化维度便于演示

    x = torch.randn(batch_size, seq_len, emb_dim)
    print(f"\n输入 x:")
    print(f"  shape: {x.shape}")
    print(f"  值:\n{x}")

    # 步骤 1: 计算均值
    mean = x.mean(dim=-1, keepdim=True)
    print(f"\n步骤 1 - 计算均值（沿着最后一维）:")
    print(f"  shape: {mean.shape}")
    print(f"  值:\n{mean}")

    # 步骤 2: 计算方差
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    print(f"\n步骤 2 - 计算方差:")
    print(f"  shape: {var.shape}")
    print(f"  值:\n{var}")

    # 步骤 3: 标准化
    epsilon = 1e-5
    x_normalized = (x - mean) / torch.sqrt(var + epsilon)
    print(f"\n步骤 3 - 标准化 (x - mean) / sqrt(var):")
    print(f"  shape: {x_normalized.shape}")
    print(f"  值:\n{x_normalized}")
    print(f"\n  验证标准化后的均值和方差:")
    print(f"    均值: {x_normalized.mean(dim=-1)}")
    print(f"    方差: {x_normalized.var(dim=-1, unbiased=False)}")

    # 步骤 4: 应用 scale 和 shift
    scale = nn.Parameter(torch.ones(emb_dim))
    shift = nn.Parameter(torch.zeros(emb_dim))

    print(f"\n步骤 4 - 应用 scale 和 shift (可学习参数):")
    print(f"  scale (γ): {scale}")
    print(f"  shift (β): {shift}")

    output = scale * x_normalized + shift
    print(f"\n  最终输出:")
    print(f"    shape: {output.shape}")
    print(f"    值:\n{output}")

    return x, output, scale, shift


def demonstrate_learnable_params():
    """演示 scale 和 shift 参数如何通过梯度更新"""
    print("\n" + "=" * 70)
    print("演示 scale 和 shift 参数的梯度更新")
    print("=" * 70)

    # 创建 LayerNorm 层
    emb_dim = 768
    layer_norm = nn.LayerNorm(emb_dim)

    print(f"\nLayerNorm 的参数:")
    for name, param in layer_norm.named_parameters():
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

    # 模拟输入
    x = torch.randn(2, 4, emb_dim)

    # 前向传播
    output = layer_norm(x)

    # 计算损失（随便一个目标）
    target = torch.zeros_like(output)
    loss = nn.MSELoss()(output, target)

    print(f"\n前向传播:")
    print(f"  输入 shape: {x.shape}")
    print(f"  输出 shape: {output.shape}")
    print(f"  损失: {loss.item():.6f}")

    # 反向传播
    loss.backward()

    print(f"\n反向传播后的梯度:")
    for name, param in layer_norm.named_parameters():
        if param.grad is not None:
            print(f"  {name}:")
            print(f"    梯度均值: {param.grad.mean().item():.6f}")
            print(f"    梯度标准差: {param.grad.std().item():.6f}")
            print(f"    梯度绝对值最大: {param.grad.abs().max().item():.6f}")

    print("\n✅ 可以看到 weight (scale) 和 bias (shift) 都有梯度！")
    print("   这证明它们是可学习的参数。")


def compare_with_without_affine():
    """对比有无 scale/shift 参数的 LayerNorm"""
    print("\n" + "=" * 70)
    print("对比：有 vs 无 affine 参数")
    print("=" * 70)

    emb_dim = 4
    x = torch.randn(2, 3, emb_dim)

    # 有 affine 参数（默认）
    ln_with_affine = nn.LayerNorm(emb_dim, elementwise_affine=True)
    output_with = ln_with_affine(x)

    # 无 affine 参数
    ln_without_affine = nn.LayerNorm(emb_dim, elementwise_affine=False)
    output_without = ln_without_affine(x)

    print(f"\n输入:")
    print(f"  shape: {x.shape}")
    print(f"  均值: {x.mean(dim=-1)}")
    print(f"  方差: {x.var(dim=-1, unbiased=False)}")

    print(f"\n有 affine 参数 (elementwise_affine=True):")
    print(f"  参数数量: {sum(p.numel() for p in ln_with_affine.parameters())}")
    print(f"  输出均值: {output_with.mean(dim=-1)}")
    print(f"  输出方差: {output_with.var(dim=-1, unbiased=False)}")

    print(f"\n无 affine 参数 (elementwise_affine=False):")
    print(f"  参数数量: {sum(p.numel() for p in ln_without_affine.parameters())}")
    print(f"  输出均值: {output_without.mean(dim=-1)}")
    print(f"  输出方差: {output_without.var(dim=-1, unbiased=False)}")

    print("\n说明:")
    print("  - 有 affine: 输出分布可以被学习调整")
    print("  - 无 affine: 输出强制为均值=0, 方差=1")


def show_parameter_initialization():
    """展示参数的初始化值"""
    print("\n" + "=" * 70)
    print("LayerNorm 参数的初始化")
    print("=" * 70)

    emb_dim = 8
    layer_norm = nn.LayerNorm(emb_dim)

    print(f"\n初始化的参数值:")
    print(f"  weight (scale, γ):")
    print(f"    {layer_norm.weight}")
    print(f"    → 初始化为全 1")

    print(f"\n  bias (shift, β):")
    print(f"    {layer_norm.bias}")
    print(f"    → 初始化为全 0")

    print(f"\n为什么这样初始化?")
    print(f"  - scale=1, shift=0 → 初始时等价于只做标准化")
    print(f"  - 让模型自己学习是否需要调整分布")
    print(f"  - 不强加先验假设")


def practical_example():
    """实际训练中的例子"""
    print("\n" + "=" * 70)
    print("实际训练示例：参数如何更新")
    print("=" * 70)

    emb_dim = 4
    layer_norm = nn.LayerNorm(emb_dim)
    optimizer = torch.optim.SGD(layer_norm.parameters(), lr=0.1)

    x = torch.randn(2, 3, emb_dim)
    target = torch.randn(2, 3, emb_dim)

    print("初始参数:")
    print(f"  scale: {layer_norm.weight.data}")
    print(f"  shift: {layer_norm.bias.data}")

    # 训练 5 步
    for step in range(5):
        optimizer.zero_grad()
        output = layer_norm(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        if step == 0 or step == 4:
            print(f"\n第 {step+1} 步后:")
            print(f"  loss: {loss.item():.6f}")
            print(f"  scale: {layer_norm.weight.data}")
            print(f"  shift: {layer_norm.bias.data}")

    print("\n✅ 可以看到 scale 和 shift 参数在训练中不断更新！")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" Layer Normalization 参数详解 ".center(70, "="))
    print("=" * 70)

    # 1. 手动实现演示
    manual_layer_norm_demo()

    # 2. 演示可学习性
    demonstrate_learnable_params()

    # 3. 对比有无 affine
    compare_with_without_affine()

    # 4. 展示初始化
    show_parameter_initialization()

    # 5. 实际训练示例
    practical_example()

    print("\n" + "=" * 70)
    print("总结:")
    print("=" * 70)
    print("""
1. scale (γ) 和 shift (β) 是可学习参数
   - scale: 控制输出的缩放（方差）
   - shift: 控制输出的平移（均值）

2. 初始化值:
   - scale = 1 (不改变方差)
   - shift = 0 (不改变均值)

3. 为什么需要它们?
   - 标准化后分布被固定为 N(0,1)
   - scale/shift 让模型学习最优分布
   - 提供表达能力的灵活性

4. 参数量:
   - 每个 LayerNorm: 2 × emb_dim 参数
   - GPT-2 (emb_dim=768): 每个 LayerNorm 有 1536 个参数

5. 训练过程:
   - 通过反向传播计算梯度
   - 通过优化器更新参数
   - 自动学习最佳的 scale 和 shift
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
