"""
model.parameters() vs model.named_parameters() 详解

展示两者的区别和使用场景
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """简单的演示模型"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x


def compare_parameters_methods():
    """对比两种方法"""
    print("=" * 70)
    print("model.parameters() vs model.named_parameters() 对比")
    print("=" * 70)

    model = SimpleModel()

    # 方法 1: model.parameters()
    print("\n【方法 1】model.parameters()")
    print("-" * 70)
    print("返回内容：只有参数张量（tensor）")
    print("类型：生成器（generator）")
    print("\n遍历结果：")

    for i, param in enumerate(model.parameters()):
        print(f"\n参数 {i}:")
        print(f"  shape: {param.shape}")
        print(f"  requires_grad: {param.requires_grad}")
        print(f"  数据类型: {param.dtype}")
        # print(f"  前几个值: {param.flatten()[:5]}")  # 显示前5个值

    # 方法 2: model.named_parameters()
    print("\n\n【方法 2】model.named_parameters()")
    print("-" * 70)
    print("返回内容：(参数名称, 参数张量) 的元组")
    print("类型：生成器（generator）")
    print("\n遍历结果：")

    for name, param in model.named_parameters():
        print(f"\n参数名: {name}")
        print(f"  shape: {param.shape}")
        print(f"  requires_grad: {param.requires_grad}")
        print(f"  数据类型: {param.dtype}")


def use_case_optimizer():
    """使用场景 1：创建优化器"""
    print("\n\n" + "=" * 70)
    print("使用场景 1：创建优化器")
    print("=" * 70)

    model = SimpleModel()

    print("\n✅ 使用 model.parameters()（推荐）")
    print("-" * 70)
    print("""
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 优化器只需要参数值，不需要名称
# 所以用 parameters() 就够了
    """)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"优化器创建成功: {type(optimizer)}")
    print(f"优化器管理的参数组数: {len(optimizer.param_groups)}")
    print(f"总参数数: {len(list(model.parameters()))}")


def use_case_selective_training():
    """使用场景 2：选择性训练（冻结某些层）"""
    print("\n\n" + "=" * 70)
    print("使用场景 2：选择性训练（冻结部分层）")
    print("=" * 70)

    model = SimpleModel()

    print("\n✅ 使用 model.named_parameters()（必须）")
    print("-" * 70)
    print("""
# 冻结 layer1，只训练 layer2
for name, param in model.named_parameters():
    if 'layer1' in name:
        param.requires_grad = False  # 冻结
    else:
        param.requires_grad = True   # 训练
    """)

    # 实际操作
    for name, param in model.named_parameters():
        if 'layer1' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    print("\n执行后的参数状态：")
    for name, param in model.named_parameters():
        status = "🔒 冻结" if not param.requires_grad else "🔥 训练"
        print(f"  {name}: {status}")


def use_case_debugging():
    """使用场景 3：调试和检查"""
    print("\n\n" + "=" * 70)
    print("使用场景 3：调试和检查模型参数")
    print("=" * 70)

    model = SimpleModel()

    print("\n✅ 使用 model.named_parameters()（必须）")
    print("-" * 70)

    # 统计参数量
    total_params = 0
    trainable_params = 0

    print("\n参数详情：")
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

        print(f"\n  {name}:")
        print(f"    shape: {param.shape}")
        print(f"    参数量: {param_count:,}")
        print(f"    可训练: {param.requires_grad}")

    print(f"\n总结：")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  冻结参数: {total_params - trainable_params:,}")


def use_case_different_lr():
    """使用场景 4：不同层使用不同学习率"""
    print("\n\n" + "=" * 70)
    print("使用场景 4：不同层使用不同学习率")
    print("=" * 70)

    model = SimpleModel()

    print("\n✅ 使用 model.named_parameters()（必须）")
    print("-" * 70)
    print("""
# layer1 使用小学习率，layer2 使用大学习率
layer1_params = []
layer2_params = []

for name, param in model.named_parameters():
    if 'layer1' in name:
        layer1_params.append(param)
    else:
        layer2_params.append(param)

optimizer = torch.optim.Adam([
    {'params': layer1_params, 'lr': 1e-5},  # 小学习率
    {'params': layer2_params, 'lr': 1e-3}   # 大学习率
])
    """)

    # 实际操作
    layer1_params = []
    layer2_params = []

    for name, param in model.named_parameters():
        if 'layer1' in name:
            layer1_params.append(param)
        else:
            layer2_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': layer1_params, 'lr': 1e-5},
        {'params': layer2_params, 'lr': 1e-3}
    ])

    print("\n优化器配置：")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  参数组 {i}: 学习率 = {group['lr']}, 参数数 = {len(group['params'])}")


def check_gradient_flow():
    """使用场景 5：检查梯度流"""
    print("\n\n" + "=" * 70)
    print("使用场景 5：检查梯度流（调试梯度消失/爆炸）")
    print("=" * 70)

    model = SimpleModel()

    # 模拟训练
    x = torch.randn(4, 4)
    target = torch.randn(4, 2)

    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    print("\n✅ 使用 model.named_parameters()（必须）")
    print("-" * 70)

    print("\n梯度统计：")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            print(f"\n  {name}:")
            print(f"    梯度均值: {grad_mean:.6f}")
            print(f"    梯度最大值: {grad_max:.6f}")

            # 检查异常
            if grad_mean < 1e-7:
                print(f"    ⚠️  梯度过小，可能梯度消失")
            elif grad_mean > 1e2:
                print(f"    ⚠️  梯度过大，可能梯度爆炸")
        else:
            print(f"\n  {name}: ❌ 无梯度（被冻结或未使用）")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" model.parameters() vs model.named_parameters() 完整对比 ".center(70, "="))
    print("=" * 70)

    # 1. 基本对比
    compare_parameters_methods()

    # 2. 使用场景
    use_case_optimizer()
    use_case_selective_training()
    use_case_debugging()
    use_case_different_lr()
    check_gradient_flow()

    # 总结
    print("\n\n" + "=" * 70)
    print("总结")
    print("=" * 70)

    print("""
┌─────────────────────────┬──────────────────┬──────────────────────┐
│         方法            │    返回内容      │      使用场景        │
├─────────────────────────┼──────────────────┼──────────────────────┤
│ model.parameters()      │ 只有参数张量     │ • 创建优化器         │
│                         │ (无名称)         │ • 计算总参数量       │
│                         │                  │ • 参数初始化         │
├─────────────────────────┼──────────────────┼──────────────────────┤
│ model.named_parameters()│ (名称, 参数)元组 │ • 选择性训练/冻结    │
│                         │                  │ • 不同层不同学习率   │
│                         │                  │ • 调试检查参数       │
│                         │                  │ • 检查梯度流         │
│                         │                  │ • 参数可视化         │
└─────────────────────────┴──────────────────┴──────────────────────┘

核心区别：
1. parameters() → 只有值（tensor）
   - 用于：传给优化器、统计参数量
   - 特点：简单、直接

2. named_parameters() → 名称 + 值
   - 用于：需要区分不同参数的场景
   - 特点：可以根据名称筛选、操作特定层

记忆口诀：
"看名字做事用 named_parameters，
 只管数值用 parameters。"

常见模式：
# ✅ 创建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ✅ 冻结某些层
for name, param in model.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False

# ✅ 不同层不同学习率
optimizer = torch.optim.Adam([
    {'params': [p for n, p in model.named_parameters() if 'encoder' in n], 'lr': 1e-5},
    {'params': [p for n, p in model.named_parameters() if 'decoder' in n], 'lr': 1e-3}
])

# ✅ 检查参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
    """)

    print("=" * 70)


if __name__ == "__main__":
    main()
