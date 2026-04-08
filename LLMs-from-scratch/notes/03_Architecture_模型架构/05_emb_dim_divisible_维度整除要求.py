"""
为什么 emb_dim 必须能被 n_heads 整除？

详解多头注意力的维度要求
"""

import torch
import torch.nn as nn


def explain_multi_head_splitting():
    """解释多头注意力如何分割维度"""
    print("=" * 70)
    print("多头注意力的维度分割机制")
    print("=" * 70)

    print("""
多头注意力的核心思想：
  将一个大的注意力分割成多个小的"头"（heads），
  每个头独立处理一部分维度，最后再合并。

具体过程：
  1. 输入：[batch, seq_len, emb_dim]
  2. 投影到 Q/K/V：[batch, seq_len, emb_dim]
  3. 分割成多个头：[batch, seq_len, n_heads, head_dim]
     ↑ 关键：head_dim = emb_dim / n_heads
  4. 每个头独立计算注意力
  5. 合并所有头：[batch, seq_len, emb_dim]
    """)

    # 示例：GPT-2 Small
    batch = 2
    seq_len = 4
    emb_dim = 768
    n_heads = 12

    print(f"\n示例：GPT-2 Small")
    print(f"  emb_dim = {emb_dim}")
    print(f"  n_heads = {n_heads}")
    print(f"  head_dim = {emb_dim} / {n_heads} = {emb_dim // n_heads}")

    # 模拟 Q 矩阵
    Q = torch.randn(batch, seq_len, emb_dim)
    print(f"\n步骤 1 - Q 的形状: {Q.shape}")

    # 分割成多个头
    head_dim = emb_dim // n_heads
    Q_multi_head = Q.view(batch, seq_len, n_heads, head_dim)
    print(f"步骤 2 - 分割成多头: {Q_multi_head.shape}")
    print(f"  [batch, seq_len, n_heads, head_dim]")
    print(f"  [{batch}, {seq_len}, {n_heads}, {head_dim}]")

    # 转置以便计算注意力
    Q_transposed = Q_multi_head.transpose(1, 2)
    print(f"\n步骤 3 - 转置便于计算: {Q_transposed.shape}")
    print(f"  [batch, n_heads, seq_len, head_dim]")


def demonstrate_division_requirement():
    """演示为什么必须整除"""
    print("\n\n" + "=" * 70)
    print("为什么必须整除？")
    print("=" * 70)

    print("\n✅ 能整除的情况：")
    print("-" * 70)

    emb_dim = 768
    n_heads = 12
    head_dim = emb_dim // n_heads

    print(f"emb_dim = {emb_dim}")
    print(f"n_heads = {n_heads}")
    print(f"head_dim = {emb_dim} // {n_heads} = {head_dim}")
    print(f"验证: {n_heads} × {head_dim} = {n_heads * head_dim} = {emb_dim} ✅")

    # 模拟 reshape
    x = torch.randn(2, 4, emb_dim)
    print(f"\n原始形状: {x.shape}")

    try:
        x_reshaped = x.view(2, 4, n_heads, head_dim)
        print(f"Reshape 成功: {x_reshaped.shape}")
        print("✅ 可以均匀分配到每个头")
    except Exception as e:
        print(f"❌ Reshape 失败: {e}")

    print("\n\n❌ 不能整除的情况（你的配置）：")
    print("-" * 70)

    emb_dim_wrong = 1280
    n_heads_wrong = 25

    print(f"emb_dim = {emb_dim_wrong}")
    print(f"n_heads = {n_heads_wrong}")
    print(f"head_dim = {emb_dim_wrong} / {n_heads_wrong} = {emb_dim_wrong / n_heads_wrong}")
    print(f"  ❌ 不是整数！")

    print(f"\n问题：")
    print(f"  如果 head_dim = 51.2")
    print(f"  → 无法创建 51.2 维的张量")
    print(f"  → PyTorch 的维度必须是整数")
    print(f"  → reshape 会失败")

    # 尝试 reshape
    x_wrong = torch.randn(2, 4, emb_dim_wrong)
    print(f"\n原始形状: {x_wrong.shape}")

    try:
        # 尝试使用浮点数维度
        head_dim_wrong = emb_dim_wrong / n_heads_wrong
        x_reshaped_wrong = x_wrong.view(2, 4, n_heads_wrong, int(head_dim_wrong))
        print(f"Reshape: {x_reshaped_wrong.shape}")

        # 检查是否有数据丢失
        original_elements = emb_dim_wrong
        reshaped_elements = n_heads_wrong * int(head_dim_wrong)
        print(f"\n原始维度: {original_elements}")
        print(f"Reshape 后: {n_heads_wrong} × {int(head_dim_wrong)} = {reshaped_elements}")
        if original_elements != reshaped_elements:
            print(f"❌ 数据丢失: {original_elements - reshaped_elements} 维")
    except Exception as e:
        print(f"❌ Reshape 失败: {e}")


def visualize_splitting():
    """可视化维度分割"""
    print("\n\n" + "=" * 70)
    print("可视化：维度如何分割")
    print("=" * 70)

    print("""
假设 emb_dim = 12, n_heads = 3

原始向量（12 维）：
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

分割成 3 个头，每头 4 维：

头 0:
┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │
└───┴───┴───┴───┘

头 1:
┌───┬───┬───┬───┐
│ 4 │ 5 │ 6 │ 7 │
└───┴───┴───┴───┘

头 2:
┌───┬───┬───┬───┐
│ 8 │ 9 │10 │11 │
└───┴───┴───┴───┘

✅ 12 维均匀分成 3 份，每份 4 维

如果不能整除（例如 emb_dim=13, n_heads=3）：
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

头 0: 4.33 维？ ❌ 维度必须是整数
头 1: 4.33 维？ ❌ 无法表示
头 2: 4.33 维？ ❌ 不可能
    """)


def show_code_implementation():
    """展示代码实现"""
    print("\n\n" + "=" * 70)
    print("代码实现：为什么会有断言")
    print("=" * 70)

    print("""
MultiHeadAttention 类的初始化：

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, ...):
        super().__init__()

        # ⚠️ 关键断言
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # ← 必须是整数

    def forward(self, x):
        # x: [batch, seq_len, d_in]

        # 投影到 Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_out]

        # 分割成多个头
        batch, seq_len, d_out = Q.shape
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim)
        #                          ^^^^^^^^^^^^^^^^^^^^^^^^
        #                          这里需要 head_dim 是整数！

        # ... 后续计算

为什么需要断言？
  1. head_dim = d_out // num_heads 必须是整数
  2. 否则无法 view() / reshape()
  3. 提前检查，给出友好的错误信息
    """)


def common_configurations():
    """常见的配置组合"""
    print("\n\n" + "=" * 70)
    print("常见的 emb_dim 和 n_heads 组合")
    print("=" * 70)

    configs = [
        ("GPT-2 Small", 124_000_000, 768, 12, 64),
        ("GPT-2 Medium", 355_000_000, 1024, 16, 64),
        ("GPT-2 Large", 774_000_000, 1280, 20, 64),
        ("GPT-2 XL", 1_500_000_000, 1600, 25, 64),
        ("GPT-3 Small", 125_000_000, 768, 12, 64),
        ("GPT-3 Medium", 350_000_000, 1024, 16, 64),
        ("GPT-3 Large", 760_000_000, 1280, 20, 64),
        ("GPT-3 XL", 1_300_000_000, 1600, 25, 64),
    ]

    print("\n┌─────────────────┬──────────────┬─────────┬─────────┬──────────┬────────┐")
    print("│     模型        │   参数量     │ emb_dim │ n_heads │ head_dim │ 整除？ │")
    print("├─────────────────┼──────────────┼─────────┼─────────┼──────────┼────────┤")

    for name, params, emb_dim, n_heads, head_dim in configs:
        params_str = f"{params/1e9:.1f}B" if params >= 1e9 else f"{params/1e6:.0f}M"
        divisible = "✅" if emb_dim % n_heads == 0 else "❌"
        print(f"│ {name:15s} │ {params_str:12s} │ {emb_dim:7d} │ {n_heads:7d} │ {head_dim:8d} │ {divisible:6s} │")

    print("└─────────────────┴──────────────┴─────────┴─────────┴──────────┴────────┘")

    print("\n观察：")
    print("  ✅ 所有 GPT 模型的 head_dim 都是 64")
    print("  ✅ emb_dim 总是能被 n_heads 整除")
    print("  ✅ 这是有意设计的，保证均匀分配")


def fix_your_config():
    """修复你的配置"""
    print("\n\n" + "=" * 70)
    print("修复你的配置")
    print("=" * 70)

    print("\n你的配置（错误）：")
    print("-" * 70)
    print("""
GPT_CONFIG_xl = {
    "emb_dim": 1280,
    "n_heads": 25,  # ❌ 1280 / 25 = 51.2
}
    """)

    print("\n方案 1：调整 n_heads（推荐）")
    print("-" * 70)
    print("""
GPT_CONFIG_xl = {
    "emb_dim": 1280,
    "n_heads": 20,  # ✅ 1280 / 20 = 64
}

理由：
  - 保持 emb_dim = 1280（GPT-2 Large 的标准）
  - head_dim = 64（GPT 系列的标准）
  - 只调整 n_heads
    """)

    print("\n方案 2：调整 emb_dim")
    print("-" * 70)
    print("""
GPT_CONFIG_xl = {
    "emb_dim": 1600,  # ✅ 1600 / 25 = 64
    "n_heads": 25,
}

理由：
  - 保持 n_heads = 25（你想要的头数）
  - 调整 emb_dim 到 1600（GPT-2 XL 的标准）
  - head_dim = 64
    """)

    print("\n推荐的标准配置：")
    print("-" * 70)

    # 测试配置
    test_configs = [
        (768, 12),
        (1024, 16),
        (1280, 20),
        (1600, 25),
        (2048, 32),
    ]

    print("\n可选的标准组合（head_dim = 64）：")
    for emb_dim, n_heads in test_configs:
        head_dim = emb_dim // n_heads
        print(f"  emb_dim={emb_dim:4d}, n_heads={n_heads:2d} → head_dim={head_dim}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" 为什么 emb_dim 必须能被 n_heads 整除？ ".center(70, "="))
    print("=" * 70)

    # 1. 解释机制
    explain_multi_head_splitting()

    # 2. 演示整除要求
    demonstrate_division_requirement()

    # 3. 可视化
    visualize_splitting()

    # 4. 代码实现
    show_code_implementation()

    # 5. 常见配置
    common_configurations()

    # 6. 修复方案
    fix_your_config()

    print("\n\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
核心原因：
  多头注意力需要将 emb_dim 均匀分配给每个头
  → head_dim = emb_dim / n_heads
  → head_dim 必须是整数（张量维度必须是整数）
  → 所以 emb_dim 必须能被 n_heads 整除

你的错误：
  emb_dim = 1280
  n_heads = 25
  head_dim = 1280 / 25 = 51.2 ❌ 不是整数

修复方案：
  方案 1（推荐）：
    emb_dim = 1280
    n_heads = 20  ← 改这里
    head_dim = 64 ✅

  方案 2：
    emb_dim = 1600  ← 改这里
    n_heads = 25
    head_dim = 64 ✅

记忆口诀：
  "多头分维度，整除是必须；
   head_dim 是整数，才能正常 view。"

检查公式：
  ✅ emb_dim % n_heads == 0
  ✅ head_dim = emb_dim // n_heads
  ✅ n_heads × head_dim == emb_dim
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
