"""
为什么计算参数量时会数组超限？

根因：模型参数不都是 2 维的，有些是 1 维的
"""

import torch
import torch.nn as nn


class SimpleGPT(nn.Module):
    """简化的 GPT 模型用于演示"""
    def __init__(self, vocab_size=100, emb_dim=8, context_length=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)


def demonstrate_problem():
    """演示问题：为什么会超限"""
    print("=" * 70)
    print("问题演示：为什么 param.shape[0] * param.shape[1] 会超限？")
    print("=" * 70)

    model = SimpleGPT()

    print("\n逐个检查模型参数的维度：")
    print("-" * 70)

    for i, param in enumerate(model.parameters()):
        print(f"\n参数 {i}:")
        print(f"  shape: {param.shape}")
        print(f"  维度数: {len(param.shape)}")

        # 尝试你的计算方式
        try:
            # ❌ 你的计算方式
            wrong_calculation = param.shape[0] * param.shape[1]
            print(f"  ✅ param.shape[0] * param.shape[1] = {wrong_calculation}")
        except IndexError as e:
            print(f"  ❌ 错误: {e}")
            print(f"     → 因为这个参数只有 {len(param.shape)} 维！")
            print(f"     → param.shape[1] 不存在")

        # ✅ 正确的计算方式
        correct_calculation = param.numel()
        print(f"  ✅ param.numel() = {correct_calculation}")


def explain_parameter_dimensions():
    """解释：为什么参数维度不同"""
    print("\n\n" + "=" * 70)
    print("根因分析：模型中的参数维度")
    print("=" * 70)

    model = SimpleGPT(vocab_size=100, emb_dim=8, context_length=4)

    print("\n完整的参数列表：")
    print("-" * 70)

    for name, param in model.named_parameters():
        dims = len(param.shape)
        print(f"\n{name}:")
        print(f"  shape: {param.shape}")
        print(f"  维度: {dims}D")

        if dims == 1:
            print(f"  ⚠️  这是 1 维参数，没有 shape[1]")
        elif dims == 2:
            print(f"  ✅ 2 维参数，有 shape[0]={param.shape[0]}, shape[1]={param.shape[1]}")

    print("\n\n" + "=" * 70)
    print("参数维度总结")
    print("=" * 70)

    print("""
GPT 模型中的参数维度：

2 维参数（矩阵）：
  ✅ tok_emb.weight:     [vocab_size, emb_dim]
  ✅ pos_emb.weight:     [context_length, emb_dim]
  ✅ linear.weight:      [out_features, in_features]
  ✅ out_head.weight:    [vocab_size, emb_dim]

1 维参数（向量）：
  ⚠️  ln.weight (scale):  [emb_dim]
  ⚠️  ln.bias (shift):    [emb_dim]
  ⚠️  linear.bias:        [out_features]

问题：
  你的代码假设所有参数都是 2 维的
  → param.shape[0] * param.shape[1]

  但实际上 LayerNorm 和 Linear 的 bias 只有 1 维
  → param.shape[1] 不存在
  → IndexError: tuple index out of range
    """)


def show_correct_methods():
    """展示正确的计算方法"""
    print("\n\n" + "=" * 70)
    print("正确的参数量计算方法")
    print("=" * 70)

    model = SimpleGPT(vocab_size=100, emb_dim=8, context_length=4)

    print("\n方法 1：使用 param.numel() (推荐)")
    print("-" * 70)
    print("""
total_params = sum(p.numel() for p in model.parameters())

# numel() = number of elements (元素数量)
# 自动处理任意维度：1D, 2D, 3D, 4D, ...
    """)

    total_params_1 = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params_1:,}")

    print("\n方法 2：逐个累加（详细版）")
    print("-" * 70)
    print("""
total_params = 0
for param in model.parameters():
    total_params += param.numel()  # ✅ 正确
    """)

    total_params_2 = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params_2 += param_count
        print(f"  {name:30s} {str(param.shape):20s} → {param_count:6,} 参数")

    print(f"\n总参数量: {total_params_2:,}")

    print("\n❌ 错误方法（你的代码）")
    print("-" * 70)
    print("""
total_params = 0
for param in model.parameters():
    total_params += param.shape[0] * param.shape[1]  # ❌ 错误
    # 问题：假设所有参数都是 2 维
    """)

    print("\n尝试运行：")
    total_params_3 = 0
    for i, param in enumerate(model.parameters()):
        try:
            count = param.shape[0] * param.shape[1]
            total_params_3 += count
            print(f"  参数 {i}: {param.shape} → {count:,}")
        except IndexError:
            print(f"  参数 {i}: {param.shape} → ❌ 超限！(1维参数没有 shape[1])")


def show_numel_internals():
    """展示 numel() 的内部逻辑"""
    print("\n\n" + "=" * 70)
    print("numel() 是如何工作的？")
    print("=" * 70)

    examples = [
        torch.randn(100, 8),      # 2D
        torch.randn(8),           # 1D
        torch.randn(4, 8, 8),     # 3D
        torch.randn(2, 4, 8, 8),  # 4D
    ]

    print("\n不同维度的张量：")
    print("-" * 70)

    for i, tensor in enumerate(examples):
        print(f"\n张量 {i}:")
        print(f"  shape: {tensor.shape}")
        print(f"  维度: {len(tensor.shape)}D")

        # 手动计算
        manual_count = 1
        for dim_size in tensor.shape:
            manual_count *= dim_size

        # numel() 计算
        numel_count = tensor.numel()

        print(f"  手动计算: {' × '.join(map(str, tensor.shape))} = {manual_count:,}")
        print(f"  numel(): {numel_count:,}")
        print(f"  ✅ 相同: {manual_count == numel_count}")

    print("\n\nnumel() 的逻辑：")
    print("-" * 70)
    print("""
def numel(tensor):
    count = 1
    for dim_size in tensor.shape:
        count *= dim_size
    return count

示例：
  [100, 8]       → 100 × 8     = 800
  [8]            → 8            = 8
  [4, 8, 8]      → 4 × 8 × 8   = 256
  [2, 4, 8, 8]   → 2×4×8×8     = 512

优点：
  ✅ 适用于任意维度 (1D, 2D, 3D, ...)
  ✅ 不需要假设维度数量
  ✅ 代码简洁
    """)


def compare_methods():
    """对比两种方法"""
    print("\n\n" + "=" * 70)
    print("对比：你的方法 vs 正确方法")
    print("=" * 70)

    print("""
┌──────────────────────────┬──────────────────────────┬──────────────┐
│         方法             │          代码            │    结果      │
├──────────────────────────┼──────────────────────────┼──────────────┤
│ ❌ 你的方法              │ param.shape[0] * [1]     │ 超限错误     │
│ (假设所有参数都是 2D)    │                          │              │
├──────────────────────────┼──────────────────────────┼──────────────┤
│ ✅ 正确方法              │ param.numel()            │ 正确计数     │
│ (适用于任意维度)         │                          │              │
└──────────────────────────┴──────────────────────────┴──────────────┘

为什么你的方法会出错？

原因：
  1. 你假设所有参数都是 2D（有 shape[0] 和 shape[1]）
  2. 实际上很多参数是 1D（只有 shape[0]）
  3. 访问 param.shape[1] 时，1D 参数会报错：
     IndexError: tuple index out of range

哪些参数是 1D 的？
  - LayerNorm 的 weight (scale)
  - LayerNorm 的 bias (shift)
  - Linear 的 bias
  - 任何使用 nn.Parameter(torch.randn(n)) 创建的参数

记忆口诀：
  "numel 管全部，不管几维度；
   shape[0]*[1] 只能算矩阵，遇到向量就出错。"
    """)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" 为什么计算参数量时会数组超限？ ".center(70, "="))
    print("=" * 70)

    # 1. 演示问题
    demonstrate_problem()

    # 2. 解释根因
    explain_parameter_dimensions()

    # 3. 正确方法
    show_correct_methods()

    # 4. numel() 原理
    show_numel_internals()

    # 5. 对比
    compare_methods()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
问题根因：
  ❌ 你的代码假设所有参数都是 2 维矩阵
     → param.shape[0] * param.shape[1]

  ✅ 实际上模型中有 1 维向量参数
     → LayerNorm 的 weight/bias
     → Linear 的 bias

  ✅ 访问不存在的 shape[1] 导致超限错误
     → IndexError: tuple index out of range

解决方案：
  ✅ 使用 param.numel()
     - 自动计算任意维度的元素总数
     - 1D, 2D, 3D, ... 都适用

  ✅ 正确代码：
     total_params = sum(p.numel() for p in model.parameters())

记住：
  - numel() = number of elements
  - 适用于任意维度
  - 这是 PyTorch 中计算参数量的标准方法
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
