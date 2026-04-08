"""
Feed-Forward Network (FFN) 正确实现

演示正确的 FFN 层和常见错误
"""

import torch
import torch.nn as nn


class FeedForwardCorrect(nn.Module):
    """✅ 正确的 FFN 实现"""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # ✅ 使用 emb_dim
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        # x: [batch_size, seq_len, emb_dim]
        return self.layers(x)


class FeedForwardWrong(nn.Module):
    """❌ 错误的 FFN 实现（常见错误）"""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["context_length"], 4 * cfg["context_length"]),  # ❌ 错误！
            nn.GELU(),
            nn.Linear(4 * cfg["context_length"], cfg["context_length"])
        )

    def forward(self, x):
        return self.layers(x)


def demonstrate_error():
    """演示错误"""
    print("=" * 70)
    print("演示维度不匹配错误")
    print("=" * 70)

    # 配置
    cfg = {
        "vocab_size": 50257,
        "context_length": 1024,  # ← 序列最大长度
        "emb_dim": 768,          # ← embedding 维度
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    # 输入
    batch_size = 2
    seq_len = 3
    x = torch.rand(batch_size, seq_len, cfg["emb_dim"])

    print(f"\n输入:")
    print(f"  shape: {x.shape}")
    print(f"  含义: [batch_size={batch_size}, seq_len={seq_len}, emb_dim={cfg['emb_dim']}]")

    # 错误的 FFN
    print(f"\n❌ 错误的 FFN (使用 context_length):")
    ffn_wrong = FeedForwardWrong(cfg)
    print(f"  第 1 层权重: {ffn_wrong.layers[0].weight.shape}")
    print(f"  期望: [4*emb_dim, emb_dim] = [3072, 768]")
    print(f"  实际: [4*context_length, context_length] = [4096, 1024]")

    try:
        out = ffn_wrong(x)
        print(f"  输出: {out.shape}")
    except RuntimeError as e:
        print(f"  错误: {e}")

    # 正确的 FFN
    print(f"\n✅ 正确的 FFN (使用 emb_dim):")
    ffn_correct = FeedForwardCorrect(cfg)
    print(f"  第 1 层权重: {ffn_correct.layers[0].weight.shape}")
    print(f"  期望: [4*emb_dim, emb_dim] = [3072, 768]")
    print(f"  实际: [4*emb_dim, emb_dim] = [3072, 768]")

    out = ffn_correct(x)
    print(f"  输出: {out.shape}")
    print(f"  ✅ 成功！")


def explain_dimensions():
    """解释维度概念"""
    print("\n" + "=" * 70)
    print("理解关键维度")
    print("=" * 70)

    print("""
GPT-2 的关键维度:

1. vocab_size (50257)
   - 词表大小
   - 用于: Token Embedding 和 Output Head
   - 示例: Embedding(50257, 768)

2. context_length (1024)
   - 最大序列长度
   - 用于: Position Embedding
   - 示例: Embedding(1024, 768)

3. emb_dim (768)
   - Embedding 维度 / Hidden 维度
   - 用于: 所有中间层（Attention, FFN）
   - 示例: Linear(768, 3072)

4. n_heads (12)
   - 注意力头数
   - 用于: Multi-Head Attention
   - 每个头的维度: 768 / 12 = 64

5. n_layers (12)
   - Transformer Block 数量
   - 用于: 模型深度

维度使用总结:
┌─────────────────┬──────────────┬──────────────┐
│      层         │   输入维度   │   使用的配置  │
├─────────────────┼──────────────┼──────────────┤
│ Token Embedding │ vocab_size   │ emb_dim      │
│ Pos Embedding   │ context_len  │ emb_dim      │
│ Attention       │ emb_dim      │ emb_dim      │
│ FFN             │ emb_dim      │ emb_dim      │
│ Output Head     │ emb_dim      │ vocab_size   │
└─────────────────┴──────────────┴──────────────┘
    """)


def correct_ffn_implementation():
    """完整的正确 FFN 实现"""
    print("\n" + "=" * 70)
    print("完整的 FFN 实现（带 Dropout）")
    print("=" * 70)

    class FeedForward(nn.Module):
        """完整的 Feed-Forward Network"""
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                nn.GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
                nn.Dropout(cfg["drop_rate"])
            )

        def forward(self, x):
            return self.layers(x)

    print("""
```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 扩展 4 倍
            nn.GELU(),                                       # 激活
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 还原
            nn.Dropout(cfg["drop_rate"])                    # Dropout
        )

    def forward(self, x):
        # x: [batch, seq_len, emb_dim]
        return self.layers(x)  # → [batch, seq_len, emb_dim]
```

关键点:
1. 输入维度: emb_dim (768)
2. 中间维度: 4 × emb_dim (3072)  ← 扩展 4 倍！
3. 输出维度: emb_dim (768)       ← 还原

维度流:
[batch, seq_len, 768] → [batch, seq_len, 3072] → [batch, seq_len, 768]
                    ↑ 扩展                  ↑ 压缩
    """)

    # 测试
    cfg = {
        "emb_dim": 768,
        "drop_rate": 0.1
    }

    ffn = FeedForward(cfg)
    x = torch.rand(2, 3, 768)
    out = ffn(x)

    print(f"\n测试:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  ✅ 维度匹配！")


def dimension_flow_visualization():
    """可视化维度流动"""
    print("\n" + "=" * 70)
    print("FFN 中的维度流动")
    print("=" * 70)

    print("""
完整的 Transformer Block 维度流:

输入: [batch, seq_len, emb_dim]
      [2, 3, 768]
         ↓
   Multi-Head Attention
      [2, 3, 768]
         ↓ (+残差连接)
   Layer Norm
      [2, 3, 768]
         ↓
   Feed-Forward Network:
      │
      ├─► Linear: [768] → [3072]
      │   [2, 3, 768] → [2, 3, 3072]
      │
      ├─► GELU
      │   [2, 3, 3072] → [2, 3, 3072]
      │
      └─► Linear: [3072] → [768]
          [2, 3, 3072] → [2, 3, 768]
         ↓ (+残差连接)
   Layer Norm
      [2, 3, 768]
         ↓
输出: [2, 3, 768]

关键观察:
1. Attention 不改变维度
2. FFN 先扩展 4 倍，再压缩回来
3. 残差连接要求输入输出维度相同
4. 整个 Block 输入输出维度相同
    """)


def common_mistakes():
    """常见错误总结"""
    print("\n" + "=" * 70)
    print("常见错误汇总")
    print("=" * 70)

    print("""
错误 1: 使用 context_length 而不是 emb_dim
❌ nn.Linear(cfg["context_length"], ...)
✅ nn.Linear(cfg["emb_dim"], ...)

错误 2: 忘记 4 倍扩展
❌ nn.Linear(768, 768)
✅ nn.Linear(768, 3072)  # 4 × 768

错误 3: 维度顺序错误
❌ nn.Linear(3072, 768) → nn.Linear(768, 3072)
✅ nn.Linear(768, 3072) → nn.Linear(3072, 768)

错误 4: 混淆 seq_len 和 emb_dim
输入: [batch, seq_len, emb_dim]
      [2, 3, 768]
             ↑
        不是 seq_len！

错误 5: 忘记残差连接的维度要求
残差连接: x + FFN(x)
要求: x 和 FFN(x) 形状完全相同
所以: FFN 输入输出维度必须相同

调试技巧:
1. 打印每层的权重形状
   print(ffn.layers[0].weight.shape)

2. 打印中间输出形状
   x = self.layer1(x)
   print(f"After layer1: {x.shape}")

3. 使用小数据测试
   x = torch.rand(2, 3, 768)
   out = ffn(x)

4. 检查配置
   print(cfg["emb_dim"])  # 应该是 768
   print(cfg["context_length"])  # 应该是 1024
    """)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" FFN 维度错误诊断与修复 ".center(70, "="))
    print("=" * 70)

    # 1. 演示错误
    demonstrate_error()

    # 2. 解释维度
    explain_dimensions()

    # 3. 正确实现
    correct_ffn_implementation()

    # 4. 可视化流动
    dimension_flow_visualization()

    # 5. 常见错误
    common_mistakes()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
核心要点:

1. FFN 的维度:
   输入: emb_dim (768)
   中间: 4 × emb_dim (3072)
   输出: emb_dim (768)

2. 不要用 context_length:
   ❌ context_length (1024) - 这是序列长度
   ✅ emb_dim (768) - 这是向量维度

3. 维度流:
   [B, L, 768] → [B, L, 3072] → [B, L, 768]
                ↑ 扩展        ↑ 压缩

4. 修复你的代码:
   检查 FFN 的第一个 Linear 层
   确保是: nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])

5. 记忆口诀:
   "FFN 用 emb_dim，不用 context_length；
    先扩展四倍，再压缩回来。"
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
