"""
GPT 训练的多位置预测详解

演示：每个位置都有自己的预测任务和标签
"""

import torch
import torch.nn as nn


def explain_gpt_training():
    """详细解释 GPT 的训练过程"""

    print("=" * 80)
    print(" GPT 训练：多位置预测详解 ".center(80, "="))
    print("=" * 80)

    # 假设参数
    batch_size = 1
    seq_len = 4
    vocab_size = 10  # 简化词表

    # 输入和目标
    input_ids = torch.tensor([[2, 5, 3, 7]])    # [batch, seq_len]
    target_ids = torch.tensor([[5, 3, 7, 1]])   # [batch, seq_len]

    print("\n输入序列:")
    print(f"  input_ids  = {input_ids.tolist()[0]}")
    print(f"  target_ids = {target_ids.tolist()[0]}")

    # 模拟模型输出（随机 logits）
    torch.manual_seed(123)
    logits = torch.randn(batch_size, seq_len, vocab_size)

    print("\n模型输出 logits 的形状:")
    print(f"  logits.shape = {logits.shape}")
    print(f"  含义: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]")

    print("\n" + "=" * 80)
    print("每个位置的预测任务:")
    print("=" * 80)

    for pos in range(seq_len):
        # 当前位置的 logits
        logits_at_pos = logits[0, pos, :]  # [vocab_size]

        # 预测的 token（概率最大的）
        predicted_token = torch.argmax(logits_at_pos).item()

        # 真实的目标 token
        target_token = target_ids[0, pos].item()

        # 计算单个位置的损失
        loss_at_pos = nn.CrossEntropyLoss()(
            logits_at_pos.unsqueeze(0),  # [1, vocab_size]
            target_ids[0, pos].unsqueeze(0)  # [1]
        )

        print(f"\n位置 {pos}:")
        print(f"  - 输入上下文: {input_ids[0, :pos+1].tolist()}")
        print(f"  - 模型预测: token_{predicted_token}")
        print(f"  - 真实标签: token_{target_token} {'✓' if predicted_token == target_token else '✗'}")
        print(f"  - 该位置损失: {loss_at_pos.item():.4f}")

    print("\n" + "=" * 80)
    print("总损失计算:")
    print("=" * 80)

    # 方法 1: 使用 PyTorch 的 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # 将 logits 和 targets 展平
    logits_flat = logits.view(-1, vocab_size)  # [batch*seq_len, vocab_size]
    targets_flat = target_ids.view(-1)         # [batch*seq_len]

    total_loss = criterion(logits_flat, targets_flat)

    print(f"\n总损失 = 所有位置损失的平均值")
    print(f"  CrossEntropyLoss(logits, targets) = {total_loss.item():.4f}")

    # 方法 2: 手动计算验证
    manual_loss = 0
    for pos in range(seq_len):
        loss_at_pos = nn.CrossEntropyLoss()(
            logits[0, pos].unsqueeze(0),
            target_ids[0, pos].unsqueeze(0)
        )
        manual_loss += loss_at_pos
    manual_loss /= seq_len

    print(f"  手动计算 (验证) = {manual_loss.item():.4f}")

    print("\n" + "=" * 80)
    print("关键理解:")
    print("=" * 80)
    print("""
1. GPT 不是只预测最后一个 token，而是同时预测序列中每个位置的下一个 token

2. 每个位置的预测任务:
   - 位置 0: 看到 [2]       → 预测 5
   - 位置 1: 看到 [2, 5]    → 预测 3
   - 位置 2: 看到 [2, 5, 3] → 预测 7
   - 位置 3: 看到 [2, 5, 3, 7] → 预测 1

3. target_ids 中的每个元素都是对应位置的标签，全部都有用！

4. 总损失 = 所有位置损失的平均值

5. 这种训练方式叫做 "Teacher Forcing"：
   - 训练时，每个位置都用真实的历史 token
   - 推理时，用模型自己生成的 token
    """)


def demonstrate_attention_mask():
    """演示 Attention Mask 如何确保每个位置只看到之前的内容"""

    print("\n" + "=" * 80)
    print(" Causal Attention Mask 的作用 ".center(80, "="))
    print("=" * 80)

    seq_len = 4

    # 创建 causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask_display = mask.long()

    print("\nCausal Attention Mask (1 表示被遮挡，0 表示可见):")
    print("\n     Token_0  Token_1  Token_2  Token_3")
    for i in range(seq_len):
        row = "  ".join([str(mask_display[i, j].item()) for j in range(seq_len)])
        print(f"Pos_{i}    {row}")

    print("\n含义解释:")
    print("""
  Pos_0: 只能看到 Token_0                      (预测 Token_1)
  Pos_1: 只能看到 Token_0, Token_1             (预测 Token_2)
  Pos_2: 只能看到 Token_0, Token_1, Token_2    (预测 Token_3)
  Pos_3: 可以看到所有 Token                    (预测 Token_4)

这确保了每个位置只能利用"过去"的信息来预测"未来"
    """)


def real_training_example():
    """真实的训练示例"""

    print("\n" + "=" * 80)
    print(" 真实训练示例：文本 'Hello world' ".center(80, "="))
    print("=" * 80)

    # 假设的 tokenization
    text = "Hello world!"
    tokens = ["Hello", " world", "!"]
    token_ids = [15496, 995, 0]  # 假设的 ID

    print(f"\n原始文本: '{text}'")
    print(f"Token 序列: {tokens}")
    print(f"Token IDs: {token_ids}")

    # 构造训练数据
    input_ids = token_ids[:-1]    # [15496, 995]
    target_ids = token_ids[1:]    # [995, 0]

    print("\n训练数据构造:")
    print(f"  input_ids  = {input_ids}")
    print(f"  target_ids = {target_ids}")

    print("\n训练任务分解:")
    print(f"  位置 0: 输入='Hello' (15496)        → 目标=' world' (995)")
    print(f"  位置 1: 输入='Hello world' (15496,995) → 目标='!' (0)")

    print("\n模型在做什么:")
    print("""
  1. 看到 "Hello" → 学习预测 " world"
  2. 看到 "Hello world" → 学习预测 "!"

  两个预测任务同时训练！这就是为什么 target_ids 的所有元素都有用。
    """)


def main():
    """主函数"""
    explain_gpt_training()
    demonstrate_attention_mask()
    real_training_example()

    print("\n" + "=" * 80)
    print(" 总结 ".center(80, "="))
    print("=" * 80)
    print("""
核心要点:

1. ❌ 错误理解: target_ids 只有最后一个元素有用
   ✅ 正确理解: target_ids 的每个元素都是对应位置的标签

2. GPT 训练 = 同时训练多个"预测下一个词"的任务
   - 位置越多，训练任务越多，学习效率越高

3. 这种方式的优势:
   - 一条序列可以产生多个训练样本 (seq_len 个)
   - 充分利用数据，提高训练效率
   - 每个位置都学习"在不同上下文长度下预测下一个词"

4. Causal Attention Mask 确保公平性:
   - 每个位置只能看到之前的 token
   - 避免"作弊"(看到未来信息)

5. 推理时的区别:
   - 训练: 并行预测所有位置 (Teacher Forcing)
   - 推理: 逐个生成 (Autoregressive)
    """)
    print("=" * 80)


if __name__ == "__main__":
    main()
