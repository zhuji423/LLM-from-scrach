"""
词表扩展方案详解

演示如何安全地扩展已训练模型的词表
"""

import torch
import torch.nn as nn
import copy


def extend_vocabulary(model, old_vocab_size, new_vocab_size, emb_dim=768):
    """
    扩展模型词表

    参数:
        model: 原始模型
        old_vocab_size: 原词表大小
        new_vocab_size: 新词表大小
        emb_dim: 嵌入维度
    """
    print("=" * 70)
    print("词表扩展方案")
    print("=" * 70)

    num_new_tokens = new_vocab_size - old_vocab_size
    print(f"\n原词表大小: {old_vocab_size}")
    print(f"新词表大小: {new_vocab_size}")
    print(f"新增词数量: {num_new_tokens}")

    # 1. 扩展 Token Embedding
    print("\n步骤 1: 扩展 Token Embedding 层")
    print("-" * 70)

    old_tok_emb = model.tok_emb.weight.data  # [old_vocab_size, emb_dim]

    # 创建新的 embedding 层
    new_tok_emb = nn.Embedding(new_vocab_size, emb_dim)

    # 复制旧权重
    new_tok_emb.weight.data[:old_vocab_size] = old_tok_emb

    # 初始化新词的 embedding
    # 方法 1: 正态分布（与预训练模型的初始化一致）
    nn.init.normal_(new_tok_emb.weight.data[old_vocab_size:], mean=0.0, std=0.02)

    # 方法 2: 用已有词的平均值（更保守）
    # mean_emb = old_tok_emb.mean(dim=0)
    # new_tok_emb.weight.data[old_vocab_size:] = mean_emb.unsqueeze(0).repeat(num_new_tokens, 1)

    model.tok_emb = new_tok_emb
    print(f"  旧 shape: {old_tok_emb.shape}")
    print(f"  新 shape: {new_tok_emb.weight.shape}")
    print(f"  新增部分初始化: 正态分布 N(0, 0.02)")

    # 2. 扩展 Output Head
    print("\n步骤 2: 扩展 Output Head 层")
    print("-" * 70)

    old_out_head = model.out_head.weight.data  # [old_vocab_size, emb_dim]

    # 创建新的输出层
    new_out_head = nn.Linear(emb_dim, new_vocab_size, bias=False)

    # 复制旧权重
    new_out_head.weight.data[:old_vocab_size] = old_out_head

    # 初始化新词的输出权重
    nn.init.normal_(new_out_head.weight.data[old_vocab_size:], mean=0.0, std=0.02)

    model.out_head = new_out_head
    print(f"  旧 shape: {old_out_head.shape}")
    print(f"  新 shape: {new_out_head.weight.shape}")
    print(f"  新增部分初始化: 正态分布 N(0, 0.02)")

    print("\n✅ 词表扩展完成！")

    return model


def demonstrate_extension():
    """演示完整的扩展流程"""
    print("\n" + "=" * 70)
    print("完整扩展流程演示")
    print("=" * 70)

    # 模拟原模型
    class SimpleGPT(nn.Module):
        def __init__(self, vocab_size, emb_dim):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, emb_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(emb_dim, nhead=8, batch_first=True),
                num_layers=2
            )
            self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

        def forward(self, x):
            x = self.tok_emb(x)
            x = self.transformer(x)
            logits = self.out_head(x)
            return logits

    # 创建原模型
    old_vocab_size = 50257
    emb_dim = 768
    model = SimpleGPT(old_vocab_size, emb_dim)

    print(f"\n原模型参数:")
    print(f"  tok_emb:  {model.tok_emb.weight.shape}")
    print(f"  out_head: {model.out_head.weight.shape}")

    # 模拟已训练（填充非零值）
    with torch.no_grad():
        model.tok_emb.weight.fill_(1.0)
        model.out_head.weight.fill_(1.0)

    # 测试原模型
    test_input = torch.randint(0, old_vocab_size, (2, 10))
    output = model(test_input)
    print(f"\n原模型输出 shape: {output.shape}")

    # 扩展词表
    new_vocab_size = 50500
    model = extend_vocabulary(model, old_vocab_size, new_vocab_size, emb_dim)

    print(f"\n扩展后模型参数:")
    print(f"  tok_emb:  {model.tok_emb.weight.shape}")
    print(f"  out_head: {model.out_head.weight.shape}")

    # 测试扩展后的模型
    # 旧词
    test_input_old = torch.randint(0, old_vocab_size, (2, 10))
    output_old = model(test_input_old)
    print(f"\n测试旧词 - 输出 shape: {output_old.shape}")

    # 新词
    test_input_new = torch.randint(old_vocab_size, new_vocab_size, (2, 10))
    output_new = model(test_input_new)
    print(f"测试新词 - 输出 shape: {output_new.shape}")

    # 检查旧权重是否保留
    print(f"\n验证旧权重是否保留:")
    old_emb_preserved = (model.tok_emb.weight[:old_vocab_size] == 1.0).all()
    print(f"  tok_emb 旧权重保留: {old_emb_preserved}")

    new_emb_initialized = (model.tok_emb.weight[old_vocab_size:] != 1.0).any()
    print(f"  tok_emb 新权重已初始化: {new_emb_initialized}")


def compare_initialization_methods():
    """对比不同的新词初始化方法"""
    print("\n" + "=" * 70)
    print("新词 Embedding 初始化方法对比")
    print("=" * 70)

    old_vocab_size = 50257
    new_vocab_size = 50500
    emb_dim = 768
    num_new_tokens = new_vocab_size - old_vocab_size

    # 已有词的 embedding（模拟）
    old_embeddings = torch.randn(old_vocab_size, emb_dim) * 0.02

    print(f"\n已有词 embedding 统计:")
    print(f"  均值: {old_embeddings.mean().item():.6f}")
    print(f"  标准差: {old_embeddings.std().item():.6f}")
    print(f"  最小值: {old_embeddings.min().item():.6f}")
    print(f"  最大值: {old_embeddings.max().item():.6f}")

    # 方法 1: 随机初始化（与预训练一致）
    method1 = torch.randn(num_new_tokens, emb_dim) * 0.02
    print(f"\n方法 1: 随机初始化 N(0, 0.02)")
    print(f"  均值: {method1.mean().item():.6f}")
    print(f"  标准差: {method1.std().item():.6f}")
    print(f"  优点: 与模型初始化一致，保留随机性")
    print(f"  缺点: 可能与已有词差异较大")

    # 方法 2: 全局平均
    mean_emb = old_embeddings.mean(dim=0)
    method2 = mean_emb.unsqueeze(0).repeat(num_new_tokens, 1)
    print(f"\n方法 2: 使用已有词的平均值")
    print(f"  均值: {method2.mean().item():.6f}")
    print(f"  标准差: {method2.std().item():.6f}")
    print(f"  优点: 与已有词分布接近")
    print(f"  缺点: 所有新词 embedding 完全相同")

    # 方法 3: 平均 + 小扰动
    method3 = mean_emb.unsqueeze(0).repeat(num_new_tokens, 1)
    method3 += torch.randn(num_new_tokens, emb_dim) * 0.01  # 添加小扰动
    print(f"\n方法 3: 平均值 + 小扰动")
    print(f"  均值: {method3.mean().item():.6f}")
    print(f"  标准差: {method3.std().item():.6f}")
    print(f"  优点: 兼顾稳定性和多样性")
    print(f"  缺点: 需要调整扰动大小")

    # 方法 4: 最近邻平均（语义相似词）
    # 假设为每个新词找到 k 个语义相似的已有词
    k = 5
    method4_list = []
    for _ in range(num_new_tokens):
        # 随机选择 k 个已有词（实际中应该根据语义选择）
        similar_indices = torch.randint(0, old_vocab_size, (k,))
        similar_embs = old_embeddings[similar_indices]
        new_emb = similar_embs.mean(dim=0)
        method4_list.append(new_emb)
    method4 = torch.stack(method4_list)
    print(f"\n方法 4: 语义相似词的平均")
    print(f"  均值: {method4.mean().item():.6f}")
    print(f"  标准差: {method4.std().item():.6f}")
    print(f"  优点: 利用语义信息，效果最好")
    print(f"  缺点: 需要额外的语义映射")

    print(f"\n推荐方案:")
    print(f"  - 快速扩展: 方法 1（随机初始化）")
    print(f"  - 稳定扩展: 方法 3（平均+扰动）")
    print(f"  - 最佳效果: 方法 4（语义相似）+ 继续训练")


def fine_tuning_strategy():
    """扩展后的微调策略"""
    print("\n" + "=" * 70)
    print("扩展后的微调策略")
    print("=" * 70)

    print("""
扩展词表后，必须进行微调（Fine-tuning）！

1. 冻结策略（推荐）:
   ├─ 冻结 Transformer 层（已训练好）
   ├─ 只训练新增的 embedding 和 output 权重
   └─ 训练速度快，不会破坏原有知识

   代码示例:
   ```python
   for name, param in model.named_parameters():
       if 'tok_emb' in name or 'out_head' in name:
           param.requires_grad = True  # 只训练这两层
       else:
           param.requires_grad = False  # 冻结其他层
   ```

2. 逐层解冻策略:
   ├─ 第 1 阶段: 只训练新增部分（1-2 epoch）
   ├─ 第 2 阶段: 解冻最后几层 Transformer（2-3 epoch）
   └─ 第 3 阶段: 全模型微调（可选，1-2 epoch）

3. 学习率设置:
   ├─ 新增部分: 较大学习率（如 1e-4）
   ├─ 已有部分: 较小学习率（如 1e-5）
   └─ 使用不同的学习率组

   代码示例:
   ```python
   optimizer = torch.optim.AdamW([
       {'params': new_params, 'lr': 1e-4},
       {'params': old_params, 'lr': 1e-5}
   ])
   ```

4. 训练数据要求:
   ├─ 必须包含新词的样本
   ├─ 数据量: 至少几千到几万条
   ├─ 覆盖新词的各种用法
   └─ 建议混合新旧词的数据

5. 训练时长:
   ├─ 最少: 1-2 epoch（紧急情况）
   ├─ 推荐: 3-5 epoch（一般情况）
   └─ 充分: 10+ epoch（重要应用）

6. 验证指标:
   ├─ 新词的困惑度（Perplexity）
   ├─ 新词的预测准确率
   ├─ 旧词的性能不能下降
   └─ 下游任务的性能
    """)


def practical_examples():
    """实际应用案例"""
    print("\n" + "=" * 70)
    print("实际应用案例")
    print("=" * 70)

    print("""
案例 1: 添加专业术语
  场景: 医疗领域 GPT，添加医学专业词汇
  方法:
    - 收集医学术语（如 "COVID-19", "mRNA疫苗"）
    - 扩展词表 (+500 词)
    - 用医学文献微调 3-5 epoch
  结果:
    - 模型能正确理解和生成医学术语
    - 通用能力保持不变

案例 2: 多语言支持
  场景: 英文 GPT 添加中文支持
  方法:
    - 添加中文 tokenizer (+20,000 词)
    - 用中英双语数据微调
    - 冻结大部分 Transformer 层
  结果:
    - 支持中文输入输出
    - 英文能力略有下降（可接受）
    - 参数量增加 ~3%

案例 3: 特定域名词
  场景: 添加公司/产品名称
  方法:
    - 添加新词 (+100 词)
    - 用公司内部文档微调
    - 只训练 embedding 层
  结果:
    - 正确识别公司名称
    - 微调时间短（几小时）

案例 4: 表情符号支持
  场景: 社交媒体 GPT 添加 emoji
  方法:
    - 添加常用 emoji (+200)
    - 用社交媒体数据微调
    - 全模型训练（emoji 语义复杂）
  结果:
    - 能生成合适的 emoji
    - 理解 emoji 的情感含义
    """)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" 词表扩展完整指南 ".center(70, "="))
    print("=" * 70)

    # 1. 演示扩展流程
    demonstrate_extension()

    # 2. 对比初始化方法
    compare_initialization_methods()

    # 3. 微调策略
    fine_tuning_strategy()

    # 4. 实际案例
    practical_examples()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("""
词表扩展的关键点:

1. 影响的层:
   - Token Embedding (输入层)
   - Output Head (输出层)
   - Transformer 层不受影响

2. 操作步骤:
   ① 创建新的 embedding/output 层
   ② 复制旧权重
   ③ 初始化新词权重
   ④ **必须微调**

3. 初始化方法:
   - 推荐: 平均值 + 小扰动
   - 最佳: 语义相似词平均

4. 微调策略:
   - 冻结 Transformer
   - 只训练新增部分
   - 使用包含新词的数据

5. 注意事项:
   - 不微调会导致新词效果很差
   - 旧词性能可能略有下降
   - 需要准备包含新词的训练数据
   - 建议从小规模测试开始

6. 参数量影响:
   - 添加 n 个词
   - 增加参数: n × emb_dim × 2
   - 例如: 添加 243 词到 GPT-2
     → 增加 243 × 768 × 2 = 373,248 参数
     → 增加 ~1.4 MB (float32)
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
