"""
GPT模型中dropout的完整位置解析
展示drop_rate在attention机制和整个模型中的所有应用位置
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MultiHeadAttention(nn.Module):
    """
    标准的Multi-Head Self-Attention实现
    展示dropout的两个关键位置
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # QKV投影层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 输出投影
        self.out_proj = nn.Linear(d_out, d_out)

        # ⭐️ DROPOUT位置1: Attention权重dropout
        # 在softmax之后,对attention weights应用dropout
        self.dropout = nn.Dropout(dropout)

        # Causal mask (下三角矩阵)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 1. 计算QKV
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 2. 分割多头
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置为 (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 3. 计算attention scores
        attn_scores = queries @ keys.transpose(2, 3)

        # 4. 应用causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 5. Softmax归一化
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)

        # ⭐️ DROPOUT位置1应用: 对attention weights dropout
        # 这会随机丢弃一些attention连接,防止过度依赖某些token
        attn_weights = self.dropout(attn_weights)

        # 6. 加权求和values
        context_vec = attn_weights @ values

        # 7. 拼接多头
        context_vec = context_vec.transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 8. 输出投影
        context_vec = self.out_proj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    """
    完整的Transformer Block
    展示所有dropout位置
    """
    def __init__(self, cfg):
        super().__init__()

        # Multi-Head Attention (内部有dropout)
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],  # ⭐️ 传入drop_rate
            qkv_bias=cfg["qkv_bias"]
        )

        # Feed-Forward Network
        self.ff = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])

        # ⭐️ DROPOUT位置2: Attention输出的residual connection dropout
        # 在残差连接之前对attention输出应用dropout
        self.drop_resid1 = nn.Dropout(cfg["drop_rate"])

        # ⭐️ DROPOUT位置3: FFN输出的residual connection dropout
        # 在残差连接之前对FFN输出应用dropout
        self.drop_resid2 = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention + Residual + Norm
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid1(x)  # ⭐️ DROPOUT位置2应用
        x = x + shortcut

        # FFN + Residual + Norm
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid2(x)  # ⭐️ DROPOUT位置3应用
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """
    完整的GPT模型
    """
    def __init__(self, cfg):
        super().__init__()

        # Token + Position Embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # ⭐️ DROPOUT位置4: Embedding dropout
        # 在embedding层之后应用dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # Embeddings
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds

        # ⭐️ DROPOUT位置4应用
        x = self.drop_emb(x)

        # Transformer blocks
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


def visualize_dropout_locations():
    """可视化dropout在GPT架构中的所有位置"""
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # 标题
    ax.text(5, 19, 'GPT模型中drop_rate的4个应用位置',
            fontsize=18, fontweight='bold', ha='center')

    # =============== 位置4: Embedding Dropout ===============
    y_start = 17

    # Input tokens
    ax.add_patch(FancyBboxPatch((3, y_start), 4, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(5, y_start+0.3, 'Input Tokens', ha='center', va='center', fontsize=11)

    # Arrow
    ax.annotate('', xy=(5, y_start-0.5), xytext=(5, y_start),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Token + Position Embedding
    y_start -= 1
    ax.add_patch(FancyBboxPatch((2, y_start), 6, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFE4B5', edgecolor='black', linewidth=2))
    ax.text(5, y_start+0.4, 'Token Emb + Position Emb',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Dropout 4
    ax.annotate('', xy=(5, y_start-0.5), xytext=(5, y_start),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    y_start -= 0.8
    ax.add_patch(FancyBboxPatch((3.5, y_start), 3, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor='#FF6B6B', edgecolor='red', linewidth=2))
    ax.text(5, y_start+0.25, '④ Embedding Dropout',
            ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # =============== Transformer Block (重复n_layers次) ===============
    y_start -= 1.2

    # Block框
    block_height = 10
    ax.add_patch(FancyBboxPatch((0.5, y_start-block_height), 9, block_height,
                                boxstyle="round,pad=0.1",
                                facecolor='#F0F0F0', edgecolor='blue',
                                linewidth=3, linestyle='--', alpha=0.5))
    ax.text(5, y_start-block_height/2, 'Transformer Block\n(重复 n_layers 次)',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='blue', alpha=0.7)

    # Layer Norm 1
    y_start -= 1
    ax.add_patch(FancyBboxPatch((3.5, y_start), 3, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor='#E8E8E8', edgecolor='black', linewidth=1.5))
    ax.text(5, y_start+0.25, 'Layer Norm', ha='center', va='center', fontsize=10)

    # Multi-Head Attention
    ax.annotate('', xy=(5, y_start-0.5), xytext=(5, y_start),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    y_start -= 1.3
    # Attention框
    ax.add_patch(FancyBboxPatch((2, y_start-2), 6, 2,
                                boxstyle="round,pad=0.1",
                                facecolor='#B8E6B8', edgecolor='green', linewidth=2))

    # Attention内部
    ax.text(5, y_start-0.3, 'Multi-Head Attention',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Q, K, V
    ax.text(5, y_start-0.7, 'Q @ K^T → Attention Scores',
            ha='center', va='center', fontsize=9)
    ax.text(5, y_start-1, '↓ Softmax ↓',
            ha='center', va='center', fontsize=9)

    # Dropout 1 (在attention weights上)
    ax.add_patch(FancyBboxPatch((2.5, y_start-1.6), 5, 0.4,
                                boxstyle="round,pad=0.05",
                                facecolor='#FF6B6B', edgecolor='red', linewidth=2))
    ax.text(5, y_start-1.4, '① Attention Weights Dropout',
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.text(5, y_start-1.85, 'Attention × V → Context',
            ha='center', va='center', fontsize=9)

    # Dropout 2 (residual)
    y_start -= 2.5
    ax.annotate('', xy=(5, y_start-0.3), xytext=(5, y_start),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax.add_patch(FancyBboxPatch((3, y_start-0.8), 4, 0.4,
                                boxstyle="round,pad=0.05",
                                facecolor='#FF6B6B', edgecolor='red', linewidth=2))
    ax.text(5, y_start-0.6, '② Residual Dropout',
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Residual connection
    y_start -= 1.2
    ax.annotate('', xy=(7.5, y_start+3.5), xytext=(7.5, y_start),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple', linestyle='--'))
    ax.text(8.2, y_start+1.7, 'Residual', ha='left', va='center',
            fontsize=9, color='purple', fontweight='bold', rotation=90)

    ax.add_patch(mpatches.Circle((5, y_start), 0.3,
                                facecolor='white', edgecolor='black', linewidth=2))
    ax.text(5, y_start, '+', ha='center', va='center', fontsize=16, fontweight='bold')

    # Layer Norm 2
    ax.annotate('', xy=(5, y_start-0.5), xytext=(5, y_start-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    y_start -= 1
    ax.add_patch(FancyBboxPatch((3.5, y_start), 3, 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor='#E8E8E8', edgecolor='black', linewidth=1.5))
    ax.text(5, y_start+0.25, 'Layer Norm', ha='center', va='center', fontsize=10)

    # FFN
    ax.annotate('', xy=(5, y_start-0.5), xytext=(5, y_start),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    y_start -= 1.3
    ax.add_patch(FancyBboxPatch((2.5, y_start), 5, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFD700', edgecolor='orange', linewidth=2))
    ax.text(5, y_start+0.4, 'Feed-Forward Network (FFN)',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Dropout 3
    ax.annotate('', xy=(5, y_start-0.5), xytext=(5, y_start),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    y_start -= 0.9
    ax.add_patch(FancyBboxPatch((3, y_start), 4, 0.4,
                                boxstyle="round,pad=0.05",
                                facecolor='#FF6B6B', edgecolor='red', linewidth=2))
    ax.text(5, y_start+0.2, '③ Residual Dropout',
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Residual connection 2
    y_start -= 0.5
    ax.annotate('', xy=(2.5, y_start+3.5), xytext=(2.5, y_start),
                arrowprops=dict(arrowstyle='->', lw=2, color='purple', linestyle='--'))
    ax.text(1.8, y_start+1.7, 'Residual', ha='right', va='center',
            fontsize=9, color='purple', fontweight='bold', rotation=90)

    ax.add_patch(mpatches.Circle((5, y_start), 0.3,
                                facecolor='white', edgecolor='black', linewidth=2))
    ax.text(5, y_start, '+', ha='center', va='center', fontsize=16, fontweight='bold')

    # Output
    y_start -= 1
    ax.annotate('', xy=(5, y_start), xytext=(5, y_start+0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.add_patch(FancyBboxPatch((3, y_start-0.5), 4, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(5, y_start-0.2, 'Output Logits', ha='center', va='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('/Users/zhuji_m4pro/code/LLM_from_scrach/LLMs-from-scratch/Playgrounds/dropout_locations.png',
                dpi=150, bbox_inches='tight')
    print("✓ 架构图已保存: dropout_locations.png")


def visualize_attention_dropout_detail():
    """详细展示attention内部的dropout机制"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 模拟数据
    seq_len = 6
    num_heads = 2

    # 1. 原始attention scores (softmax前)
    np.random.seed(42)
    attn_scores = np.random.randn(seq_len, seq_len)
    # 应用causal mask
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    attn_scores_masked = np.where(mask, -np.inf, attn_scores)

    im1 = axes[0, 0].imshow(attn_scores_masked, cmap='RdYlBu_r', aspect='auto')
    axes[0, 0].set_title('步骤1: Attention Scores (Softmax前)',
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Key位置')
    axes[0, 0].set_ylabel('Query位置')
    plt.colorbar(im1, ax=axes[0, 0])

    # 添加causal mask标注
    for i in range(seq_len):
        for j in range(seq_len):
            if mask[i, j]:
                axes[0, 0].add_patch(mpatches.Rectangle((j-0.5, i-0.5), 1, 1,
                                     fill=True, facecolor='black', alpha=0.3))

    # 2. Attention weights (softmax后)
    from scipy.special import softmax
    attn_weights = softmax(attn_scores_masked, axis=-1)
    attn_weights = np.nan_to_num(attn_weights)  # 处理-inf产生的nan

    im2 = axes[0, 1].imshow(attn_weights, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('步骤2: Attention Weights (Softmax后)\n归一化为概率分布',
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Key位置')
    axes[0, 1].set_ylabel('Query位置')
    plt.colorbar(im2, ax=axes[0, 1])

    # 添加数值
    for i in range(seq_len):
        for j in range(seq_len):
            if not mask[i, j]:
                text = axes[0, 1].text(j, i, f'{attn_weights[i, j]:.2f}',
                                      ha="center", va="center", color="white", fontsize=8)

    # 3. Dropout mask (drop_rate=0.3)
    drop_rate = 0.3
    dropout_mask = np.random.binomial(1, 1-drop_rate, size=(seq_len, seq_len))
    dropout_mask = dropout_mask * (1 / (1 - drop_rate))  # 缩放补偿

    im3 = axes[1, 0].imshow(dropout_mask, cmap='RdYlGn', aspect='auto')
    axes[1, 0].set_title(f'步骤3: Dropout Mask (drop_rate={drop_rate})\n白色=保留(×{1/(1-drop_rate):.2f}), 红色=丢弃(×0)',
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Key位置')
    axes[1, 0].set_ylabel('Query位置')
    plt.colorbar(im3, ax=axes[1, 0])

    # 4. Dropout后的attention weights
    attn_weights_dropped = attn_weights * dropout_mask

    im4 = axes[1, 1].imshow(attn_weights_dropped, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    axes[1, 1].set_title('步骤4: Dropout后的Attention Weights\n随机断开部分连接,防止过拟合',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Key位置')
    axes[1, 1].set_ylabel('Query位置')
    plt.colorbar(im4, ax=axes[1, 1])

    # 添加数值
    for i in range(seq_len):
        for j in range(seq_len):
            if not mask[i, j]:
                color = "white" if attn_weights_dropped[i, j] > 0.3 else "yellow"
                text = axes[1, 1].text(j, i, f'{attn_weights_dropped[i, j]:.2f}',
                                      ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig('/Users/zhuji_m4pro/code/LLM_from_scrach/LLMs-from-scratch/Playgrounds/attention_dropout_detail.png',
                dpi=150, bbox_inches='tight')
    print("✓ Attention dropout详细图已保存: attention_dropout_detail.png")


def print_summary():
    """打印总结信息"""
    summary = """
================================================================================
                    GPT模型中 drop_rate 的4个应用位置总结
================================================================================

📍 位置① : Attention Weights Dropout (在MultiHeadAttention内部)
    - 时机: Softmax之后, 与Values相乘之前
    - 作用: 随机断开部分attention连接
    - 效果: 防止模型过度依赖某些特定token,增强泛化能力
    - 代码: attn_weights = self.dropout(attn_weights)

📍 位置② : Attention输出的Residual Dropout (在TransformerBlock中)
    - 时机: Attention输出之后, 残差连接之前
    - 作用: 对attention的输出进行dropout
    - 效果: 稳定训练,防止梯度爆炸
    - 代码: x = self.drop_resid1(x)

📍 位置③ : FFN输出的Residual Dropout (在TransformerBlock中)
    - 时机: FFN输出之后, 残差连接之前
    - 作用: 对FFN的输出进行dropout
    - 效果: 与位置②类似,稳定训练
    - 代码: x = self.drop_resid2(x)

📍 位置④ : Embedding Dropout (在GPTModel的最开始)
    - 时机: Token + Position Embedding之后
    - 作用: 对输入的embedding进行dropout
    - 效果: 防止模型过度记忆训练数据的embedding模式
    - 代码: x = self.drop_emb(x)

================================================================================
                               关键理解
================================================================================

1. **为什么Attention Weights需要Dropout?**
   - Attention机制会学习到token之间的依赖关系
   - 训练时如果某些连接过强,模型会过度依赖这些路径
   - Dropout随机断开连接,强迫模型学习多条冗余路径
   - 类似于集成学习的效果

2. **Dropout的数值稳定性技巧**
   - 训练时: dropped_value = value × mask / (1 - drop_rate)
   - 测试时: 直接使用value (不应用dropout)
   - 为什么缩放? 保证期望值不变: E[dropped] = E[value]

3. **典型的drop_rate设置**
   - GPT-2: 0.1 (10%)
   - BERT: 0.1 (10%)
   - 大模型通常: 0.0 - 0.1
   - 小数据集: 0.3 - 0.5 (需要更强的正则化)

4. **Dropout vs Layer Norm**
   - Dropout: 随机性正则化,防止过拟合
   - Layer Norm: 确定性归一化,稳定训练
   - 两者配合使用,效果最佳

5. **训练vs推理模式**
   - model.train(): 启用dropout (随机丢弃)
   - model.eval(): 禁用dropout (保留所有连接)
   - 切换模式很重要!

================================================================================
"""
    print(summary)


if __name__ == "__main__":
    print("开始生成Dropout位置可视化...\n")

    # 生成架构图
    visualize_dropout_locations()

    # 生成attention内部dropout的详细图
    visualize_attention_dropout_detail()

    # 打印总结
    print_summary()

    print("\n" + "="*80)
    print("所有可视化已完成!")
    print("="*80)

    # 展示配置示例
    print("\n示例配置 (GPT-124M):")
    GPT_CONFIG_124M = {
        "vocab_size": 50257,      # 词汇表大小
        "context_length": 1024,   # 最大序列长度
        "emb_dim": 768,           # Embedding维度
        "n_heads": 12,            # Attention头数
        "n_layers": 12,           # Transformer层数
        "drop_rate": 0.1,         # ⭐️ Dropout率 (应用在上述4个位置)
        "qkv_bias": False         # QKV投影是否使用bias
    }

    for key, value in GPT_CONFIG_124M.items():
        print(f"  {key:20s}: {value}")
