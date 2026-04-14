# 门控 DeltaNet 实现线性注意力

最近，[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 和 [Kimi Linear](https://arxiv.org/abs/2510.26692) 提出了混合 transformer，实现了注意力机制的替代方案，这些方案相对于上下文长度呈线性而非二次方缩放。

Qwen3-Next 和 Kimi Linear 都使用 3:1 的比例，即每三个使用线性 Gated DeltaNet 变体的 transformer 块，就有一个使用完整注意力的块，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next 与 Kimi Linear">



&nbsp;

## 简介与概述

Gated DeltaNet 是一种线性注意力变体，灵感来自循环神经网络，包括来自 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文的门控机制。在某种意义上，Gated DeltaNet 是带有 Mamba 风格门控的 DeltaNet，而 DeltaNet 本身是一种线性注意力机制。

Kimi Linear 通过 Kimi Delta Attention (KDA) 机制修改了 Qwen3-Next 的线性注意力机制，这本质上是 Gated DeltaNet 的改进。Qwen3-Next 使用标量门（每个注意力头一个值）来控制记忆衰减率，而 Kimi Linear 将其替换为针对每个特征维度的通道级门控。根据作者的说法，这提供了对记忆的更多控制，反过来又改善了长上下文推理。

此外，对于全注意力层，Kimi Linear 将 Qwen3-Next 的门控注意力层（本质上是带有输出门控的标准多头注意力层）替换为多头潜在注意力（MLA）。这与我们之前在 DeepSeek V3/R1 部分讨论的 MLA 机制相同，但增加了一个门。（回顾一下，MLA 压缩键/值空间以减少 KV 缓存大小。）

Kimi Linear 中的 MLA 不使用门，这是有意的，以便作者可以更直接地将架构与标准 MLA 进行比较，然而，他们[表示](https://x.com/yzhang_cs/status/1984631714464088563)计划在未来添加它。

由于我们已经在 [../05_mla](../05_mla) 中实现了 MLA，本补充材料重点介绍 Gated DeltaNet 部分。


&nbsp;
## 门控注意力

在我们进入 Gated DeltaNet 本身之前，让我们简要谈谈门。如前面图中 Qwen3-Next 架构的上半部分所示，Qwen3-Next 使用"门控注意力"。这本质上是常规完整注意力加上一个额外的 sigmoid 门。

这种门控是一个简单的修改，我在下面将其添加到第 3 章的 `MultiHeadAttention` 代码中以作说明：

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### 新增：添加门
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        ####################################################
        ### 新增：添加门
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### 新增：添加门
        context = context * torch.sigmoid(gate)
        ####################################################
        out = self.out_proj(context)
        return out
```



正如我们所见，在像往常一样计算注意力之后，模型使用来自相同输入的单独门控信号，对其应用 sigmoid 以保持在 0 和 1 之间，然后将其与注意力输出相乘。这允许模型动态地放大或缩小某些特征。Qwen3-Next 开发者[表示](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list)这有助于训练稳定性：

> [...] 注意力输出门控机制有助于消除注意力下沉（Attention Sink）和大规模激活（Massive Activation）等问题，确保模型在数值上的稳定性。


&nbsp;
## Gated DeltaNet

那么，什么是 Gated DeltaNet？Gated DeltaNet（*Gated Delta Network* 的缩写）是 Qwen3-Next 的线性注意力层，旨在作为标准 softmax 注意力的替代方案。它如前所述采用自 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文。

Gated DeltaNet 最初被提出作为 Mamba2 的改进版本，它将 Mamba2 的门控衰减机制与 delta 规则相结合。

Mamba 是一种状态空间模型（transformer 的替代方案），这是一个值得未来单独讨论的大话题。

delta 规则部分指的是计算新值和预测值之间的差值（delta, Δ）来更新用作记忆状态的隐藏状态（稍后详述）。

（附带说明：熟悉经典机器学习文献的读者可以将其类比为受生物学启发的赫布学习："一起发放的神经元连接在一起。"这本质上是感知器更新规则和基于梯度下降的学习的前身，但没有监督。）

Gated DeltaNet 有一个类似于前面讨论的门控注意力中的门，不同之处在于它使用 SiLU 而不是逻辑 sigmoid 激活，如下图所示。（选择 SiLU 可能是为了在标准 sigmoid 之上改善梯度流动和稳定性。）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="Gated DeltaNet" width=500px>

然而，如上图所示，Gated DeltaNet 中的"门控"还指几个额外的门：

- `α`（衰减门）控制记忆随时间衰减或重置的速度，
- `β`（更新门）控制新输入修改状态的强度。

在代码中，上图描绘的 Gated DeltaNet 的简化版本（没有卷积混合）可以实现如下（代码灵感来自 Qwen3 团队的[官方实现](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835)）。

（请注意，一些实现将衰减门称为 `gk`（步骤 k 的门），其中 `exp(gk)` 匹配论文中的 $\alpha_t$。为了保持这种关系的明确性，下面的代码片段将对数空间的门 `alpha_log` 与指数化的衰减 `alpha` 分开。）


```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### 新增：delta 规则和输出门控的门
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # 注意：衰减门 alpha 对应于
        # A_log + W_alpha(x) + dt_bias
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        # 我们可以将其实现为
        # W_alpha = nn.Linear(d_in, num_heads, bias=True)
        # 但偏置是分开的，为了可解释性以及
        # 模仿官方实现

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        ####################################################
        ### 新增：计算 delta 规则门
        beta = torch.sigmoid(self.W_beta(x))
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        alpha = alpha_log.exp()
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # 新增

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)
        gate = gate.transpose(1, 2)  # 新增

        ####################################################
        ### 新增：QKNorm 式归一化用于 delta 规则
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)
        ####################################################

        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        ####################################################
        ### 新增：门控 delta 规则更新
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
            ####################################################
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        ### 新增：应用 RMSNorm 和 SiLU 门
        context = self.norm(context)
        context = context * F.silu(gate)
        ####################################################

        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

（请注意，为简单起见，我省略了 Qwen3-Next 和 Kimi Linear 使用的卷积混合，以保持代码更具可读性并专注于循环方面。）

因此，如上所示，与标准（或门控）注意力有很多不同。

在门控注意力中，模型计算所有 token 之间的常规注意力（每个 token 关注或查看其他每个 token）。然后，在获得注意力输出后，一个门（sigmoid）决定保留多少输出。关键点是，它仍然是随上下文长度呈二次方缩放的常规缩放点积注意力。

回顾一下，缩放点积注意力计算为 softmax(QKᵀ)V，其中 Q 和 K 是 *n*×*d* 矩阵，*n* 是输入 token 的数量，*d* 是嵌入维度。因此 QKᵀ 产生一个 *n*×*n* 的注意力矩阵，然后与 *n*×*d* 维的值矩阵 V 相乘：

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="二次方注意力" width=500px />

在 Gated DeltaNet 中，没有 *n*×*n* 的注意力矩阵。相反，模型逐个处理 token。它保持一个运行中的记忆（状态），随着每个新 token 的到来而更新。这就是实现中的内容，其中 `S` 是在每个时间步 *t* 循环更新的状态。

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

门控制记忆如何变化：

- α（`alpha`）调节要遗忘多少旧记忆（衰减）。

- β（`beta`）调节时间步 *t* 的当前 token 更新记忆的程度。

（最终的输出门，在上面的代码片段中未显示，类似于门控注意力；它控制保留多少输出。）

因此，在某种意义上，Gated DeltaNet 中的状态更新类似于循环神经网络（RNN）的工作方式。优点是它随上下文长度呈线性（通过 for 循环）而非二次方缩放。

这种循环状态更新的缺点是，与常规（或门控）注意力相比，它牺牲了来自完整成对注意力的全局上下文建模能力。

Gated DeltaNet 在某种程度上仍然可以捕获上下文，但它必须通过记忆（*S*）瓶颈。该记忆是固定大小的，因此更高效，但它将过去的上下文压缩成单个隐藏状态，类似于 RNN。

这就是为什么 Qwen3-Next 和 Kimi Linear 架构不将所有注意力层替换为 DeltaNet 层，而是使用前面提到的 3:1 比例。

&nbsp;
## DeltaNet 内存节省

在上一节中，我们讨论了 DeltaNet 相对于全注意力在计算复杂度方面的优势（线性而非二次方）。

除了线性计算复杂度外，DeltaNet 的另一个大优势是内存节省，因为 DeltaNet 模块不会增长 KV 缓存。（有关 KV 缓存的更多信息，请参见 [../03_kv-cache](../03_kv-cache)）。相反，如前所述，它们保持固定大小的循环状态，因此内存随上下文长度保持恒定。

对于常规多头注意力（MHA）层，我们可以如下计算 KV 缓存大小：

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

（2 倍乘数是因为我们在缓存中存储了键和值。）

对于上面实现的简化 DeltaNet 版本，我们有：

```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

请注意，`KV_cache_DeltaNet` 内存大小没有上下文长度（`n_tokens`）依赖。此外，我们只存储记忆状态 S 而不是单独的键和值，因此 `2 × bytes` 变为仅 `bytes`。但请注意，我们现在这里有二次方的 `d_head × d_head`。这来自状态：

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

但这通常不需要担心，因为头维度通常相对较小。例如，在 Qwen3-Next 中它是 128。

带有卷积混合的完整版本更复杂，包括卷积核大小等，但上面的公式应该说明 Gated DeltaNet 背后的主要趋势和动机。

我们可以通过以下辅助脚本可视化不同上下文长度的内存估计和节省：

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

请注意，上面将 `head_dim` 计算为 `emb_dim / n_heads`。即 2048 / 16 = 128。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="Gated DeltaNet 缩放" width=500px>
