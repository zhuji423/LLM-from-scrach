# 分组查询注意力（Grouped-Query Attention, GQA）

本补充材料说明了使用分组查询注意力（GQA）相对于常规多头注意力（MHA）时的内存节省。

&nbsp;
## 简介

近年来，分组查询注意力（GQA）已成为多头注意力（MHA）的新标准替代方案，它在计算和参数效率方面更优。请注意，它并不是新技术，可以追溯到 2023 年的论文 [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)。甚至早期的 Llama 2 系列中的较大变体也使用了它。

以下是 GQA 的简要总结。与 MHA 不同（其中每个头都有自己的键和值集），为了减少内存使用，GQA 将多个头分组以共享相同的键和值投影。

例如，如下图所示，如果有 3 个键值组和 6 个注意力头，那么头 1 和头 2 共享一组键和值，而头 3 和头 4、以及头 5 和头 6 分别共享另一组。

&nbsp;

![GQA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)

&nbsp;

这种键和值的共享减少了键和值计算的总数，从而降低了内存使用并提高了效率。

总而言之，GQA 背后的核心思想是通过在多个查询头之间共享键和值头来减少键和值头的数量。这样做 (1) 降低了模型的参数数量，(2) 减少了推理期间键和值张量的内存带宽使用，因为需要从 KV 缓存中存储和检索的键和值更少。

虽然 GQA 主要是 MHA 的计算效率解决方案，但消融研究（例如[原始 GQA 论文](https://arxiv.org/abs/2305.13245)和 [Llama 2 论文](https://arxiv.org/abs/2307.09288)中的研究）表明，在 LLM 建模性能方面，它的表现与标准 MHA 相当。

然而，这假设键值组的数量是经过仔细选择的。在极端情况下，所有注意力头共享单个键值组（称为多查询注意力），内存使用会更显著地减少，但建模性能可能会受到影响。（而在另一个极端，如果我们将键值组的数量设置为等于查询头的数量，我们就回到了标准的多头注意力。）

&nbsp;
## GQA 内存节省

内存节省主要体现在 KV 存储上。我们可以使用以下公式计算 KV 存储大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

您可以使用此文件夹中的 [memory_estimator_gqa.py](memory_estimator_gqa.py) 脚本将此应用于不同的模型配置，以查看使用 GQA 相对于 MHA 可以节省多少内存：

```bash
➜ uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== 配置 ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 字节/元素)
head_dim         : 128
GQA n_kv_heads   : 8

==== 所有层的 KV 缓存总计 ====
MHA 总 KV 缓存    : 17.18 GB
GQA 总 KV 缓存    : 4.29 GB
比率 (MHA / GQA)  : 4.00x
节省 (GQA vs MHA) : 75.00%
```

下图进一步显示了使用 GQA 相对于 MHA 的节省，针对不同的键值组大小作为上下文长度的函数：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="GQA" width="500px" />

&nbsp;

您可以通过 `uv run plot_memory_estimates_gqa.py` 重现该图。

&nbsp;
## GQA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) 脚本提供了在 GPT 模型实现的上下文中比较 MHA 和 GQA 内存使用的实践示例。

请注意，GQA 也用于 [Llama 3](../../ch05/07_gpt_to_llama)、[Gemma 3](../../ch05/12_gemma3) 和 [Qwen3](../../ch05/11_qwen3) 补充材料中。然而，为了简单起见，此文件夹中的代码脚本修改了 GPT 架构，传统上它不使用 GQA。

请注意，该模型未经训练，因此会生成无意义的文本。但是，您可以将其用作第 5-7 章中标准 GPT 模型的即插即用替代品并对其进行训练。

此外，此实现使用了[另一个补充章节](../03_kv-cache)中解释的 KV 缓存，因此内存节省更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

时间: 453.81 秒
72 tokens/秒
最大分配内存: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

时间: 516.33 秒
63 tokens/秒
最大分配内存: 0.63 GB
```

我们没有看到与上图一样大的节省的原因有两个：

1. 我使用了较小的配置，以便模型在合理的时间内完成生成。
2. 更重要的是，我们在这里查看的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（但这是另一个分析的主题）。
