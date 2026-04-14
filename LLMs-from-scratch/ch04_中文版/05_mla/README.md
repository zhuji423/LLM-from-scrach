# 多头潜在注意力（Multi-Head Latent Attention, MLA）

本补充材料说明了使用多头潜在注意力（MLA）相对于常规多头注意力（MHA）时的内存节省。

&nbsp;
## 简介

在 [../04_gqa](../04_gqa) 中，我们讨论了分组查询注意力（GQA）作为 MHA 的计算效率解决方案。消融研究（例如[原始 GQA 论文](https://arxiv.org/abs/2305.13245)和 [Llama 2 论文](https://arxiv.org/abs/2307.09288)中的研究）表明，在 LLM 建模性能方面，它的表现与标准 MHA 相当。

现在，[DeepSeek V2、V3 和 R1](https://arxiv.org/abs/2412.19437) 中使用的多头潜在注意力（MLA）提供了一种不同的内存节省策略，它与 KV 缓存特别搭配。MLA 不像 GQA 那样共享键和值头，而是将键和值张量压缩到低维空间后再存储到 KV 缓存中。

在推理时，这些压缩的张量被投影回原始大小后再使用，如下图所示。这增加了一次额外的矩阵乘法，但减少了内存使用。

&nbsp;

![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

&nbsp;

（附带说明，查询也会被压缩，但仅在训练期间，推理时不压缩。）

顺便提一下，如前所述，MLA 并不是 DeepSeek V3 首创的，其前身 [DeepSeek V2](https://arxiv.org/abs/2405.04434) 也使用（甚至引入了）它。此外，V2 论文包含一些有趣的消融研究，可能解释了为什么 DeepSeek 团队选择 MLA 而不是 GQA（见下图）。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" />

&nbsp;

如上图所示，GQA 的表现似乎比 MHA 差，而 MLA 提供了比 MHA 更好的建模性能，这可能是 DeepSeek 团队选择 MLA 而非 GQA 的原因。（如果能同时看到 MLA 和 GQA 之间"每个 Token 的 KV 缓存"节省对比就更有趣了！）

总结本节，在我们进入下一个架构组件之前，MLA 是一种减少 KV 缓存内存使用的巧妙技巧，甚至在建模性能上略优于 MHA。

&nbsp;
## MLA 内存节省

内存节省主要体现在 KV 存储上。我们可以使用以下公式计算 KV 存储大小：

bytes ≈ batch_size × seqlen × n_layers × latent_dim × bytes_per_elem

相比之下，MHA KV 缓存内存计算如下：

bytes ≈ batch_size × seqlen × n_layers × embed_dim × 2 (K,V) × bytes_per_elem

这意味着，在 MLA 中，我们将 "embed_dim × 2 (K,V)" 减少为 "latent_dim"，因为我们只存储压缩的潜在表示，而不是完整的键和值向量，如前面的图所示。

您可以使用此文件夹中的 [memory_estimator_mla.py](memory_estimator_mla.py) 脚本将此应用于不同的模型配置，以查看使用 MLA 相对于 MHA 可以节省多少内存：

```bash
➜ uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== 配置 ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 字节/元素)
head_dim         : 86
GQA n_kv_heads   : 6

==== 所有层的 KV 缓存总计 ====
MHA 总 KV 缓存    : 3.25 GB
GQA 总 KV 缓存    : 0.81 GB
MLA 总 KV 缓存    : 0.81 GB
比率 (MHA / GQA)  : 4.00x
节省 (GQA vs MHA) : 75.00%
比率 (MHA / MLA)  : 4.03x
节省 (MLA vs MHA) : 75.19%
```

请注意，上面的压缩（`--emb_dim 2048 -> latent_dim 1024`）实现了与 GQA 类似的节省。在实践中，压缩是一个需要仔细研究的超参数，因为选择过小的 `latent_dim` 可能对建模性能产生负面影响（类似于在 GQA 中选择过多的 `n_kv_groups`）。

下图进一步显示了使用 MLA 相对于 MHA 的节省，针对不同的 `latent_dim` 值作为上下文长度的函数：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" />

&nbsp;

您可以通过 `uv run plot_memory_estimates_mla.py` 重现该图。



&nbsp;
## MLA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_mla.py](gpt_with_kv_mla.py) 脚本提供了在 GPT 模型实现的上下文中比较 MHA 和 MLA 内存使用的实践示例。

这里的 MLA 代码灵感来自 [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla) 的实现。

请注意，MLA 可以与 [GQA](../04_gqa) 结合使用，但为了简单起见，这里没有这样做。（目前我也不知道有哪个知名 LLM 这样做。）

另外请注意，该模型未经训练，因此会生成无意义的文本。但是，您可以将其用作第 5-7 章中标准 GPT 模型的即插即用替代品并对其进行训练。

最后，此实现使用了[另一个补充章节](../03_kv-cache)中解释的 KV 缓存，因此内存节省更加明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

时间: 453.81 秒
72 tokens/秒
最大分配内存: 1.54 GB
```

```bash
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× 压缩

...

时间: 487.21 秒
67 tokens/秒
最大分配内存: 0.68 GB
```

我们没有看到与上图一样大的节省的原因有两个：

1. 我使用了较小的配置，以便模型在合理的时间内完成生成。
2. 更重要的是，我们在这里查看的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（但这是另一个分析的主题）。
