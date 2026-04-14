# 滑动窗口注意力（Sliding Window Attention, SWA）

本补充材料说明了使用滑动窗口注意力（SWA）相对于常规多头注意力（MHA）时的内存节省。



&nbsp;
## 简介

什么是滑动窗口注意力（SWA）？如果我们把常规自注意力看作一种*全局*注意力机制（因为每个序列元素都可以访问其他所有序列元素），那么我们可以把 SWA 看作*局部*注意力，因为在这里我们限制了当前查询位置周围的上下文大小。如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="滑动窗口注意力" width="500px" />

如上图所示，每个 token 不再关注所有先前的 token，而是只关注其位置周围固定大小的局部窗口。这种局部化注意力大幅降低了 KV 缓存的大小。

在本介绍的剩余部分，我们将讨论 [Gemma 3](https://arxiv.org/abs/2503.19786) 上下文中的 SWA，Gemma 3 在 [../../ch05/12_gemma3](../../ch05/12_gemma3) 中从零实现。

滑动窗口注意力最初在 [2020 年的 LongFormer 论文](https://arxiv.org/abs/2004.05150)中引入，但我们关注 Google 的 Gemma 模型的原因是，它们是非常好的开放权重模型，表明滑动窗口注意力确实是近期高性能模型中的可行方法。

[Gemma 2](https://arxiv.org/abs/2408.00118) 使用了一种混合方法，以 1:1 的比例组合了局部（滑动窗口）和全局注意力层。每个 token 可以关注 4k 个 token 的上下文窗口。采用 1:1 混合的原因是它在效率和全局上下文建模之间取得了平衡，因为仅使用局部注意力的 LLM 可能过于受限。

[Gemma 3](https://arxiv.org/abs/2503.19786) 则进一步向效率方向推进设计。它以 5:1 的比例使用滑动窗口和全注意力层，这意味着每五个局部注意力层就有一个全局层。此外，滑动窗口大小从 Gemma 2 的 4096 个 token 减少到 Gemma 3 的 1024 个 token。

有趣的是，Gemma 3 技术报告中的消融研究表明，这些变化对整体模型质量只有很小的影响。换句话说，通过滑动窗口注意力实现的显著内存和计算节省，只带来极小的建模性能损失。



&nbsp;
## 滑动窗口注意力（SWA）内存节省

内存节省主要体现在 KV 存储上。我们可以使用以下公式计算 KV 存储大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

使用 SWA 时，我们将上面的序列长度（seqlen）替换为窗口大小 W。因此，使用滑动窗口注意力时，我们将 KV 缓存大小减少了 "W / seqlen" 的因子。（请注意，为简单起见，这里假设每个层都使用滑动窗口注意力。）

您可以使用此文件夹中的 [memory_estimator_swa.py](memory_estimator_swa.py) 脚本将此应用于不同的模型配置，以查看使用 SWA 相对于 MHA 可以节省多少内存：

```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== 配置 ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 字节/元素)
head_dim               : 128
GQA n_kv_heads         : 8
有效 SWA 窗口 W        : 1024
层比例 (SWA:Full)      : 5:1
分布层                 : 27 SWA, 5 FULL

==== 所有层的 KV 缓存总计 ====
MHA KV 总计            : 17.18 GB
GQA KV 总计            : 4.29 GB
MHA + SWA (比例: 5:1)  : 3.14 GB
MHA + GQA (比例: 5:1)  : 0.78 GB
```

请注意，Gemma 3 将 SWA 与 GQA 结合使用。

下图进一步显示了使用 SWA 相对于 MHA 的节省，针对不同的上下文长度：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="800px" />

&nbsp;

您可以通过以下命令重现该图：

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


&nbsp;
## SWA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_swa.py](gpt_with_kv_swa.py) 脚本提供了在 GPT 模型实现的上下文中比较 MHA 和 SWA 内存使用的实践示例。

请注意，SWA 可以与 MLA 和 GQA 结合使用（如前所述），但为了简单起见，这里没有这样做。

请注意，该模型未经训练，因此会生成无意义的文本。但是，您可以将其用作第 5-7 章中标准 GPT 模型的即插即用替代品并对其进行训练。

此外，此实现使用了[另一个补充章节](../03_kv-cache)中解释的 KV 缓存，因此内存节省更加明显。

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
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # 类似 Gemma 3

...

时间: 514.38 秒
63 tokens/秒
最大分配内存: 0.63 GB
```

我们没有看到与上图一样大的节省的原因有两个：

1. 我使用了较小的配置，以便模型在合理的时间内完成生成。
2. 更重要的是，我们在这里查看的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（但这是另一个分析的主题）。
