# 专家混合（Mixture of Experts, MoE）

本补充材料说明了使用专家混合（MoE）层代替常规前馈（FFN）层时的（每个 token 的）内存节省。



&nbsp;
## 简介

MoE 的核心思想是将 transformer 块中的每个前馈模块替换为多个专家层，其中每个专家层也是一个前馈模块。这意味着我们将单个前馈块替换为多个前馈块，如下图所示。



&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="SWA" width="800px" />

transformer 块内部的前馈块（如上图中的深灰色块所示）通常包含模型总参数的很大一部分。（请注意，transformer 块，以及其中的前馈块，在 LLM 中被重复多次；在 DeepSeek-V3 的情况下，重复了 61 次。）

因此，将*单个*前馈块替换为*多个*前馈块（如 MoE 设置中所做的那样）大幅增加了模型的总参数量。然而，关键技巧是，我们不会对每个 token 都使用（"激活"）所有专家。相反，路由器为每个 token 只选择一小部分专家。

因为每次只有少数专家处于活跃状态，MoE 模块通常被称为*稀疏*的，与始终使用完整参数集的*稠密*模块形成对比。然而，通过 MoE 获得的大量总参数增加了 LLM 的容量，这意味着它可以在训练期间吸收更多知识。稀疏性保持了推理效率，因为我们不会同时使用所有参数。

例如，DeepSeek-V3 每个 MoE 模块有 256 个专家，总共有 6710 亿个参数。但在推理时，一次只有 9 个专家处于活跃状态（1 个共享专家加上路由器选择的 8 个）。这意味着每个 token 推理步骤只使用 370 亿个参数，而不是全部 6710 亿个。

DeepSeek-V3 的 MoE 设计的一个显著特点是使用共享专家。这是一个对每个 token 始终处于活跃状态的专家。这个想法并不新，已经在 [2022 年的 DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 和 [2024 年的 DeepSeek MoE](https://arxiv.org/abs/2401.06066) 论文中引入。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE 共享专家" width="500px" />

（来自 [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) 论文的注释图。）

&nbsp;

拥有共享专家的好处首先在 [DeepSpeed-MoE 论文](https://arxiv.org/abs/2201.05596)中被注意到，他们发现与没有共享专家相比，它提升了整体建模性能。这可能是因为常见或重复的模式不需要被多个独立专家学习，这使得它们有更多空间学习更专业的模式。

&nbsp;
## 专家混合（MoE）内存节省

MoE 模型中的内存节省主要来自减少的激活存储和计算。在常规（稠密）前馈层（FFN）中，每个 token 都激活完整的中间维度。

相比之下，MoE 层将每个 token 只路由到少数专家子集（例如，`num_experts` 中的 `top_k` 个）。

使用 MoE 层时，每个 token 只有 `top_k` 个专家处于活跃状态，因此有效内存（和计算）相对于相同总容量的稠密 FFN 大致按 `top_k / num_experts` 的因子缩放。

您可以使用此文件夹中的 [memory_estimator_moe.py](memory_estimator_moe.py) 脚本将此应用于不同的模型配置，以查看使用 MoE 相对于 FFN 可以节省多少内存（请注意，这是针对单个 transformer 块的，要获得总节省量，请乘以模型中 transformer 块的数量）：

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense
==== 配置 ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 字节/元素)
match_dense            : True

==== 模型权重（参数）====
稠密 FFN 参数           : 308,281,344 (0.62 GB)
每个专家参数            : 38,535,168 (0.08 GB)
路由器参数              : 57,344 (0.00 GB)
MoE 总参数             : 308,338,688 (0.62 GB)
MoE 每个Token活跃参数   : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

因此，基于上述结果，我们可以看到，如果我们有一个输入/输出维度（`emb_dim`）为 7,168、中间大小（`hidden_dim`）为 14,336 的 FFN，该层约有 3.08 亿参数，并且所有这些参数在前向传播中都处于活跃状态。

现在，如果我们使用一个总参数数量大致相同（约 3.08 亿）的 MoE 层，有 8 个专家，其中 2 个专家处于活跃状态，那么每次前向传播中只有约 7700 万参数处于活跃状态。

此外，在专家总数恒定的情况下，我们拥有的专家越多，活跃参数数量就越少，"节省"就越大：

&nbsp;

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="SWA" width="500px" />



&nbsp;

您可以通过以下命令重现该图：

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


&nbsp;
## MoE 代码示例

此文件夹中的 [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) 和 [gpt_with_kv_moe.py](gpt_with_kv_moe.py) 脚本提供了在 GPT 模型实现的上下文中比较常规 FFN 和 MoE 内存使用的实践示例。请注意，两个脚本都使用 [SwiGLU](https://arxiv.org/abs/2002.05202) 前馈模块，如本页第一个图所示（GPT-2 传统上使用 GELU）。

**注意：该模型未经训练，因此会生成无意义的文本。您可以在补充材料 [../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb) 中找到训练好的 MoE。**



首先，让我们使用常规 FFN 运行模型：


```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
平均 FFN 时间/调用: 0.759 ms
平均 FFN 内存增量/调用: 0.19 MB (最大 0.75 MB)
...
时间: 25.13 秒
40 tokens/秒
最大分配内存: 11.47 GB
```

为了与 MoE 进行公平比较，我们必须缩小专家大小。例如，如果我们使用 32 个专家，我们需要设置 `--hidden_dim 32768/32`：


```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
平均 MoE 前馈时间/调用: 1.555 ms
平均 MoE 前馈内存增量/调用: 0.04 MB (最大 0.11 MB)
...
时间: 35.11 秒
29 tokens/秒
最大分配内存: 11.48 GB
```

我们可以看到，稠密前馈层处理一个 token 约需 0.76 ms，使用约 0.19 MB 的激活（峰值接近 0.75 MB），稀疏 MoE 层仅保持约 0.04 MB 的内存（峰值 0.11 MB）。然而，这是以大约两倍的计算时间为代价的。（存在额外的路由开销，而且我的实现可能也不是最高效的。）

总体生成在两种情况下仍然峰值约 11.5 GB 的 GPU 内存，因为两个版本加载相同数量的权重参数并具有相同的 KV 缓存大小，这些在这里占主导地位。

无论如何，我们可以看到这里的权衡：MoE 将 FFN 内存减少约 4-5 倍，同时大约使前馈计算时间翻倍。

请注意，如果我们一次处理更多 token，例如使用大于 1 的批量大小（这里为了代码简单性没有使用批量），节省会更加明显。



