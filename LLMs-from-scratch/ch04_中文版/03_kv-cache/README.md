# 补充内容：KV 缓存



**本文件夹实现了为 GPT 模型添加 KV 缓存的功能。**

&nbsp;
## 概述

简而言之，KV 缓存存储中间的键（K）和值（V）计算结果以便在推理时复用，从而在生成响应时实现显著的速度提升。缺点是它会增加代码复杂性，增加内存使用，且无法在训练时使用。然而，在部署 LLM 时，推理速度的提升往往非常值得在代码复杂性和内存方面做出权衡。

&nbsp;
## 工作原理

想象 LLM 正在生成一些文本。具体来说，假设 LLM 收到以下提示："Time flies"。

下图展示了底层注意力分数计算的一个片段，使用了第 3 章的修改图形，并突出显示了键和值向量：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

现在，正如我们在第 2 章和第 4 章中学到的，LLM 每次生成一个词（或 token）。假设 LLM 生成了单词 "fast"，那么下一轮的提示就变成了 "Time flies fast"。这在下图中得到了说明：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

正如我们所见，通过比较前面两个图，前两个 token 的键和值向量是完全相同的，在每一轮的下一个 token 文本生成中重新计算它们是一种浪费。

因此，KV 缓存的思想是实现一个缓存机制，存储先前生成的键和值向量以供复用，这有助于我们避免不必要的重复计算。

&nbsp;

## KV 缓存实现

实现 KV 缓存有很多方法，主要思想是在每个生成步骤中只计算新生成 token 的键和值张量。

我选择了一个简单的实现，强调代码可读性。我认为直接浏览代码更改是了解其实现方式的最简单方法。

本文件夹中有两个文件：

1. [`gpt_ch04.py`](gpt_ch04.py)：从第 3 章和第 4 章获取的独立代码，用于实现 LLM 并运行简单的文本生成函数
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：与上述相同，但添加了实现 KV 缓存所需的更改。

您可以：

a. 打开 [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) 文件，查找标记新更改的 `# NEW` 部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 通过您选择的文件对比工具查看这两个代码文件以比较更改：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

为了总结实现细节，下面是一个简短的演练。

&nbsp;

### 1. 注册缓存缓冲区

在 `MultiHeadAttention` 构造函数内部，我们添加两个缓冲区 `cache_k` 和 `cache_v`，它们将在各步骤之间保存连接的键和值：

```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)
```

&nbsp;

### 2. 带有 `use_cache` 标志的前向传播

接下来，我们扩展 `MultiHeadAttention` 类的 `forward` 方法以接受 `use_cache` 参数。在将新的 token 块投影到 `keys_new`、`values_new` 和 `queries` 后，我们要么初始化 kv 缓存，要么附加到我们的缓存：

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # 形状: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new

    # ...

    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;


### 3. 清除缓存

在生成文本时，在独立序列之间（例如在文本生成调用之间），我们必须重置两个缓冲区，因此我们还向 `MultiHeadAttention` 类添加了一个缓存重置方法：

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. 在完整模型中传播 `use_cache`

对 `MultiHeadAttention` 类进行更改后，我们现在修改 `GPTModel` 类。首先，我们在构造函数中添加 token 索引的位置跟踪：

```python
self.current_pos = 0
```

然后，我们用显式循环替换单行块调用，通过每个 transformer 块传递 `use_cache`：

```python
def forward(self, in_idx, use_cache=False):
    # ...

    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )

    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上述更改还需要对 `TransformerBlock` 类进行小的修改以接受 `use_cache` 参数：
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，我们向 `GPTModel` 添加模型级重置，以便一次性清除所有块缓存：

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. 在生成中使用缓存

对 `GPTModel`、`TransformerBlock` 和 `MultiHeadAttention` 进行更改后，最后，下面是我们如何在简单的文本生成函数中使用 KV 缓存：

```python
def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # 用完整提示初始化缓存
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) 选择具有最高对数概率的 token（贪婪采样）
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) 将其附加到运行序列
                idx = torch.cat([idx, next_idx], dim=1)
                # c) 仅向模型提供新 token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

请注意，我们在 c) 中仅通过 `logits = model(next_idx, use_cache=True)` 向模型提供新 token。没有缓存时，我们向模型提供整个输入 `logits = model(idx[:, -ctx_len:], use_cache=False)`，因为它没有存储的键和值可供复用。

&nbsp;

## 简单的性能比较

在概念层面介绍了 KV 缓存之后，最大的问题是它在一个小示例上的实际性能如何。为了试用该实现，我们可以将上述两个代码文件作为 Python 脚本运行，这将运行小型 124M 参数 LLM 来生成 200 个新 token（给定一个 4-token 提示 "Hello, I am" 作为起始）：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在带有 M4 芯片（CPU）的 Mac Mini 上，结果如下：

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

因此，正如我们所见，使用小型 124M 参数模型和较短的 200 token 序列长度，我们已经获得了约 5 倍的速度提升。（请注意，此实现针对代码可读性进行了优化，而不是针对 CUDA 或 MPS 运行时速度进行了优化，后者需要预分配张量而不是重新实例化和连接它们。）

**注意：** 在这两种情况下，模型都会生成"乱码"，即看起来像这样的文本：

> 输出文本: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

这是因为我们还没有训练模型。下一章训练模型，您可以在训练好的模型上使用 KV 缓存（但是，KV 缓存仅用于推理）来生成连贯的文本。在这里，我们使用未经训练的模型来保持代码简单。

不过，更重要的是，`gpt_ch04.py` 和 `gpt_with_kv_cache.py` 实现产生的文本完全相同。这告诉我们 KV 缓存实现正确——很容易犯索引错误，导致结果不一致。


&nbsp;

## KV 缓存的优势和劣势

随着序列长度的增加，KV 缓存的优势和劣势在以下方面变得更加明显：

- [优势] **计算效率提高**：没有缓存时，步骤 *t* 的注意力必须将新查询与 *t* 个先前的键进行比较，因此累积工作量呈二次方增长，O(n²)。使用缓存，每个键和值计算一次然后复用，将每步的总复杂度降低到线性，O(n)。

- [劣势] **内存使用线性增加**：每个新 token 都会附加到 KV 缓存。对于长序列和较大的 LLM，累积的 KV 缓存会变得更大，这可能会消耗大量甚至令人望而却步的（GPU）内存。作为解决方法，我们可以截断 KV 缓存，但这会增加更多复杂性（但同样，在部署 LLM 时这可能非常值得。）



&nbsp;
## 优化 KV 缓存实现

虽然我上面的 KV 缓存概念实现有助于清晰度，主要面向代码可读性和教育目的，但在实际场景中部署它（尤其是使用更大的模型和更长的序列长度时）需要更仔细的优化。

&nbsp;
### 扩展缓存时的常见陷阱

- **内存碎片化和重复分配**：如前所示，通过 `torch.cat` 连续连接张量会由于频繁的内存分配和重新分配而导致性能瓶颈。

- **内存使用的线性增长**：如果没有适当的处理，KV 缓存大小对于非常长的序列来说变得不切实际。

&nbsp;
#### 技巧 1：预分配内存

与其重复连接张量，我们可以根据预期的最大序列长度预分配一个足够大的张量。这确保了一致的内存使用并减少了开销。在伪代码中，这可能看起来如下：

```python
# 键和值的预分配示例
max_seq_len = 1024  # 预期的最大序列长度
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

在推理期间，我们可以简单地写入这些预分配张量的切片。

&nbsp;
#### 技巧 2：通过滑动窗口截断缓存

为了避免耗尽 GPU 内存，我们可以实现带有动态截断的滑动窗口方法。通过滑动窗口，我们仅在缓存中保留最后 `window_size` 个 token：


```python
# 滑动窗口缓存实现
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### 实践中的优化

您可以在 [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) 文件中找到这些优化。


在带有 M4 芯片（CPU）的 Mac Mini 上，使用 200 token 生成和等于上下文长度的窗口大小（以保证相同结果），代码运行时间比较如下：

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

不幸的是，在 CUDA 设备上速度优势消失了，因为这是一个微型模型，设备传输和通信超过了 KV 缓存对这个小模型的好处。


&nbsp;
## 其他资源

1. [Qwen3 从零开始的 KV 缓存基准测试](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 从零开始的 KV 缓存基准测试](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [理解和从零编码 LLM 中的 KV 缓存](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- 这个 README 的更详细说明
