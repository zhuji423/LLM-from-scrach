"""
模型内存占用计算器

演示不同场景下的内存占用
"""

import torch


def bytes_to_human_readable(bytes_val):
    """将字节数转换为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def calculate_model_size(num_params, dtype='float32'):
    """计算模型大小"""
    dtype_sizes = {
        'float64': 8,
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1,
    }

    bytes_per_param = dtype_sizes.get(dtype, 4)
    total_bytes = num_params * bytes_per_param

    return total_bytes


def inference_memory(num_params, dtype='float32', batch_size=1, seq_len=512, emb_dim=768):
    """计算推理时的内存占用"""
    print("=" * 70)
    print(f"推理时内存占用 (dtype={dtype})")
    print("=" * 70)

    # 模型参数
    model_bytes = calculate_model_size(num_params, dtype)

    # 激活值（一个 batch 的输入输出）
    # 简化估计：batch_size × seq_len × emb_dim
    activation_elements = batch_size * seq_len * emb_dim
    activation_bytes = calculate_model_size(activation_elements, dtype)

    print(f"\n1. 模型参数:")
    print(f"   数量: {num_params:,}")
    print(f"   大小: {bytes_to_human_readable(model_bytes)}")

    print(f"\n2. 激活值 (单 batch):")
    print(f"   batch_size={batch_size}, seq_len={seq_len}, emb_dim={emb_dim}")
    print(f"   大小: {bytes_to_human_readable(activation_bytes)}")

    total_bytes = model_bytes + activation_bytes
    print(f"\n总内存占用: {bytes_to_human_readable(total_bytes)}")

    return total_bytes


def training_memory(num_params, dtype='float32', batch_size=8, seq_len=512, emb_dim=768, optimizer='adam'):
    """计算训练时的内存占用"""
    print("\n" + "=" * 70)
    print(f"训练时内存占用 (dtype={dtype}, optimizer={optimizer})")
    print("=" * 70)

    # 1. 模型参数
    model_bytes = calculate_model_size(num_params, dtype)

    # 2. 梯度
    gradient_bytes = calculate_model_size(num_params, dtype)

    # 3. 优化器状态
    if optimizer == 'sgd':
        # SGD 只需存储 momentum (1x params)
        optimizer_bytes = calculate_model_size(num_params, dtype)
    elif optimizer == 'adam':
        # Adam 需要存储 momentum + variance (2x params)
        optimizer_bytes = calculate_model_size(num_params * 2, dtype)
    else:
        optimizer_bytes = 0

    # 4. 激活值（训练时需要保存更多）
    # 粗略估计：每层都要保存激活值
    num_layers = 12  # 假设 12 层
    activation_elements = batch_size * seq_len * emb_dim * num_layers * 2  # × 2 因为前向+反向
    activation_bytes = calculate_model_size(activation_elements, dtype)

    print(f"\n1. 模型参数:")
    print(f"   {bytes_to_human_readable(model_bytes)}")

    print(f"\n2. 梯度:")
    print(f"   {bytes_to_human_readable(gradient_bytes)}")

    print(f"\n3. 优化器状态 ({optimizer}):")
    print(f"   {bytes_to_human_readable(optimizer_bytes)}")

    print(f"\n4. 激活值 (batch_size={batch_size}):")
    print(f"   {bytes_to_human_readable(activation_bytes)}")

    total_bytes = model_bytes + gradient_bytes + optimizer_bytes + activation_bytes
    print(f"\n总内存占用: {bytes_to_human_readable(total_bytes)}")

    return total_bytes


def compare_dtypes(num_params):
    """对比不同数据类型的模型大小"""
    print("\n" + "=" * 70)
    print("不同数据类型的模型大小对比")
    print("=" * 70)

    dtypes = ['float64', 'float32', 'float16', 'bfloat16', 'int8']

    print(f"\n参数数量: {num_params:,}\n")
    print(f"{'数据类型':<12} {'字节/参数':<10} {'模型大小':<15} {'压缩比':<10}")
    print("-" * 50)

    float32_size = calculate_model_size(num_params, 'float32')

    for dtype in dtypes:
        size_bytes = calculate_model_size(num_params, dtype)
        size_readable = bytes_to_human_readable(size_bytes)
        compression = float32_size / size_bytes

        dtype_sizes = {'float64': 8, 'float32': 4, 'float16': 2, 'bfloat16': 2, 'int8': 1}
        bytes_per_param = dtype_sizes[dtype]

        print(f"{dtype:<12} {bytes_per_param:<10} {size_readable:<15} {compression:.2f}x")


def practical_examples():
    """实际模型的例子"""
    print("\n" + "=" * 70)
    print("实际模型大小示例")
    print("=" * 70)

    models = [
        ("GPT-2 Small", 124_000_000),
        ("GPT-2 Medium", 355_000_000),
        ("GPT-2 Large", 774_000_000),
        ("GPT-2 XL", 1_500_000_000),
        ("LLaMA-7B", 7_000_000_000),
        ("LLaMA-13B", 13_000_000_000),
        ("LLaMA-65B", 65_000_000_000),
        ("GPT-3", 175_000_000_000),
    ]

    print(f"\n{'模型':<15} {'参数量':<15} {'float32':<12} {'float16':<12} {'int8':<12}")
    print("-" * 70)

    for model_name, num_params in models:
        f32 = bytes_to_human_readable(calculate_model_size(num_params, 'float32'))
        f16 = bytes_to_human_readable(calculate_model_size(num_params, 'float16'))
        i8 = bytes_to_human_readable(calculate_model_size(num_params, 'int8'))

        params_str = f"{num_params/1e9:.1f}B" if num_params >= 1e9 else f"{num_params/1e6:.0f}M"

        print(f"{model_name:<15} {params_str:<15} {f32:<12} {f16:<12} {i8:<12}")


def memory_optimization_tips():
    """内存优化技巧"""
    print("\n" + "=" * 70)
    print("内存优化技巧")
    print("=" * 70)

    print("""
1. 混合精度训练 (Mixed Precision)
   - 使用 float16/bfloat16 进行前向和反向传播
   - 使用 float32 存储主权重
   - 内存节省: ~50%
   - 代码: torch.cuda.amp.autocast()

2. 梯度累积 (Gradient Accumulation)
   - 减小 batch_size，多个 step 累积梯度
   - 内存节省: 正比于累积步数
   - 代码: 每 N 步才调用 optimizer.step()

3. 梯度检查点 (Gradient Checkpointing)
   - 不保存所有激活值，需要时重新计算
   - 内存节省: ~40-60%（激活值部分）
   - 代价: 增加 ~30% 计算时间
   - 代码: torch.utils.checkpoint.checkpoint()

4. 量化 (Quantization)
   - 推理时使用 int8/int4
   - 内存节省: 75-87.5%
   - 代价: 轻微精度损失
   - 工具: bitsandbytes, GPTQ, AWQ

5. LoRA 微调
   - 只训练低秩适配器（<1% 参数）
   - 内存节省: ~90%（训练时）
   - 精度: 与全参数微调相当

6. 模型并行 (Model Parallelism)
   - 将模型切分到多个 GPU
   - 适用于超大模型
   - 工具: DeepSpeed, Megatron-LM

7. CPU Offloading
   - 将不常用的参数/梯度移到 CPU
   - 适合 GPU 显存不足时
   - 代价: 数据传输开销
    """)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" 模型内存占用详解 ".center(70, "="))
    print("=" * 70)

    # GPT-2 Small 参数量
    gpt2_params = 163_009_536

    # 1. 推理内存
    inference_memory(gpt2_params, dtype='float32', batch_size=1)

    # 2. 训练内存
    training_memory(gpt2_params, dtype='float32', batch_size=8, optimizer='adam')

    # 3. 数据类型对比
    compare_dtypes(gpt2_params)

    # 4. 实际模型示例
    practical_examples()

    # 5. 优化技巧
    memory_optimization_tips()

    print("\n" + "=" * 70)
    print("核心要点:")
    print("=" * 70)
    print("""
1. 基本换算:
   内存 (MB) = 参数数量 × 字节数 / 1,048,576

2. 数据类型:
   float32 (4 bytes) - 默认，训练用
   float16 (2 bytes) - 混合精度，节省 50%
   int8 (1 byte)     - 量化推理，节省 75%

3. 训练 vs 推理:
   推理: 仅需模型参数 (~621 MB)
   训练: 需要参数 + 梯度 + 优化器 + 激活值 (~2.4 GB+)

4. 优化策略:
   - 混合精度: 最常用，效果好
   - 梯度累积: 减小 batch_size
   - 梯度检查点: 时间换空间
   - 量化: 推理加速

5. 实际考虑:
   - GPU 显存通常是瓶颈
   - 训练内存是推理的 3-4 倍
   - 激活值占用与 batch_size 成正比
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
