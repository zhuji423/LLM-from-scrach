# LLM 学习笔记索引

> 学习 LLMs-from-scratch 过程中的笔记整理

---

## 目录结构

```
notes/
├── 00_Overview_概览/                     (3 文件)
├── 01_Fundamentals_基础原理/              (9 文件) ⭐ +3
├── 02_DataProcessing_数据处理/            (3 文件)
├── 03_Architecture_模型架构/              (10 文件) ⭐ +2
├── 04_Training_训练机制/                  (5 文件) ⭐ +1
├── 05_ScalingLaws_扩展定律/               (3 文件)
└── 06_MultimodalFrontiers_多模态前沿论文/ (6 文件)
```

---

## 📋 00_Overview_概览

| 文件 | 类型 | 内容 |
|------|------|------|
| [00_QA_Summary_学习问答总结](00_Overview_概览/00_QA_Summary_学习问答总结.md) | 📝 | 学习过程中的问答汇总（21个问题） |
| [01_Core_Concepts_核心概念图解](00_Overview_概览/01_Core_Concepts_核心概念图解.md) | 📝 | 维度流、注意力机制、Scaling Law 图解 |
| [02_Understanding_vs_Generation_理解与生成](00_Overview_概览/02_Understanding_vs_Generation_理解与生成.md) | 📝 | Attention 与 FFN 的分工、Output Head 的作用 |

---

## 🧮 01_Fundamentals_基础原理

| 文件 | 类型 | 内容 |
|------|------|------|
| [01_Backpropagation_反向传播与链式法则](01_Fundamentals_基础原理/01_Backpropagation_反向传播与链式法则.md) | 📝 | 链式法则可视化详解 |
| [01_demo_chain_rule_链式法则演示](01_Fundamentals_基础原理/01_demo_chain_rule_链式法则演示.py) | 🐍 | 手动 vs PyTorch 梯度计算 |
| [02_Dimensions_seq_len与emb_dim的区别](01_Fundamentals_基础原理/02_Dimensions_seq_len与emb_dim的区别.md) | 📝 | 序列长度 vs 嵌入维度 |
| [03_parameter_counting_参数计数错误](01_Fundamentals_基础原理/03_parameter_counting_参数计数错误.py) | 🐍 | 为什么 shape[1] 会越界 |
| [04_parameters_vs_named_参数获取对比](01_Fundamentals_基础原理/04_parameters_vs_named_参数获取对比.py) | 🐍 | parameters() vs named_parameters() |
| [05_verify_same_params_验证参数相同](01_Fundamentals_基础原理/05_verify_same_params_验证参数相同.py) | 🐍 | 验证两者返回同一对象 |
| [06_memory_calculator_内存计算器](01_Fundamentals_基础原理/06_memory_calculator_内存计算器.py) | 🐍 | 模型内存占用计算 |
| [07_optimizer_storage_优化器存储](01_Fundamentals_基础原理/07_optimizer_storage_优化器存储.md) | 📝 | 为什么文件从653MB到1.95GB ⭐ NEW |
| [08_parameter_vs_storage_参数与存储换算](01_Fundamentals_基础原理/08_parameter_vs_storage_参数与存储换算.md) | 📝 | 124M参数为什么是652.9MB ⭐ NEW |
| [09_numeric_separator_数字分隔符](01_Fundamentals_基础原理/09_numeric_separator_数字分隔符.py) | 🐍 | Python 数字分隔符详解 ⭐ NEW |

---

## 📝 02_DataProcessing_数据处理

| 文件 | 类型 | 内容 |
|------|------|------|
| [01_Tokenization_文本预处理](02_DataProcessing_数据处理/01_Tokenization_文本预处理.md) | 📝 | BPE、Token ID、Vocabulary |
| [02_tokenizer_decode_分词器解码测试](02_DataProcessing_数据处理/02_tokenizer_decode_分词器解码测试.py) | 🐍 | 解码时的 UTF-8 问题 |
| [03_vocab_extension_词表扩展指南](02_DataProcessing_数据处理/03_vocab_extension_词表扩展指南.py) | 🐍 | 如何扩展已训练模型词表 |

---

## 🏗️ 03_Architecture_模型架构

| 文件 | 类型 | 内容 |
|------|------|------|
| [01_LayerNorm_归一化维度](03_Architecture_模型架构/01_LayerNorm_归一化维度.md) | 📝 | LayerNorm 对 emb_dim 归一化 |
| [01_layernorm_demo_归一化参数演示](03_Architecture_模型架构/01_layernorm_demo_归一化参数演示.py) | 🐍 | scale/shift 参数演示 |
| [02_MultiHead_vs_Conv_多头注意力vs卷积核](03_Architecture_模型架构/02_MultiHead_vs_Conv_多头注意力vs卷积核.md) | 📝 | 多头注意力与 CNN 对比 |
| [03_MHA_vs_MoE_多头vs单头vs混合专家](03_Architecture_模型架构/03_MHA_vs_MoE_多头vs单头vs混合专家.md) | 📝 | Dense vs Sparse 架构 |
| [04_qkv_bias_QKV偏置详解](03_Architecture_模型架构/04_qkv_bias_QKV偏置详解.py) | 🐍 | 为什么 GPT-2 不用 bias |
| [05_emb_dim_divisible_维度整除要求](03_Architecture_模型架构/05_emb_dim_divisible_维度整除要求.py) | 🐍 | emb_dim 必须整除 n_heads |
| [06_ffn_dimension_FFN维度修复](03_Architecture_模型架构/06_ffn_dimension_FFN维度修复.py) | 🐍 | FFN 正确使用 emb_dim |
| [07_FFN_Expansion_FFN维度扩展倍数](03_Architecture_模型架构/07_FFN_Expansion_FFN维度扩展倍数.md) | 📝 | FFN 4倍扩展、Qwen/MiniCPM 调研 |
| [08_Attention_Implementation_注意力机制实现细节](03_Architecture_模型架构/08_Attention_Implementation_注意力机制实现细节.md) | 📝 | QKV 转置原因、out_proj 作用 |
| [09_FFN_hidden_dim_FFN维度扩展详解](03_Architecture_模型架构/09_FFN_hidden_dim_FFN维度扩展详解.md) | 📝 | hidden_dim是MLP最高维度吗 ⭐ NEW |
| [10_head_dim_注意力头维度详解](03_Architecture_模型架构/10_head_dim_注意力头维度详解.md) | 📝 | head_dim与Qwen的GQA设计 ⭐ NEW |

---

## 🎯 04_Training_训练机制

| 文件 | 类型 | 内容 |
|------|------|------|
| [01_MultiPosition_GPT多位置训练](04_Training_训练机制/01_MultiPosition_GPT多位置训练.md) | 📝 | Teacher Forcing 机制 |
| [01_multi_position_多位置训练演示](04_Training_训练机制/01_multi_position_多位置训练演示.py) | 🐍 | 每个位置的预测任务 |
| [02_Loss_Calculation_损失计算详解](04_Training_训练机制/02_Loss_Calculation_损失计算详解.md) | 📝 | 概率提取、维度不匹配、批次训练详解 |
| [02_TeacherForcing_训练与推理的自回归机制](04_Training_训练机制/02_TeacherForcing_训练与推理的自回归机制.md) | 📝 | 训练并行 vs 推理串行、Causal Mask、KV Cache |
| [03_Sampling_Temperature_采样与温度参数](04_Training_训练机制/03_Sampling_Temperature_采样与温度参数.md) | 📝 | 采样机制、温度参数、top-k/top-p、优化器状态 ⭐ NEW |

---

## 📈 05_ScalingLaws_扩展定律

| 文件 | 类型 | 内容 |
|------|------|------|
| [01_Bitter_Lesson_苦涩的教训](05_ScalingLaws_扩展定律/01_Bitter_Lesson_苦涩的教训.md) | 📝 | Rich Sutton 文章翻译 |
| [02_Scaling_Law_扩展定律详解](05_ScalingLaws_扩展定律/02_Scaling_Law_扩展定律详解.md) | 📝 | 幂律、Chinchilla 最优 |
| [03_Early_Stopping_大模型提前停止](05_ScalingLaws_扩展定律/03_Early_Stopping_大模型提前停止.md) | 📝 | 大模型 sample-efficiency |

---

## 🌐 06_MultimodalFrontiers_多模态前沿论文 ⭐ NEW

> 每日自动更新最新多模态统一模型前沿论文（10:03 AM）

| 文件 | 类型 | 内容 |
|------|------|------|
| [README](06_MultimodalFrontiers_多模态前沿论文/README.md) | 📋 | 目录、研究方向分类、自动更新说明 |
| [2026-04-07_Overview](06_MultimodalFrontiers_多模态前沿论文/2026-04-07_Overview.md) | 📊 | 论文总览与趋势分析（11篇论文） |
| [2026-04-07_CHEERS](06_MultimodalFrontiers_多模态前沿论文/2026-04-07_CHEERS.md) | 📝 | 解耦语义与细节的统一多模态模型 |
| [2026-04-07_DyninOmni](06_MultimodalFrontiers_多模态前沿论文/2026-04-07_DyninOmni.md) | 📝 | 首个掩码扩散全模态统一模型 |
| [2026-04-07_MATHENA](06_MultimodalFrontiers_多模态前沿论文/2026-04-07_MATHENA.md) | 📝 | Mamba驱动的医学多任务框架 |
| [2026-04-07_LongCatNext](06_MultimodalFrontiers_多模态前沿论文/2026-04-07_LongCatNext.md) | 📝 | 词汇化多模态的自回归统一 |

**核心趋势**：
- 🔹 离散Token空间统治 - 所有模态转换为离散token
- 🔹 效率优化关键 - 从O(N²)到O(N)的架构演进
- 🔹 理解与生成真正统一 - 单一架构处理双向任务

---

## 统计

| 类型 | 数量 |
|------|------|
| 📝 Markdown 笔记 | 25 |
| 🐍 Python 演示 | 13 |
| 📋 索引文档 | 2 |
| 📊 总览分析 | 1 |
| **总计** | **41** |

---

## 图例

- 📝 = Markdown 文档（概念解释、理论推导）
- 🐍 = Python 脚本（代码演示、实验验证）
- 📋 = 索引文档（目录、分类、导航）
- 📊 = 总览分析（趋势、对比、综述）

---

> 最后更新: 2026-04-08
>
> **新增**：
> - 📝 `01_Fundamentals_基础原理/07_optimizer_storage_优化器存储.md` - Adam优化器状态详解，为什么文件膨胀到3倍
> - 📝 `01_Fundamentals_基础原理/08_parameter_vs_storage_参数与存储换算.md` - 参数数量与存储大小的换算关系
> - 🐍 `01_Fundamentals_基础原理/09_numeric_separator_数字分隔符.py` - Python 数字分隔符详解（151_936）
> - 📝 `03_Architecture_模型架构/09_FFN_hidden_dim_FFN维度扩展详解.md` - FFN扩展-压缩结构，hidden_dim详解
> - 📝 `03_Architecture_模型架构/10_head_dim_注意力头维度详解.md` - head_dim与Qwen的GQA设计
> - 📝 `04_Training_训练机制/03_Sampling_Temperature_采样与温度参数.md` - 采样机制、温度参数、top-k/top-p过滤
> - 📝 `00_Overview_概览/00_QA_Summary_学习问答总结.md` - 更新至21个问题（新增Ch05采样、优化器、Qwen架构）
