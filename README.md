# LLM from Scratch

学习 [Build a Large Language Model From Scratch](https://github.com/rasbt/LLMs-from-scratch) 的笔记与代码。

## Notes 内容

### 00 概览
学习问答汇总(21问)、核心概念图解、理解与生成的区别

### 01 基础原理
反向传播与链式法则、seq_len vs emb_dim、参数计数与内存计算、优化器存储、参数与存储换算

### 02 数据处理
Tokenization 预处理、分词器解码、词表扩展

### 03 模型架构
LayerNorm 归一化维度、多头注意力 vs 卷积核、MHA/MQA/GQA/MoE 对比、QKV 偏置、FFN 维度扩展与 hidden_dim、head_dim 与 GQA 设计、注意力机制实现细节

### 04 训练机制
GPT 多位置训练与 Teacher Forcing、损失计算详解、自回归机制与 KV Cache、采样/温度/top-k/top-p

### 05 扩展定律
Bitter Lesson、Scaling Law 幂律与 Chinchilla 最优、大模型 Early Stopping

### 06 多模态前沿论文
CHEERS、DyninOmni、LongCatNext、MATHENA 等论文笔记；趋势：离散 Token 统一、O(N^2)->O(N) 架构演进、理解与生成统一
