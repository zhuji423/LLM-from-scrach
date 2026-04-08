# Python 数字分隔符详解

> Python 3.6+ 引入的数字可读性特性

---

## 问题：为什么代码中写 151_936 而不是 151936？

### 快速答案

**这是 Python 的数字视觉分隔符**，用于提高大数字的可读性。

```python
# 以下三种写法完全等价
vocab_size = 151936      # 难以快速识别
vocab_size = 151_936     # 清晰：15万1936
vocab_size = 1_51_936    # 也合法，但不符合习惯

# 验证
>>> 151_936 == 151936
True

>>> type(151_936)
<class 'int'>
```

**重要**：下划线**仅影响视觉**，不影响数值本身！

---

## 为什么需要这个特性？

### 可读性对比

| 写法 | 可读性 | 一眼能看出大小吗？ |
|------|--------|-------------------|
| `151936` | ⭐⭐ | 需要数位数：1..5..1..9..3..6（6位？5位？） |
| `151_936` | ⭐⭐⭐⭐⭐ | 直接看出：**15万1千9百36** |

### 真实场景对比

```python
# ❌ 不易读：这是多少？
model_params = 124356864
file_size = 1958739456
max_tokens = 1000000

# ✅ 易读：一目了然
model_params = 124_356_864     # 1.24亿参数
file_size = 1_958_739_456      # 约19.6亿字节
max_tokens = 1_000_000         # 100万 tokens
```

---

## 语法规则

### 1. 位置灵活但有惯例

```python
# ✅ 推荐：每三位一组（模拟千位分隔符）
million = 1_000_000
billion = 1_000_000_000

# ✅ 合法但不常见：其他分组方式
chinese_style = 1_0000_0000  # 按中文"万"分组（1亿）
binary = 0b_1111_0000         # 二进制按4位分组

# ✅ 合法：多个下划线连续
weird = 1__000___000  # 不推荐，但语法合法

# ❌ 非法：不能在开头或结尾
invalid = _1000  # SyntaxError
invalid = 1000_  # SyntaxError
```

### 2. 支持所有数字类型

#### 整数

```python
int_num = 1_000_000
print(int_num)  # 1000000
```

#### 浮点数

```python
pi = 3.141_592_653_589
print(pi)  # 3.141592653589

price = 1_234.56
print(price)  # 1234.56
```

#### 科学计数法

```python
avogadro = 6.022_140_76e23  # 阿伏伽德罗常数
print(avogadro)  # 6.02214076e+23
```

#### 十六进制

```python
color = 0xFF_AA_00  # RGB 颜色
print(color)  # 16755200
print(hex(color))  # 0xffaa00
```

#### 二进制

```python
flags = 0b_1010_1100_0011_1111
print(flags)  # 44095
print(bin(flags))  # 0b1010110000111111
```

#### 八进制

```python
permissions = 0o_777_666
print(permissions)  # 261046
print(oct(permissions))  # 0o777666
```

### 3. 在表达式中的行为

```python
# 计算结果不保留下划线
result = 100_000 + 50_000
print(result)  # 150000（不是 150_000）

# 但字面量本身保留可读性
vocab_sizes = [151_936, 50_257, 32_000]  # 代码中易读
print(vocab_sizes)  # [151936, 50257, 32000]
```

---

## 实际应用场景

### 1. 模型配置中的大数字

```python
# Qwen3 模型配置
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,               # 词汇表大小：15万
    "context_length": 40_960,            # 上下文长度：4万
    "emb_dim": 1024,                     # 嵌入维度：1024（不大，不用分隔）
    "hidden_dim": 3072,                  # 隐藏层：3072（不大，不用分隔）
    "rope_base": 1_000_000.0,            # RoPE基数：100万
    "num_parameters": 600_000_000,       # 参数量：6亿
}
```

### 2. 内存和文件大小

```python
# 内存限制
MAX_MEMORY = 16_000_000_000  # 16GB
BATCH_SIZE_THRESHOLD = 1_000_000  # 100万样本

# 文件大小检查
if file_size > 1_000_000_000:  # 1GB
    print("Large file warning")

# 缓存配置
CACHE_SIZE = 500_000_000  # 500MB
```

### 3. 时间和速率

```python
# 时间常量（毫秒）
ONE_SECOND_MS = 1_000
ONE_MINUTE_MS = 60_000
ONE_HOUR_MS = 3_600_000
ONE_DAY_MS = 86_400_000

# 速率限制
MAX_REQUESTS_PER_DAY = 1_000_000
TOKENS_PER_MINUTE = 90_000
```

### 4. 科学计算

```python
# 物理常数
SPEED_OF_LIGHT = 299_792_458  # m/s
AVOGADRO = 6.022_140_76e23
PLANCK = 6.626_070_15e-34  # J⋅s

# 精确的金融计算
CENTS_IN_MILLION = 100_000_000  # $1M = 100,000,000 cents
```

---

## 字符串转换

### 输出时不保留下划线

```python
# 字面量转字符串：不保留下划线
num = 1_000_000
print(str(num))  # "1000000"

# 格式化输出：可以加千位分隔符
print(f"{num:,}")  # "1,000,000" (逗号分隔)
print(f"{num:_}")  # "1_000_000" (下划线分隔)
```

### 字符串转数字：可以包含下划线

```python
# int() 和 float() 支持下划线
int("1_000_000")   # 有效！返回 1000000
float("3.14_159")  # 有效！返回 3.14159

# 但字符串中的下划线位置必须合法
int("_1000")  # ValueError: invalid literal
int("1000_")  # ValueError: invalid literal
```

### f-string 中的格式化

```python
num = 1234567890

# 不同的格式化方式
print(f"{num}")           # "1234567890"
print(f"{num:,}")         # "1,234,567,890" (逗号，常用)
print(f"{num:_}")         # "1_234_567_890" (下划线)
print(f"{num:.2e}")       # "1.23e+09" (科学记数法)

# 与 locale 的关系
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
print(f"{num:n}")         # "1,234,567,890" (根据 locale)
```

---

## 不同编程语言的对比

| 语言 | 数字分隔符语法 | 引入版本 | 示例 |
|------|---------------|---------|------|
| **Python** | `1_000_000` | Python 3.6 (2016) | ✓ |
| **Java** | `1_000_000` | Java 7 (2011) | ✓ |
| **C++** | `1'000'000` | C++14 (2014) | 单引号 |
| **Rust** | `1_000_000` | 一直支持 | ✓ |
| **JavaScript** | `1_000_000` | ES2021 (2021) | ✓ |
| **Ruby** | `1_000_000` | 一直支持 | ✓ |
| **Go** | 不支持 | - | ✗ |
| **C** | 不支持（C23前） | - | ✗ |

---

## PEP 515 标准

这个特性来自 [PEP 515](https://www.python.org/dev/peps/pep-0515/)：

> **Underscores in Numeric Literals**
>
> **目标**：提高长数字字面量的可读性，避免手工数位数时出错
>
> **动机**：
> - 人类在阅读长数字时容易出错（123456789 vs 12345678）
> - 其他语言（C++14, Java 7, Ruby）已经支持类似特性
> - 编程实践中经常需要使用大数字（文件大小、内存地址、科学常数）

### 设计原则

1. **向后兼容**：不破坏现有代码
2. **可选使用**：开发者可以选择是否使用
3. **类型无关**：适用于所有数字类型
4. **位置灵活**：允许多种分组方式（但推荐千位分隔）

---

## 最佳实践

### ✅ 推荐用法

```python
# 1. 大于 4 位数时使用下划线
small = 1000      # 可以不用
medium = 10_000   # 建议使用
large = 1_000_000 # 强烈建议

# 2. 遵循千位分隔惯例（每三位）
population = 7_800_000_000  # 78亿
gdp = 21_000_000_000_000    # 21万亿

# 3. 二进制、十六进制按4位分组
flags = 0b_1111_0000_1010_0101
color = 0xFF_AA_00
ipv6 = 0x2001_0db8_0000_0000_0000_ff00_0042_8329

# 4. 科学常数保留原有精度的分组
GOLDEN_RATIO = 1.618_033_988_749_895
E = 2.718_281_828_459_045

# 5. 配置文件中的大数字
config = {
    "max_memory": 4_294_967_296,  # 4GB
    "timeout_ms": 30_000,          # 30秒
    "batch_size": 10_000,          # 1万
}
```

### ❌ 避免的用法

```python
# ❌ 不要随意分组，让人困惑
weird = 1_2_3_4_5_6

# ❌ 小数点两边不要混乱分组
bad_float = 12_34.56_78  # 不直观

# ✅ 更好的浮点数分组
good_float = 1_234.567_8  # 整数部分按千位，小数部分按习惯

# ❌ 不要在不需要的地方用
year = 20_26  # 年份不需要分隔
age = 2_5     # 年龄不需要分隔

# ✅ 只在确实难以阅读的数字上使用
salary = 75_000    # 这个合适
price = 29.99      # 这个不需要
```

---

## 实际代码示例

### Qwen 配置文件

```python
# 实际的 Qwen3 配置
QWEN3_CONFIG_06_B = {
    "vocab_size": 151_936,           # 词汇表：15万
    "context_length": 40_960,        # 上下文：4万token
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 3072,
    "head_dim": 128,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,        # RoPE基数：100万
}

QWEN3_CONFIG_32B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 5120,
    "n_heads": 64,
    "n_layers": 64,
    "hidden_dim": 25600,
    "rope_base": 1_000_000.0,
}
```

### 内存计算示例

```python
def calculate_memory_usage(config):
    """计算模型内存占用"""
    # 参数量计算
    vocab_size = config["vocab_size"]  # 151_936
    emb_dim = config["emb_dim"]        # 1024

    token_emb_params = vocab_size * emb_dim
    # 151_936 * 1024 = 155_582_464 参数

    # 存储大小（float32）
    bytes_per_param = 4
    model_size_bytes = token_emb_params * bytes_per_param
    model_size_mb = model_size_bytes / (1024 ** 2)

    print(f"Token Embedding 参数: {token_emb_params:_}")
    print(f"存储大小: {model_size_mb:_.2f} MB")

calculate_memory_usage(QWEN3_CONFIG_06_B)
# 输出:
# Token Embedding 参数: 155_582_464
# 存储大小: 148.38 MB
```

---

## 类比：日常写数字

这和我们日常写数字时的习惯类似：

| 语言 | 写法 | Python 等价 |
|------|------|-------------|
| **英文** | 1,000,000 | `1_000_000` |
| **中文** | 100万 | `1_000_000` |
| **科学** | 1×10⁶ | `1_000_000` 或 `1e6` |

**Python 的选择**：使用下划线而不是逗号，因为：
1. 逗号在 Python 中有语法意义（元组分隔符）
2. 下划线在数字中没有歧义
3. 其他语言（Java, Rust）也使用下划线

---

## 性能影响

**没有任何性能影响！**

```python
import timeit

# 测试
t1 = timeit.timeit("x = 1000000", number=10_000_000)
t2 = timeit.timeit("x = 1_000_000", number=10_000_000)

print(f"无下划线: {t1:.6f}s")
print(f"有下划线: {t2:.6f}s")
# 结果：完全相同

# 原因：编译时就已经转换为相同的字节码
import dis
dis.dis("1_000_000")
# 输出的字节码和 "1000000" 完全相同
```

---

## 总结

### 核心要点

| 概念 | 要点 |
|------|------|
| **作用** | 提高数字字面量的可读性 |
| **语法** | 数字中可以用 `_` 分隔（除了开头和结尾） |
| **效果** | 纯视觉，不影响数值 |
| **推荐** | 大于 4 位数时使用，按千位分组 |
| **性能** | 无影响（编译时移除） |

### 记忆口诀

> "下划线只管看，不管算"
> "数字太长记不住，每三位画一道"

### 实用建议

1. **一致性**：团队内保持统一风格
2. **适度使用**：不要在小数字上滥用
3. **遵循惯例**：整数按千位（3位），二进制/十六进制按4位
4. **提高可读性**：让代码更容易理解是首要目标

---

> 最后更新: 2026-04-08
>
> **参考**：
> - [PEP 515 -- Underscores in Numeric Literals](https://www.python.org/dev/peps/pep-0515/)
> - [Python 3.6 Release Notes](https://docs.python.org/3/whatsnew/3.6.html#pep-515-underscores-in-numeric-literals)
