"""
Softmax函数可视化
展示softmax的核心特性:归一化、温度效应、多维度行为
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def softmax(x, temperature=1.0):
    """
    计算softmax函数

    参数:
        x: 输入数组
        temperature: 温度参数,控制分布的"锐利度"
            - T → 0: 分布越尖锐,趋近one-hot
            - T → ∞: 分布越平缓,趋近均匀分布
    """
    x_scaled = x / temperature
    exp_x = np.exp(x_scaled - np.max(x_scaled))  # 数值稳定技巧:减去最大值
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def plot_softmax_single_dimension():
    """可视化1:单维度变化时的softmax输出"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图:固定其他维度,变化一个维度
    x_vary = np.linspace(-5, 5, 100)
    fixed_values = [0, 1, 2]

    for fixed in fixed_values:
        outputs = []
        for x in x_vary:
            logits = np.array([x, fixed, 0])  # 3个类别的logits
            probs = softmax(logits)
            outputs.append(probs[0])  # 关注第一个类别的概率

        axes[0].plot(x_vary, outputs, label=f'其他类别logit=[{fixed}, 0]', linewidth=2)

    axes[0].set_xlabel('目标类别的logit值', fontsize=12)
    axes[0].set_ylabel('目标类别的softmax概率', fontsize=12)
    axes[0].set_title('Softmax特性:相对值决定概率', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50%概率线')

    # 右图:温度效应
    temperatures = [0.5, 1.0, 2.0, 5.0]
    logits = np.array([3.0, 1.0, 0.2])  # 固定的logits

    bar_width = 0.15
    x_pos = np.arange(len(logits))

    for i, T in enumerate(temperatures):
        probs = softmax(logits, temperature=T)
        offset = bar_width * (i - len(temperatures)/2 + 0.5)
        axes[1].bar(x_pos + offset, probs, bar_width,
                   label=f'T={T}', alpha=0.8)

    axes[1].set_xlabel('类别索引', fontsize=12)
    axes[1].set_ylabel('概率', fontsize=12)
    axes[1].set_title('温度效应:控制分布的锐利度', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(['类别0\n(logit=3.0)', '类别1\n(logit=1.0)', '类别2\n(logit=0.2)'])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/Users/zhuji_m4pro/code/LLM_from_scrach/LLMs-from-scratch/Playgrounds/softmax_properties.png',
                dpi=150, bbox_inches='tight')
    print("✓ 图1已保存: softmax_properties.png")


def plot_softmax_3d_surface():
    """可视化2:三维空间中的softmax表面"""
    fig = plt.figure(figsize=(15, 5))

    # 创建网格
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)

    # 对于3个类别,固定第三个类别的logit为0
    for idx, title in enumerate(['类别0的概率', '类别1的概率', '类别2的概率']):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                logits = np.array([X[i,j], Y[i,j], 0.0])
                probs = softmax(logits)
                Z[i,j] = probs[idx]

        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.9,
                              linewidth=0, antialiased=True)

        ax.set_xlabel('类别0的logit', fontsize=10)
        ax.set_ylabel('类别1的logit', fontsize=10)
        ax.set_zlabel('概率', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=20, azim=45)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig('/Users/zhuji_m4pro/code/LLM_from_scrach/LLMs-from-scratch/Playgrounds/softmax_3d_surface.png',
                dpi=150, bbox_inches='tight')
    print("✓ 图2已保存: softmax_3d_surface.png")


def plot_softmax_comparison():
    """可视化3:softmax vs 其他归一化方法"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 生成测试数据
    logits = np.array([5.0, 2.0, 1.0, 0.5, 0.1])
    x_pos = np.arange(len(logits))

    # 1. 原始logits
    axes[0,0].bar(x_pos, logits, color='steelblue', alpha=0.7)
    axes[0,0].set_title('原始Logits(未归一化)', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('类别')
    axes[0,0].set_ylabel('值')
    axes[0,0].grid(True, alpha=0.3, axis='y')

    # 2. Softmax归一化
    softmax_probs = softmax(logits)
    axes[0,1].bar(x_pos, softmax_probs, color='coral', alpha=0.7)
    axes[0,1].set_title('Softmax归一化(指数+归一化)', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('类别')
    axes[0,1].set_ylabel('概率')
    axes[0,1].axhline(y=1/len(logits), color='g', linestyle='--',
                     alpha=0.5, label='均匀分布')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3, axis='y')
    axes[0,1].text(0.5, 0.95, f'总和={softmax_probs.sum():.4f}',
                  transform=axes[0,1].transAxes, ha='center')

    # 3. 线性归一化(L1)
    linear_norm = logits / np.sum(logits)
    axes[1,0].bar(x_pos, linear_norm, color='lightgreen', alpha=0.7)
    axes[1,0].set_title('线性归一化(简单除以总和)', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('类别')
    axes[1,0].set_ylabel('比例')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    axes[1,0].text(0.5, 0.95, f'总和={linear_norm.sum():.4f}',
                  transform=axes[1,0].transAxes, ha='center')

    # 4. Hardmax(argmax one-hot)
    hardmax = np.zeros_like(logits)
    hardmax[np.argmax(logits)] = 1.0
    axes[1,1].bar(x_pos, hardmax, color='orchid', alpha=0.7)
    axes[1,1].set_title('Hardmax(Argmax one-hot)', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('类别')
    axes[1,1].set_ylabel('选择')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    axes[1,1].text(0.5, 0.95, '硬性选择最大值',
                  transform=axes[1,1].transAxes, ha='center')

    plt.tight_layout()
    plt.savefig('/Users/zhuji_m4pro/code/LLM_from_scrach/LLMs-from-scratch/Playgrounds/softmax_comparison.png',
                dpi=150, bbox_inches='tight')
    print("✓ 图3已保存: softmax_comparison.png")


def plot_gradient_flow():
    """可视化4:softmax的梯度流动"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 模拟前向传播
    x = np.linspace(-4, 4, 100)
    logits_3d = np.stack([x, np.zeros_like(x), -x], axis=1)
    probs = softmax(logits_3d)

    # 左图:概率随logit变化
    axes[0].plot(x, probs[:, 0], label='P(类别0)', linewidth=2)
    axes[0].plot(x, probs[:, 1], label='P(类别1)', linewidth=2)
    axes[0].plot(x, probs[:, 2], label='P(类别2)', linewidth=2)
    axes[0].set_xlabel('类别0的logit (其他固定)', fontsize=12)
    axes[0].set_ylabel('概率', fontsize=12)
    axes[0].set_title('Softmax输出的平滑性', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图:梯度分析
    # 对于softmax(x_i),关于x_i的梯度为: p_i * (1 - p_i)
    # 关于x_j(j≠i)的梯度为: -p_i * p_j
    target_idx = 0  # 假设类别0是目标
    p_i = probs[:, target_idx]

    # 自身梯度
    grad_self = p_i * (1 - p_i)
    axes[1].plot(x, grad_self, label='∂P₀/∂logit₀ (自身)', linewidth=2)

    # 对其他类别的梯度
    grad_other = -p_i * probs[:, 1]
    axes[1].plot(x, grad_other, label='∂P₀/∂logit₁ (交叉)', linewidth=2, linestyle='--')

    axes[1].set_xlabel('类别0的logit', fontsize=12)
    axes[1].set_ylabel('梯度值', fontsize=12)
    axes[1].set_title('Softmax的梯度特性', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/zhuji_m4pro/code/LLM_from_scrach/LLMs-from-scratch/Playgrounds/softmax_gradients.png',
                dpi=150, bbox_inches='tight')
    print("✓ 图4已保存: softmax_gradients.png")


if __name__ == "__main__":
    print("开始生成Softmax函数可视化...\n")

    plot_softmax_single_dimension()
    plot_softmax_3d_surface()
    plot_softmax_comparison()
    plot_gradient_flow()

    print("\n" + "="*60)
    print("所有可视化已完成!")
    print("="*60)
    print("\n关键洞察:")
    print("1. Softmax将任意实数映射到概率分布(和为1)")
    print("2. 温度参数控制分布的'锐利度'")
    print("3. 相比线性归一化,softmax放大了差异(指数效应)")
    print("4. 梯度特性:p_i(1-p_i)使得极端概率(接近0或1)梯度小")
    print("5. 这种梯度特性有助于稳定训练,但也可能导致饱和")
