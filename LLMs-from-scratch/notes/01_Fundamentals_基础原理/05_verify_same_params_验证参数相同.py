"""
验证：parameters() 和 named_parameters() 返回的是同一个参数对象

证明两者返回的参数张量是完全相同的对象（内存地址相同）
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 2)


def verify_same_parameters():
    """验证两者返回的参数是同一个对象"""
    print("=" * 70)
    print("验证：parameters() 和 named_parameters() 是否返回相同参数")
    print("=" * 70)

    model = SimpleModel()

    # 方法 1: 收集 parameters()
    params_list = list(model.parameters())

    # 方法 2: 收集 named_parameters()
    named_params_list = list(model.named_parameters())

    print(f"\nparameters() 返回数量: {len(params_list)}")
    print(f"named_parameters() 返回数量: {len(named_params_list)}")

    print("\n逐个对比：")
    print("-" * 70)

    for i, (param_only, (name, param_named)) in enumerate(zip(params_list, named_params_list)):
        # 检查是否是同一个对象（内存地址）
        is_same_object = param_only is param_named

        # 检查数据指针是否相同
        same_data_ptr = param_only.data_ptr() == param_named.data_ptr()

        print(f"\n参数 {i}: {name}")
        print(f"  shape: {param_only.shape}")
        print(f"  是同一个对象? {is_same_object} (id相同: {id(param_only) == id(param_named)})")
        print(f"  数据指针相同? {same_data_ptr}")
        print(f"    - parameters() 数据指针: {param_only.data_ptr()}")
        print(f"    - named_parameters() 数据指针: {param_named.data_ptr()}")


def test_modification():
    """测试：修改一个会影响另一个吗？"""
    print("\n\n" + "=" * 70)
    print("测试：修改通过 parameters() 获取的参数，会影响 named_parameters() 吗？")
    print("=" * 70)

    model = SimpleModel()

    # 获取参数
    param_list = list(model.parameters())
    named_param_dict = {name: param for name, param in model.named_parameters()}

    # 记录第一个参数的原始值
    first_param = param_list[0]
    first_param_name = list(named_param_dict.keys())[0]
    first_named_param = named_param_dict[first_param_name]

    print(f"\n修改前:")
    print(f"  parameters()[0] 的前3个值: {first_param.flatten()[:3]}")
    print(f"  named_parameters()['{first_param_name}'] 的前3个值: {first_named_param.flatten()[:3]}")

    # 通过 parameters() 修改
    with torch.no_grad():
        first_param.fill_(999.0)

    print(f"\n修改后 (通过 parameters()[0].fill_(999.0)):")
    print(f"  parameters()[0] 的前3个值: {first_param.flatten()[:3]}")
    print(f"  named_parameters()['{first_param_name}'] 的前3个值: {first_named_param.flatten()[:3]}")
    print(f"\n✅ 两者同步变化，证明是同一个对象！")


def test_gradient():
    """测试：梯度也是共享的吗？"""
    print("\n\n" + "=" * 70)
    print("测试：梯度也是共享的吗？")
    print("=" * 70)

    model = SimpleModel()

    # 前向+反向传播
    x = torch.randn(2, 4)
    target = torch.randn(2, 2)
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()

    # 获取参数和梯度
    params_list = list(model.parameters())
    named_params_dict = {name: param for name, param in model.named_parameters()}

    print("\n检查第一个参数的梯度：")
    first_param = params_list[0]
    first_param_name = list(named_params_dict.keys())[0]
    first_named_param = named_params_dict[first_param_name]

    print(f"  parameters()[0].grad 的前3个值:")
    print(f"    {first_param.grad.flatten()[:3]}")
    print(f"  named_parameters()['{first_param_name}'].grad 的前3个值:")
    print(f"    {first_named_param.grad.flatten()[:3]}")

    # 检查梯度对象是否相同
    same_grad = first_param.grad is first_named_param.grad
    print(f"\n  梯度是同一个对象? {same_grad}")
    print(f"✅ 梯度也是共享的！")


def visualize_relationship():
    """可视化两者的关系"""
    print("\n\n" + "=" * 70)
    print("关系图解")
    print("=" * 70)

    print("""
模型的参数存储在内存中：

    内存中的参数对象
    ┌─────────────────┐
    │ layer1.weight   │  ← 同一个对象
    │ [8, 4] tensor   │
    └─────────────────┘
           ↑     ↑
           │     │
           │     └─────────────────┐
           │                       │
    ┌──────┴─────────┐    ┌────────┴──────────┐
    │ parameters()   │    │ named_parameters()│
    │ 返回: 张量     │    │ 返回: (名称, 张量)│
    └────────────────┘    └───────────────────┘
         只有值               名称 + 值

总结：
1. 两者返回的参数张量是 **同一个对象**
2. 修改一个会影响另一个
3. 梯度也是共享的
4. 唯一区别：named_parameters() 额外提供名称

类比：
- parameters() 就像：给你一本书
- named_parameters() 就像：给你一本书 + 告诉你书名

书还是同一本书，只是 named_parameters() 多了个标签。
    """)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" 验证 parameters() 和 named_parameters() 返回相同参数 ".center(70, "="))
    print("=" * 70)

    # 1. 验证是同一个对象
    verify_same_parameters()

    # 2. 测试修改
    test_modification()

    # 3. 测试梯度
    test_gradient()

    # 4. 可视化关系
    visualize_relationship()

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print("""
✅ 是的，parameters() 和 named_parameters() 返回的参数完全相同！

证据：
1. ✅ 内存地址相同 (id 相同)
2. ✅ 数据指针相同 (data_ptr() 相同)
3. ✅ 修改一个会同步到另一个
4. ✅ 梯度也是共享的

区别：
- parameters():      只返回参数张量
- named_parameters(): 返回 (名称, 参数张量) 元组

本质：
两者指向内存中同一个参数对象，
named_parameters() 只是额外提供了名称信息。

记忆：
就像一个人的两个称呼：
- "张三" = named_parameters() 返回的 ("name", person)
- person = parameters() 返回的 person
人还是同一个人！
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
