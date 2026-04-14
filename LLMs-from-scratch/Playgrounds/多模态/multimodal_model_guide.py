"""
多模态模型调用、部署与调试指南
========================================

支持的模型来源:
1. TorchVision: ViT, ResNet, EfficientNet 等视觉模型
2. OpenAI CLIP: 文本-图像对齐模型 (需单独安装)
3. Salesforce BLIP: 多模态理解模型 (需单独安装)
"""

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

print("=" * 70)
print("第1部分: TorchVision ViT 模型 (已在 torchvision 中)")
print("=" * 70)

# ============================================================================
# 1. ViT 模型列表 & 加载
# ============================================================================

def list_vit_models():
    """列出所有可用的 ViT 模型"""
    vit_models = [
        'vit_b_16',      # Base, patch size 16
        'vit_b_32',      # Base, patch size 32
        'vit_l_16',      # Large, patch size 16
        'vit_l_32',      # Large, patch size 32
        'vit_h_14',      # Huge, patch size 14
    ]
    return vit_models

print("\n✅ 可用的 ViT 模型:")
for model_name in list_vit_models():
    print(f"   - {model_name}")

# ============================================================================
# 2. 加载预训练 ViT 模型
# ============================================================================

def load_vit_model(model_name='vit_b_16', pretrained=True, device='cpu'):
    """
    加载预训练 ViT 模型

    Args:
        model_name: 模型名称 (vit_b_16, vit_l_16 等)
        pretrained: 是否使用预训练权重
        device: 'cpu' 或 'cuda'

    Returns:
        model: PyTorch 模型
    """
    model = getattr(models, model_name)(pretrained=pretrained, progress=True)
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model

print("\n✅ 加载 ViT 模型...")
try:
    model = load_vit_model('vit_b_16', device='cpu')
    print(f"   成功加载 vit_b_16")
    print(f"   模型参数数: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ⚠️  加载失败: {e}")

# ============================================================================
# 3. 图像预处理
# ============================================================================

def get_image_transforms():
    """获取 ViT 的标准预处理"""
    # ViT 期望的输入大小: 224x224
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform

def load_image_from_url(url, size=(224, 224)):
    """从 URL 加载图像"""
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None

# ============================================================================
# 4. 模型推理 (单张图像)
# ============================================================================

def infer_vit(model, image_path_or_url, device='cpu'):
    """
    使用 ViT 进行推理

    Args:
        model: ViT 模型
        image_path_or_url: 本地路径或 URL
        device: 'cpu' 或 'cuda'

    Returns:
        logits: 模型输出 (batch_size, 1000)
    """
    transform = get_image_transforms()

    # 加载图像
    if image_path_or_url.startswith('http'):
        img = load_image_from_url(image_path_or_url)
    else:
        img = Image.open(image_path_or_url).convert('RGB')

    if img is None:
        return None

    # 预处理
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加 batch 维度

    # 推理
    with torch.no_grad():
        output = model(img_tensor)

    return output, img

print("\n✅ ViT 推理设置完成")

# ============================================================================
# 第2部分: 安装 CLIP 和 BLIP 指南
# ============================================================================

print("\n" + "=" * 70)
print("第2部分: CLIP 和 BLIP 模型 (需单独安装)")
print("=" * 70)

print("""
❌ CLIP 和 BLIP 不在 torchvision 中,需要单独安装:

【CLIP 安装】
  pip install git+https://github.com/openai/CLIP.git
  或
  pip install open-clip-torch

【BLIP 安装】
  pip install salesforce-lavis
  或
  pip install git+https://github.com/salesforce-ai/LAVIS.git

【用途对比】
  - ViT (torchvision):     单模态视觉分类模型
  - CLIP:                  文本-图像对齐,零样本分类,图文匹配
  - BLIP:                  图像标注,视觉问答,图文检索
""")

# ============================================================================
# 第3部分: 调试辅助工具
# ============================================================================

def debug_model_info(model):
    """打印模型信息用于调试"""
    print("\n📊 模型调试信息:")
    print(f"   模型类型: {type(model).__name__}")
    print(f"   总参数数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 打印第一层信息
    first_layer = next(iter(model.children()))
    print(f"   第一层: {type(first_layer).__name__}")

    # 打印最后一层信息
    last_layer = list(model.children())[-1]
    print(f"   最后一层: {type(last_layer).__name__}")

def debug_inference(model, image_tensor, device='cpu'):
    """调试推理过程"""
    print("\n🔍 推理过程调试:")
    print(f"   输入形状: {image_tensor.shape}")
    print(f"   输入范围: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    print(f"   输入设备: {image_tensor.device}")

    with torch.no_grad():
        output = model(image_tensor)

    print(f"   输出形状: {output.shape}")
    print(f"   输出范围: [{output.min():.3f}, {output.max():.3f}]")

    # 获取 top-5 预测
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top5 = torch.topk(probabilities, 5, dim=1)

    print(f"   Top-5 概率: {top5.values[0].tolist()}")

    return output

# ============================================================================
# 第4部分: 部署建议
# ============================================================================

print("\n" + "=" * 70)
print("第3部分: 模型部署建议")
print("=" * 70)

print("""
【推理部署】:
  1. 批量推理优化:
     - 使用 model.eval() 禁用 dropout/batchnorm
     - 使用 torch.no_grad() 禁用梯度计算
     - 批量处理图像提高吞吐量

  2. 模型量化:
     - 动态量化: torch.quantization.quantize_dynamic()
     - 模型大小减少 75%，推理加速 3-4 倍

  3. ONNX 导出 (跨平台):
     torch.onnx.export(model, dummy_input, "model.onnx")

  4. 服务化:
     - FastAPI + Gunicorn (生产环境)
     - TensorFlow Serving (大规模部署)
     - TorchServe (PyTorch 官方工具)

【性能优化】:
  - ViT 速度最慢,ResNet/ConvNeXt 更快
  - 使用混合精度训练 (torch.cuda.amp)
  - 使用 xFormers 加速 attention 计算
""")

# ============================================================================
# 实例代码: 完整推理管道
# ============================================================================

def full_pipeline_example():
    """完整的推理管道示例"""
    print("\n" + "=" * 70)
    print("完整推理管道示例")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 1. 加载模型
    print("\n1️⃣  加载模型...")
    model = load_vit_model('vit_b_16', device=device)
    debug_model_info(model)

    # 2. 加载示例图像
    print("\n2️⃣  加载示例图像...")
    sample_img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"

    img = load_image_from_url(sample_img_url)
    if img is not None:
        print(f"   图像大小: {img.size}")

    # 3. 推理
    print("\n3️⃣  执行推理...")
    transform = get_image_transforms()
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    # 4. 调试信息
    debug_inference(model, img_tensor, device)

    # 5. 显示结果
    print("\n4️⃣  推理完成!")
    print(f"   输出形状: {output.shape} (batch_size=1, 1000 classes)")

if __name__ == '__main__':
    # 运行示例
    try:
        full_pipeline_example()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
