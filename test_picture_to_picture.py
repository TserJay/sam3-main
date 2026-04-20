import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as v2
from torch.nn.functional import interpolate
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ======================== 全局设置：解决中文显示和可视化问题 ========================
# 设置matplotlib支持中文（解决字体警告）
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]  # 优先使用黑体，兼容中文
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ======================== 工具函数：移除box_ops依赖，手动实现坐标转换 ========================
def xyxy2cxcywh_norm(box_xyxy, img_width, img_height):
    """
    将[x1,y1,x2,y2]格式的像素坐标，转换为归一化的[cx, cy, w, h]格式（适配Sam3Processor）
    :param box_xyxy: [x1, y1, x2, y2] 像素坐标
    :param img_width: 图片宽度
    :param img_height: 图片高度
    :return: 归一化后的 [cx, cy, w, h]
    """
    x1, y1, x2, y2 = box_xyxy
    # 计算中心坐标、宽高
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    # 归一化到[0,1]
    cx /= img_width
    cy /= img_height
    w /= img_width
    h /= img_height
    return [cx, cy, w, h]

def box_cxcywh_to_xyxy(boxes):
    """
    手动实现box_ops.box_cxcywh_to_xyxy的功能（兼容旧版torchvision）
    将[cx, cy, w, h]格式转换为[x1, y1, x2, y2]格式
    :param boxes: 张量，shape=[N,4]
    :return: 转换后的张量，shape=[N,4]
    """
    x_c = boxes[:, 0]
    y_c = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes_xyxy

# ======================== 1. 初始化模型（适配CPU/GPU） ========================
# 自动检测设备，无CUDA则用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 使用设备：{device}")

# 加载模型和处理器（强制指定device，消除CUDA警告）
model = build_sam3_image_model()
model.to(device)  # 模型移到指定设备
processor = Sam3Processor(
    model=model,
    resolution=1008,
    device=device,  # 适配CPU
    confidence_threshold=0.5
)
model.eval()  # 评估模式

# ======================== 2. 参考图处理：勾选新物体（几何prompt） ========================
# 2.1 加载参考图
ref_image_path = r"E:\工作日志文件夹\sam3算法\sam3-main\assets\images\truck.jpg"
try:
    ref_image = Image.open(ref_image_path).convert("RGB")
    ref_width, ref_height = ref_image.size
    print(f"✅ 参考图加载完成：尺寸 {ref_width}x{ref_height}")
except FileNotFoundError:
    print(f"❌ 参考图路径 {ref_image_path} 不存在！")
    exit(1)

# 2.2 提取参考图特征（仅执行一次）
ref_inference_state = processor.set_image(ref_image)

# 2.3 交互/手动框选新物体（转换为模型要求的格式）
# --- 方案A：手动指定[x1,y1,x2,y2]像素坐标（根据你的物体调整）---
# ref_box_xyxy = [100, 80, 500, 400]  # 替换为你的物体实际坐标
# # 转换为模型需要的归一化cxcywh格式
# ref_box_cxcywh_norm = xyxy2cxcywh_norm(ref_box_xyxy, ref_width, ref_height)

# --- 方案B：鼠标交互框选（已解决中文显示问题）---
plt.figure(figsize=(10, 8))
plt.imshow(ref_image)
plt.title("请用鼠标框选新物体（点击左上角、右下角），按Enter确认")
clicks = plt.ginput(2, timeout=30)  # 等待30秒，选2个点
ref_box_xyxy = [clicks[0][0], clicks[0][1], clicks[1][0], clicks[1][1]]
ref_box_cxcywh_norm = xyxy2cxcywh_norm(ref_box_xyxy, ref_width, ref_height)
plt.close()

# 2.4 添加几何prompt（框选），提取参考物体特征
ref_inference_state = processor.add_geometric_prompt(
    box=ref_box_cxcywh_norm,  # 归一化的cxcywh格式
    label=True,  # 正样本（要找的物体）
    state=ref_inference_state
)
# 此时ref_inference_state已包含参考物体的特征和分割结果
print(f"✅ 参考图物体勾选完成，特征已提取！")
print(f"   参考框（像素）：{[round(x,2) for x in ref_box_xyxy]} → 归一化cxcywh：{[round(x,4) for x in ref_box_cxcywh_norm]}")

# ======================== 3. 跨图匹配：在其他图片中找同款新物体 ========================
# 3.1 目标图片列表（替换为你的路径）
target_image_paths = [
    r"C:\Users\Intel\Desktop\picture\car1.jpg",
    r"C:\Users\Intel\Desktop\picture\car2.jpg",
    r"C:\Users\Intel\Desktop\picture\car3.png",
]

# 3.2 遍历目标图，复用参考特征逻辑匹配
for idx, target_path in enumerate(target_image_paths):
    print(f"\n--- 处理第 {idx+1} 张目标图：{target_path} ---")
    try:
        target_image = Image.open(target_path).convert("RGB")
        target_width, target_height = target_image.size
    except FileNotFoundError:
        print(f"⚠️  目标图 {target_path} 不存在，跳过！")
        continue

    # 3.2.1 提取目标图全局特征
    target_state = processor.set_image(target_image)

    # 3.2.2 基于参考物体的视觉特征，用几何prompt匹配（核心：复用参考框的相对位置逻辑）
    # 方法：将参考框的归一化比例应用到目标图，作为匹配的初始prompt
    target_box_cxcywh_norm = ref_box_cxcywh_norm  # 复用归一化的框比例
    target_state = processor.add_geometric_prompt(
        box=target_box_cxcywh_norm,
        label=True,
        state=target_state
    )

    # 3.2.3 提取分割结果（从state中获取）
    masks = target_state["masks"]
    boxes = target_state["boxes"]
    scores = target_state["scores"]

    # 3.2.4 结果校验与可视化
    if len(scores) == 0:
        print(f"❌ 未匹配到目标物体！")
        continue

    # 取置信度最高的结果
    best_idx = torch.argmax(scores)
    best_mask = masks[best_idx].cpu().numpy()
    # 关键修复：去掉多余的单维度轴（解决shape (1,H,W)错误）
    best_mask = np.squeeze(best_mask)  # 从(1, H, W) → (H, W)
    best_box = boxes[best_idx].cpu().numpy().round(2)
    best_score = scores[best_idx].cpu().item()

    print(f"✅ 匹配成功！")
    print(f"   置信度：{best_score:.4f}")
    print(f"   检测框（像素）：{best_box}")

    # 可视化结果
    plt.figure(figsize=(12, 6))
    # 原始图
    plt.subplot(1, 2, 1)
    plt.imshow(target_image)
    plt.title("原始目标图")
    plt.axis("off")
    # 带掩码的图
    plt.subplot(1, 2, 2)
    plt.imshow(target_image)
    plt.imshow(best_mask, alpha=0.5, cmap="jet")  # 现在shape为(H,W)，可正常显示
    plt.title(f"匹配结果（置信度：{best_score:.4f}）")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ======================== 4. 资源释放 ========================
del ref_inference_state
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("\n✅ 所有目标图处理完成！")