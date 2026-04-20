import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ======================== 1. 初始化模型（仅执行一次） ========================
# 加载SAM3图像模型和处理器
model = build_sam3_image_model()
processor = Sam3Processor(model)

# ======================== 2. 提取图像特征（仅执行一次） ========================
# 加载图像（注意：替换为你的实际图像路径）
try:
    image = Image.open(r"E:\工作日志文件夹\sam3算法\sam3-main\assets\images\truck.jpg")
except FileNotFoundError:
    print("错误：图像文件路径不存在，请检查路径是否正确！")
    exit(1)

# 提取图像特征，生成inference_state（核心：这一步只做一次，缓存特征）
inference_state = processor.set_image(image)
print("✅ 图像特征提取完成，开始处理多个文本查询...")

# ======================== 3. 复用图像特征，处理多个文本查询 ========================
# 定义多个需要查询的文本prompt
text_prompts = [
    "truck",
    "wheels of the truck",
    "window of the truck",
    "headlight of the truck"
]

# 遍历所有prompt，复用同一个inference_state进行推理
for idx, prompt in enumerate(text_prompts):
    print(f"\n--- 处理第 {idx + 1} 个查询：{prompt} ---")
    # 复用已提取的图像特征，仅执行文本prompt的推理
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    # 获取分割结果
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    # 打印结果信息
    print(f"类型：mask={type(masks)}, box={type(boxes)}, score={type(scores)}")
    print(f"mask形状：{masks.shape if masks is not None else 'None'}")
    print(f"box数量：{len(boxes) if boxes is not None else 0}")
    print(f"最高置信度：{scores.max().item() if scores is not None else 0:.4f}")

# 可选：如果需要清理内存，推理完成后释放资源
del inference_state
torch.cuda.empty_cache()  # 如果使用GPU，清理显存