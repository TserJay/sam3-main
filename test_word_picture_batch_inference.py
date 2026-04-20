import torch
import os
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def this_main(image_paths,batch_prompts,model,processor,batch_size=None):
    if not batch_size:
        batch_size = len(image_paths)
    '''
    image_paths:需要识别的图像文件集合；
    batch_size：每一批次大小
    batch_prompts:批量文本提示,一张图里可能存在多个文本提示
    '''

    # ===================== 3. 准备批量输入数据 =====================
    # 批量图像路径（确保路径正确，可替换为你的图像）
    # image_paths = [
    #     r"E:\工作日志文件夹\sam3算法\sam3-main\assets\images\truck.jpg",
    #     r"C:\Users\Intel\Desktop\picture\car1.jpg"
    # ]
    #
    # # 批量文本提示（与图像一一对应，支持中英文）
    # batch_prompts = [
    #     "truck",  # 第一张图的提示
    #     "truck"    # 第二张图的提示
    # ]

    # 批量加载图像（验证图像是否能正常打开，避免路径错误）
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        images.append(img)

    # ===================== 4. 批量推理核心逻辑（修复set_image不支持列表的问题） =====================
    # 步骤1：循环处理每张图像，生成对应的inference_state（替代直接传列表）
    inference_states = []
    for img in images:
        # 单张图像调用set_image，生成单个state，添加到列表
        state = processor.set_image(img)
        inference_states.append(state)

    # 步骤2：批量设置文本提示并推理
    batch_outputs = []
    with torch.no_grad():  # 禁用梯度，节省内存（CPU环境同样重要）
        for idx, state in enumerate(inference_states):
            # 对每个样本的state设置对应的文本提示
            output = processor.set_text_prompt(
                state=state,
                prompt=batch_prompts[idx]
            )
            batch_outputs.append(output)

    # ===================== 5. 解析批量结果 =====================
    for idx, output in enumerate(batch_outputs):
        # masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        yield output