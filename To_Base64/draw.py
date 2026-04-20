# -*- coding: utf-8 -*-  # 核心修复：添加编码声明，解决非ASCII字符报错
import cv2
import json
import os

def draw_boxes_on_image(image_path, detection_result, output_image_path, 
                        box_color=(0, 0, 255), box_thickness=3, text_font_scale=0.8):
    """
    根据检测结果在图片上绘制边界框（仅显示Wheel和置信度），并保存新图片
    
    参数:
        image_path (str): 原始图片路径
        detection_result (dict): 检测结果的JSON字典（包含results字段）
        output_image_path (str): 绘制后图片的保存路径
        box_color (tuple): 框的颜色（BGR格式，默认红色 (0,0,255)）
        box_thickness (int): 框的线条宽度，默认3
        text_font_scale (float): 文字大小比例，默认0.8
    返回:
        bool: 绘制成功返回True，失败返回False
    """
    # 1. 校验输入图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：图片文件不存在 - {image_path}")
        return False
    
    # 2. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 错误：无法读取图片 - {image_path}（可能格式不支持）")
        return False
    
    # 3. 解析检测结果中的边界框
    if not isinstance(detection_result, dict) or "results" not in detection_result:
        print(f"❌ 错误：检测结果格式不正确，缺少'results'字段")
        return False
    
    results = detection_result["results"]
    if len(results) == 0:
        print("⚠️  提示：检测结果中没有找到任何目标框")
        # 仍保存原图
        cv2.imwrite(output_image_path, img)
        return True
    
    # 4. 遍历所有检测框并绘制（仅保留Wheel和置信度）
    for idx, result in enumerate(results):
        # 提取框坐标（浮点数转整数，OpenCV需要整数坐标）
        box = result.get("box", [])
        if len(box) != 4:
            print(f"⚠️  跳过第{idx+1}个框：坐标格式错误 {box}")
            continue
        
        x1, y1, x2, y2 = map(int, box)
        name = result.get("name", "Wheel")  # 默认值设为Wheel
        score = result.get("score", 0.0)
        
        # 绘制矩形框（保留框体，仅简化文字）
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, box_thickness)
        
        # 核心修改：仅显示Wheel和置信度（简化文字内容）
        text = f"{name}: {score:.2f}"
        # 设置文字位置（框的左上角上方，避免超出图片）
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        
        # 绘制文字背景（保留，保证文字可读性）
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, 2)
        text_w, text_h = text_size
        cv2.rectangle(img, (text_x, text_y - text_h - 5), 
                      (text_x + text_w, text_y + 5), box_color, -1)
        
        # 绘制文字（仅显示Wheel和置信度，白色字体）
        cv2.putText(img, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_font_scale, 
                    (255, 255, 255), 2)  # 白色字体，线条宽度2
    
    # 5. 保存绘制后的图片
    cv2.imwrite(output_image_path, img)
    
    # 打印成功信息
    print(f"✅ 绘制完成！共绘制 {len(results)} 个目标框")
    print(f"📄 新图片已保存到：{output_image_path}")
    
    # 可选：显示图片尺寸信息
    height, width = img.shape[:2]
    print(f"📏 图片尺寸：{width}x{height} 像素")
    
    return True

# 示例使用
if __name__ == "__main__":
    # ===================== 配置参数（修改这里！）=====================
    # 1. 原始图片路径
    INPUT_IMAGE_PATH = "bucket2.jpg"  # 替换为你的图片路径
    # 2. 绘制后图片的保存路径
    OUTPUT_IMAGE_PATH = "bucket2_with_boxes.jpg"  # 输出新图片路径
    # 3. 检测结果（仅保留Wheel和置信度相关）
    DETECTION_RESULT = {
  "inference_time_ms": 4748.466064453125,
  "results": [
    {
      "box": [
        393.72552490234375,
        417.6777038574219,
        630.8574829101562,
        525.0747680664062
      ],
      "mask_shape": [
        1,
        790,
        1444
      ],
      "name": "flange",
      "score": 0.5406079292297363
    },
    {
      "box": [
        864.4526977539062,
        421.1744689941406,
        1128.4818115234375,
        563.363525390625
      ],
      "mask_shape": [
        1,
        790,
        1444
      ],
      "name": "flange",
      "score": 0.8260776400566101
    },
    {
      "box": [
        387.8464660644531,
        147.64697265625,
        890.4852905273438,
        529.3404541015625
      ],
      "mask_shape": [
        1,
        790,
        1444
      ],
      "name": "flange",
      "score": 0.5095126032829285
    }
  ],
  "success": True
}
    
    # ===================== 执行绘制 =====================
    try:
        # 调用绘制函数
        success = draw_boxes_on_image(
            image_path=INPUT_IMAGE_PATH,
            detection_result=DETECTION_RESULT,
            output_image_path=OUTPUT_IMAGE_PATH,
            box_color=(0, 0, 255),  # 红色框（BGR格式）
            box_thickness=3,
            text_font_scale=0.8
        )
        
        if success:
            print("\n🎉 所有操作完成！")
        else:
            print("\n❌ 绘制失败，请检查上述错误提示")
    
    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")