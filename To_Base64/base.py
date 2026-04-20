import base64
import os

def image_to_base64(image_path, output_file_path, add_prefix=True):
    """
    将JPG/JPEG或PNG图片转换为Base64编码，并保存到指定文件
    
    参数:
        image_path (str): 输入的图片文件路径（支持JPG/JPEG/PNG）
        output_file_path (str): 输出Base64编码的文本文件路径
        add_prefix (bool): 是否添加Base64前缀（如data:image/jpeg;base64,），默认True（适配接口）
    返回:
        str: 生成的Base64编码字符串
    """
    # 校验输入文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在：{image_path}")
    
    # 校验文件扩展名并获取图片格式（支持jpg/jpeg/png）
    file_ext = os.path.splitext(image_path)[1].lower()
    supported_exts = {
        '.jpg': 'jpeg',
        '.jpeg': 'jpeg',
        '.png': 'png'
    }
    if file_ext not in supported_exts:
        raise ValueError(f"不支持的文件格式：{image_path}，仅支持JPG/JPEG/PNG")
    
    # 获取对应格式的MIME类型（用于Base64前缀）
    image_format = supported_exts[file_ext]
    
    # 读取图片并转换为Base64
    with open(image_path, 'rb') as image_file:
        # 读取二进制数据并编码
        base64_bytes = base64.b64encode(image_file.read())
        # 转换为字符串（bytes -> str）
        base64_str = base64_bytes.decode('utf-8')
    
    # 动态添加对应格式的Base64前缀（适配接口使用）
    if add_prefix:
        base64_str = f"data:image/{image_format};base64,{base64_str}"
    
    # 将Base64字符串保存到文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(base64_str)
    
    print(f"✅ Base64编码已保存到：{output_file_path}")
    print(f"📄 编码长度：{len(base64_str)} 字符")
    print(f"📌 图片格式：{image_format.upper()}")
    
    return base64_str

# 示例使用（修改这里的路径即可）
if __name__ == "__main__":
    # 配置参数：替换为你的图片路径（支持JPG/PNG）和输出文件路径
    INPUT_IMAGE_PATH = "bucket2.jpg"       # 你的图片路径（PNG/JPG都可以）
    OUTPUT_BASE64_FILE = "bucket2_base64.txt"  # 输出的Base64文本文件路径
    
    try:
        # 执行转换
        base64_result = image_to_base64(
            image_path=INPUT_IMAGE_PATH,
            output_file_path=OUTPUT_BASE64_FILE,
            add_prefix=True  # 保持True，适配接口
        )
        
        # 可选：打印前100个字符预览（Base64字符串很长，不建议全打印）
        print(f"\n🔍 Base64编码预览（前100字符）：{base64_result[:100]}...")
    
    except Exception as e:
        print(f"❌ 转换失败：{str(e)}")