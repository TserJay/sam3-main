"""
图像处理工具函数
"""
import os
import base64
import tempfile
from datetime import datetime
from typing import Optional
import random


def decode_base64_to_image(b64_str: str, suffix: str = ".jpg") -> str:
    """
    将Base64编码的图片字符串解码并保存为临时文件
    
    Args:
        b64_str: Base64字符串（支持带前缀如data:image/jpeg;base64,）
        suffix: 临时文件后缀
    
    Returns:
        临时文件路径
    """
    try:
        # 移除Base64前缀（如果有）
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        # 解码Base64
        img_data = base64.b64decode(b64_str)
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(img_data)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        raise ValueError(f"Base64解码失败: {str(e)}")


def save_image_with_timestamp(
    image_data: bytes,
    save_dir: str,
    object_name: Optional[str] = None,
    suffix: str = ".jpg"
) -> str:
    """
    将图片数据保存到指定目录，使用时间戳命名
    
    Args:
        image_data: 图片二进制数据
        save_dir: 保存目录
        object_name: 对象名称，用于生成文件名
        suffix: 文件后缀
    
    Returns:
        保存的文件路径
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        
        if object_name:
            safe_object_name = "".join(c for c in object_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
            filename = f"{safe_object_name}_{timestamp}_{random_str}{suffix}"
        else:
            filename = f"image_{timestamp}_{random_str}{suffix}"
        
        file_path = os.path.join(save_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        return file_path
    except Exception as e:
        raise ValueError(f"图片保存失败: {str(e)}")
