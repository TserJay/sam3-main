from PIL import Image
import os

def png_to_jpg_single(input_path, output_path, background_color=(255, 255, 255)):
    """
    将单张PNG图片转换为JPG格式
    :param input_path: 输入PNG图片的路径（如："test.png"）
    :param output_path: 输出JPG图片的路径（如："test.jpg"）
    :param background_color: 填充透明区域的背景色，默认白色(255,255,255)
    """
    try:
        # 打开PNG图片
        img = Image.open(input_path)
        
        # 如果PNG有透明通道（RGBA模式），需要转换为RGB并填充背景色
        if img.mode == 'RGBA':
            # 创建和原图大小一致的白色背景画布
            bg = Image.new('RGB', img.size, background_color)
            # 将PNG图片粘贴到背景画布上（透明区域会显示背景色）
            bg.paste(img, mask=img.split()[3])  # mask参数指定透明通道
            img = bg
        elif img.mode == 'P':  # 处理索引色模式的PNG
            img = img.convert('RGB')
        
        # 保存为JPG格式，quality控制画质（1-95，越高画质越好）
        img.save(output_path, 'JPEG', quality=95)
        print(f"转换成功！已保存至：{output_path}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
    except Exception as e:
        print(f"转换失败：{str(e)}")

# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 替换为你的PNG路径和想要输出的JPG路径
    input_png = "/mnt/pengyi-sam3/sam3-main/assets/task_images/bucket/bucket2.png"   # 输入PNG文件
    output_jpg = "bucket2.jpg"  # 输出JPG文件
    png_to_jpg_single(input_png, output_jpg)