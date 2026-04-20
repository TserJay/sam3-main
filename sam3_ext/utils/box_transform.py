"""
坐标转换工具函数
"""
import torch
from typing import List, Union


def xyxy2cxcywh_norm(box_xyxy: List[float], img_width: int, img_height: int) -> List[float]:
    """
    将[x1,y1,x2,y2]格式的像素坐标，转换为归一化的[cx, cy, w, h]格式
    
    Args:
        box_xyxy: [x1, y1, x2, y2] 像素坐标
        img_width: 图片宽度
        img_height: 图片高度
    
    Returns:
        归一化后的 [cx, cy, w, h]
    """
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    cx /= img_width
    cy /= img_height
    w /= img_width
    h /= img_height
    return [cx, cy, w, h]


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    将[cx, cy, w, h]格式转换为[x1, y1, x2, y2]格式
    
    Args:
        boxes: 张量，shape=[N,4]
    
    Returns:
        转换后的张量，shape=[N,4]
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


def convert_original_box_to_current_norm_box(
    original_pixel_box: List[float],
    original_img_size: List[int],
    current_img_size: List[int]
) -> List[float]:
    """
    将原始图片的像素框（xyxy）转换为适配当前图片的归一化框（cxcywh）
    
    Args:
        original_pixel_box: 参考图A的像素框 [x1,y1,x2,y2]
        original_img_size: 参考图A尺寸 [宽, 高]
        current_img_size: 目标图B尺寸 [宽, 高]
    
    Returns:
        归一化cxcywh框 [cx, cy, w, h]
    """
    try:
        x1, y1, x2, y2 = original_pixel_box
        orig_w, orig_h = original_img_size
        curr_w, curr_h = current_img_size

        # 转cxcywh
        orig_cx = (x1 + x2) / 2.0
        orig_cy = (y1 + y2) / 2.0
        orig_w_box = x2 - x1
        orig_h_box = y2 - y1

        # 计算相对比例（适配当前图片）
        ratio_cx = orig_cx / orig_w
        ratio_cy = orig_cy / orig_h
        ratio_w = orig_w_box / orig_w
        ratio_h = orig_h_box / orig_h

        return [ratio_cx, ratio_cy, ratio_w, ratio_h]
    except Exception as e:
        print(f"框转换失败：{e}")
        return [0.0, 0.0, 0.0, 0.0]
