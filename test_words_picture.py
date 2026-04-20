import json
import os
import torch

FEATURE_FILE_PATH = "sam3_custom_objects.json"
object_feature_map = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

def convert_original_box_to_current_norm_box(original_pixel_box, original_img_size, current_img_size):
    """
    将原始图片的像素框（xyxy）转换为适配当前图片的归一化框（cxcywh）
    """
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


def words_to_picture(processor, inference_state, the_prompt,en_prompt, current_img_size):#en_prompt:增强提示词
    """
    全图查找图片A中框选的目标（英文提示词+代表性框）
    逻辑：
    1. 若提示词在特征库中 → 英文文本提示（全图语义） + 批量添加代表性框（视觉辅助）
    2. 若不在/结构异常 → 自动生成英文提示词检索
    """
    # 加载特征映射
    load_success = load_feature_map()
    if not load_success and not object_feature_map:
        # 无特征库，自动生成英文提示词检索
        output = processor.set_text_prompt(state=inference_state, prompt=en_prompt)
        return output

    if the_prompt in object_feature_map:
        try:
            obj_info = object_feature_map[the_prompt]
            if isinstance(obj_info, list):
                feature_list = obj_info
                if not isinstance(feature_list, list) or len(feature_list) == 0:
                    raise ValueError("特征数据格式错误，非有效列表")
                output = processor.set_text_prompt(state=inference_state, prompt=en_prompt)
                for feature in feature_list:
                    original_pixel_box = feature["pixel_box"]
                    original_img_size = feature["ref_image_size"]
                    norm_box = convert_original_box_to_current_norm_box(
                        original_pixel_box=original_pixel_box,
                        original_img_size=original_img_size,
                        current_img_size=current_img_size
                    )
                    output = processor.add_geometric_prompt(
                        box=norm_box,
                        label=True,
                        state=output
                    )
            else:
                if not all(key in obj_info for key in ["representative_boxes", "ref_image_size", "en_semantic_prompt"]):
                    raise KeyError("新特征结构缺失关键字段：representative_boxes/ref_image_size/en_semantic_prompt")

                # 步骤1：英文文本提示 → 全图语义检索（SAM3能理解）
                en_prompt = obj_info["en_semantic_prompt"]
                output = processor.set_text_prompt(state=inference_state, prompt=en_prompt)

                # 步骤2：批量添加代表性框 → 视觉特征增强（非限定位置）
                representative_boxes = obj_info["representative_boxes"]
                original_img_size = obj_info["ref_image_size"]
                for box in representative_boxes:
                    # 转换为适配当前图片的norm_box
                    norm_box = convert_original_box_to_current_norm_box(
                        original_pixel_box=box,
                        original_img_size=original_img_size,
                        current_img_size=current_img_size
                    )
                    # 批量添加几何提示词（增强视觉匹配）
                    output = processor.add_geometric_prompt(
                        box=norm_box,
                        label=True,
                        state=output
                    )
        except KeyError as e:
            print(f"⚠️  特征结构缺失字段 {e}")
            output = processor.set_text_prompt(state=inference_state, prompt=en_prompt)
        except Exception as e:
            print(f"❌ 多框辅助推理失败{e}")
            output = processor.set_text_prompt(state=inference_state, prompt=en_prompt)
    else:
        output = processor.set_text_prompt(state=inference_state, prompt=en_prompt)
    return output


def load_feature_map():
    """加载特征库（不变）"""
    global object_feature_map
    object_feature_map = {}
    if os.path.exists(FEATURE_FILE_PATH):
        try:
            with open(FEATURE_FILE_PATH, "r", encoding="utf-8") as f:
                object_feature_map = json.load(f)
            return True
        except Exception as e:
            print(f"加载特征库失败：{e}")
            object_feature_map = {}
            return False
    else:
        print("特征库文件不存在")
        return False


# 兼容旧版特征库的函数（保留，防止后续有旧数据）
def convert_old_feature_map(old_map):
    new_map = {}
    for name, feature in old_map.items():
        new_map[name] = [feature] if isinstance(feature, dict) else feature
    return new_map