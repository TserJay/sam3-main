import json
import os
import numpy as np
import torch

# ======================== 全局配置 ========================
FEATURE_FILE_PATH = "tmp_load.json"
object_feature_map = {}


# ======================== 持久化工具函数 ========================
def convert_old_feature_map(old_map):
    new_map = {}
    for name, feature in old_map.items():
        new_map[name] = [feature]
    return new_map


def load_feature_map(file_path=FEATURE_FILE_PATH):
    global object_feature_map
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                loaded_map = json.loads(content) if content else {}
            object_feature_map = loaded_map
        except Exception as e:
            print(f"加载特征文件失败（文件损坏），创建新文件：{e}")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
            object_feature_map = {}
    else:
        print(f"创建特征文件：{file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
    return True


def convert_original_box_to_current_norm_box(original_pixel_box, original_img_size, current_img_size):
    """
    将参考图的像素框（xyxy）转换为当前图的归一化框（cxcywh）
    :param original_pixel_box: 参考图A的像素框 [x1,y1,x2,y2]
    :param original_img_size: 参考图A尺寸 [宽, 高]
    :param current_img_size: 当前图B尺寸 [宽, 高]
    :return: 归一化cxcywh框 [cx, cy, w, h]
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



# ======================== 核心识别函数 ========================
def recognize_object(processor, target_image,pro_set_img, object_name, en_prompt, device):
    """
    核心识别函数（适配新JSON结构 + GPU优化 + 无全局变量依赖）
    :param processor: SAM3处理器
    :param pro_set_img: 目标图的inference_state（processor.set_image后的结果）
    :param object_name: 目标物体名（中文，如"汽车轮子"）
    :param en_prompt: 增强提示词（英文）
    :param device: 推理设备（cuda/cpu）
    :return: (all_results, best_result)
        - all_results：所有识别结果列表
        - best_result：置信度最高的结果（None表示无结果）
    """
    all_results = []
    # 获取当前图片尺寸（从inference_state提取）
    current_img_size = target_image.size

    obj_feature = object_feature_map[object_name]
    try:
        # 步骤1：先设置英文语义提示（SAM3核心）
        state = processor.set_text_prompt(
            state=pro_set_img,
            prompt=en_prompt
        )
        # 步骤2：批量添加几何提示（代表性框）
        representative_boxes = obj_feature.get("representative_boxes", [])
        original_img_size = obj_feature.get("ref_image_size", [1, 1])

        for feature_idx, pixel_box in enumerate(representative_boxes):
            # 转换参考框到当前图片的归一化框
            norm_box = convert_original_box_to_current_norm_box(
                original_pixel_box=pixel_box,
                original_img_size=original_img_size,
                current_img_size=current_img_size
            )
            # 转为GPU张量（避免循环内重复传输）
            norm_box_tensor = torch.tensor([norm_box], dtype=torch.float32).to(device)
            # 添加几何提示
            state = processor.add_geometric_prompt(
                box=norm_box_tensor,
                label=True,
                state=state  # 基于当前state更新
            )
            # 提取结果
            masks = state["masks"]
            boxes = state["boxes"]
            scores = state["scores"]

            # 解析结果（延迟转CPU，提升效率）
            if len(scores) > 0:
                masks_np = masks.cpu().numpy().squeeze(axis=1)
                boxes_np = boxes.cpu().numpy().round(2)
                scores_list = scores.cpu().tolist()

                for res_idx in range(len(scores_list)):
                    all_results.append({
                        "mask": masks_np[res_idx] if len(masks_np.shape) > 0 else masks_np,
                        "box": boxes_np[res_idx],
                        "score": scores_list[res_idx],
                        "feature_idx": feature_idx,
                        "result_idx": res_idx,
                        "name": "您勾选物体"
                    })
    except Exception as e:
        print(f"结构检索失败：{e}，降级为纯文本检索")
        # 降级为纯文本检索
        state = processor.set_text_prompt(
            state=pro_set_img,
            prompt=en_prompt
        )
        masks = state["masks"]
        boxes = state["boxes"]
        scores = state["scores"]
        if len(scores) > 0:
            for idx in range(len(scores)):
                all_results.append({
                    "mask": masks[idx].cpu().numpy().squeeze(),
                    "box": boxes[idx].cpu().numpy().round(2),
                    "score": scores[idx].cpu().item(),
                    "feature_idx": -1,
                    "result_idx": idx,
                    "name": en_prompt
                })

    return all_results


# ======================== 主流程函数 ========================
def this_main(obj_name, processor, target_image,pro_set_img, en_prompt,device):
    '''
    主调用函数（适配新JSON结构，无全局变量依赖）
    :param obj_name: 目标物体名称（中文，如"汽车轮子"）
    :param processor: SAM3处理器
    :param pro_set_img: 目标图的inference_state（processor.set_image后的结果）
    :param en_prompt: 增强提示词（英文）
    :return: (all_results, best_result)
    '''
    # 加载特征库
    load_feature_map()
    # 调用修复后的识别函数
    all_results = recognize_object(
        processor=processor,
        target_image=target_image,
        pro_set_img=pro_set_img,
        object_name=obj_name,
        en_prompt=en_prompt,
        device=device
    )
    return all_results
