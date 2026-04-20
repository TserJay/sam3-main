import torch
import json
import os
import warnings

# ======================== 全局配置 ========================
tmp_file_path="tmp_load.json"
tmp_feature_map = {}
# ======================== 关闭所有无关警告 ========================
warnings.filterwarnings("ignore")




def xyxy2cxcywh_norm(box_xyxy, img_width, img_height):
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


def box_cxcywh_to_xyxy(boxes):
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


def convert_old_feature_map(old_map):
    new_map = {}
    for name, feature in old_map.items():
        new_map[name] = [feature]
    return new_map


def load_feature_map():
    global tmp_feature_map
    if os.path.exists(tmp_file_path):
        try:
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                loaded_map = json.load(f)
            if loaded_map and isinstance(list(loaded_map.values())[0], dict):
                tmp_feature_map = convert_old_feature_map(loaded_map)
            else:
                tmp_feature_map = loaded_map
        except Exception as e:
            tmp_feature_map = {}
    else:
        tmp_feature_map = {}


def convert_original_box_to_current_norm_box(original_pixel_box, original_img_size, current_img_size):
    """
    将原始图片的像素框（xyxy）转换为适配当前图片的归一化框（cxcywh）
    :param original_pixel_box: 参考图A的像素框 [x1,y1,x2,y2]
    :param original_img_size: 参考图A尺寸 [宽, 高]
    :param current_img_size: 目标图B尺寸 [宽, 高]
    :return: 归一化cxcywh框 [cx, cy, w, h]
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


def batch_recognize_object(processor, this_batch_size, batch_states, object_name, zengqiang_prompt, device):
    """
    批量识别：基于GPU的高效批量推理
    :param processor: SAM3处理器
    :param this_batch_size: 批次大小
    :param batch_states: 批量图片的inference_state列表（每张图的初始状态）
    :param object_name: 目标物体名（中文）
    :param zengqiang_prompt: 增强提示词（预留，优先用特征库中的英文提示词）
    :param device: 推理设备
    :return: batch_all_results（所有结果）, batch_best_results（最优结果）
    """
    batch_size = this_batch_size
    batch_all_results = [[] for _ in range(batch_size)]

    # 1. 校验特征库是否存在目标物体
    if object_name not in tmp_feature_map:
        en_prompt = zengqiang_prompt
        for img_idx in range(batch_size):
            state = processor.set_text_prompt(state=batch_states[img_idx], prompt=en_prompt)
            batch_states[img_idx] = state
            # 提取结果
            masks = state["masks"]
            boxes = state["boxes"]
            scores = state["scores"]
            if len(scores) > 0:
                for res_idx in range(len(scores)):
                    batch_all_results[img_idx].append({
                        "mask": masks[res_idx].cpu().numpy().squeeze(),
                        "box": boxes[res_idx].cpu().numpy().round(2),
                        "score": scores[res_idx].cpu().item(),
                        "feature_idx": -1,
                        "result_idx": res_idx,
                        "name": "您标注的物体"
                    })

    # 2. 解析特征库
    obj_feature = tmp_feature_map[object_name]
    norm_boxes = []  # 存储归一化后的框（GPU张量）
    en_prompt = zengqiang_prompt

    # 2.1 字典（含representative_boxes/en_semantic_prompt）
    if isinstance(obj_feature, dict):
        try:
            # 提取关键信息
            representative_boxes = obj_feature["representative_boxes"]
            original_img_size = obj_feature["ref_image_size"]
            en_prompt = obj_feature.get("en_semantic_prompt", en_prompt)

            # 批量转换框为归一化格式（适配每张图的尺寸）
            for img_idx in range(batch_size):
                # 从inference_state提取当前图片尺寸
                current_img_size = [batch_states[img_idx]["image_size"][1], batch_states[img_idx]["image_size"][0]]
                # 转换所有代表性框
                img_norm_boxes = []
                for box in representative_boxes:
                    norm_box = convert_original_box_to_current_norm_box(
                        original_pixel_box=box,
                        original_img_size=original_img_size,
                        current_img_size=current_img_size
                    )
                    img_norm_boxes.append(norm_box)
                # 转为GPU张量（批量处理，减少传输）
                if img_norm_boxes:
                    norm_box_tensor = torch.tensor(img_norm_boxes, dtype=torch.float32).to(device)
                    norm_boxes.append(norm_box_tensor)
                else:
                    norm_boxes.append(None)
        except Exception as e:
            print(f"⚠️  解析新特征结构失败：{e}，降级为文本检索")
            en_prompt = zengqiang_prompt
            # 批量设置文本提示
            for img_idx in range(batch_size):
                state = processor.set_text_prompt(state=batch_states[img_idx], prompt=en_prompt)
                batch_states[img_idx] = state

    # 2.2 旧结构：列表（多框）
    elif isinstance(obj_feature, list):
        en_prompt = zengqiang_prompt
        # 提取所有norm_box并转为GPU张量
        for feature in obj_feature:
            if "norm_box" in feature:
                norm_box = torch.tensor([feature["norm_box"]], dtype=torch.float32).to(device)
                norm_boxes.append(norm_box)

    # 3. 核心推理（GPU加速）
    with torch.no_grad():  # 禁用梯度计算，节省GPU显存
        # 3.1 先批量设置文本提示（语义检索，GPU加速）
        if en_prompt:
            for img_idx in range(batch_size):
                state = processor.set_text_prompt(state=batch_states[img_idx], prompt=en_prompt)
                batch_states[img_idx] = state

        # 3.2 批量添加几何提示并提取结果
        for img_idx in range(batch_size):
            current_state = batch_states[img_idx]
            # 处理当前图片的所有几何提示框
            if isinstance(obj_feature, dict) and norm_boxes[img_idx] is not None:
                # 新结构：批量添加当前图片的所有代表性框
                for box_idx, box in enumerate(norm_boxes[img_idx]):
                    state = processor.add_geometric_prompt(
                        box=box.unsqueeze(0),  # 增加维度适配API
                        label=True,
                        state=current_state
                    )
                    current_state = state
            elif isinstance(obj_feature, list) and norm_boxes:
                # 旧结构：遍历所有特征框
                for feature_idx, box_tensor in enumerate(norm_boxes):
                    state = processor.add_geometric_prompt(
                        box=box_tensor,
                        label=True,
                        state=current_state
                    )
                    current_state = state

            # 4. 提取GPU上的结果，延迟转CPU（提升效率）
            masks = current_state["masks"]
            boxes = current_state["boxes"]
            scores = current_state["scores"]

            # 解析结果（仅此时转回CPU）
            if len(scores) > 0:
                # 批量转CPU并处理，减少循环次数
                masks_np = masks.cpu().numpy().squeeze(axis=1)  # 批量转CPU
                boxes_np = boxes.cpu().numpy().round(2)
                scores_list = scores.cpu().tolist()

                for res_idx in range(len(scores_list)):
                    batch_all_results[img_idx].append({
                        "mask": masks_np[res_idx] if len(masks_np.shape) > 0 else masks_np,
                        "box": boxes_np[res_idx],
                        "score": scores_list[res_idx],
                        "feature_idx": res_idx,
                        "result_idx": res_idx,
                        "name": "您标注的物体"
                    })
    return batch_all_results

# ======================== 主流程========================
def this_main(target_name,zengqiang_prompt,processor,batch_states,this_batch_size,device):
    '''
    target_image_paths:批量需要识别的图片文件集；
    BATCH_SIZE：每一批次大小
    target_name：需要找到的物体名词
    '''
    load_feature_map()
    recog_param = target_name
    # 调用简化版批量识别函数
    batch_all_results= batch_recognize_object(processor, this_batch_size, batch_states,
                                                                   recog_param, zengqiang_prompt, device)
    return  batch_all_results