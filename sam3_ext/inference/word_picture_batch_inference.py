import copy

import torch
import json
import os
import warnings
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ======================== 全局配置 ========================
FEATURE_FILE_PATH = "sam3_custom_objects.json"
object_feature_map = {}
warnings.filterwarnings("ignore")


def convert_original_box_to_current_norm_box(original_pixel_box, original_img_size, current_img_size):
    """
    将参考图的像素框（xyxy）转换为当前图的归一化框（cxcywh）
    :param original_pixel_box: 参考图A的像素框 [x1,y1,x2,y2]
    :param original_img_size: 参考图A尺寸 [宽, 高]
    :param current_img_size: 当前图B尺寸 [宽, 高]
    :return: 归一化cxcywh框 [cx, cy, w, h]
    """
    x1, y1, x2, y2 = original_pixel_box
    orig_w, orig_h = original_img_size
    curr_w, curr_h = current_img_size

    orig_cx = (x1 + x2) / 2.0
    orig_cy = (y1 + y2) / 2.0
    orig_w_box = x2 - x1
    orig_h_box = y2 - y1

    ratio_cx = orig_cx / orig_w
    ratio_cy = orig_cy / orig_h
    ratio_w = orig_w_box / orig_w
    ratio_h = orig_h_box / orig_h

    return [ratio_cx, ratio_cy, ratio_w, ratio_h]


def convert_old_feature_map(old_map):
    """单字典→列表"""
    new_map = {}
    for name, feature in old_map.items():
        new_map[name] = [feature] if isinstance(feature, dict) else feature
    return new_map


def load_feature_map():
    """加载特征库（适配新结构：字典/列表）"""
    global object_feature_map
    object_feature_map = {}
    if os.path.exists(FEATURE_FILE_PATH):
        try:
            with open(FEATURE_FILE_PATH, "r", encoding="utf-8") as f:
                loaded_map = json.load(f)

            # 适配新结构
            if loaded_map:
                first_key = next(iter(loaded_map.keys()))
                first_val = loaded_map[first_key]
                # 旧结构：值为列表 → 直接用
                if isinstance(first_val, list):
                    object_feature_map = loaded_map
                # 新结构：值为字典 → 保留原结构
                elif isinstance(first_val, dict):
                    object_feature_map = loaded_map
                # 最旧结构：单字典 → 转列表
                else:
                    object_feature_map = convert_old_feature_map(loaded_map)
        except Exception as e:
            print(f"加载特征文件失败：{e}")
            object_feature_map = {}
    else:
        print(f"未检测到特征文件：{FEATURE_FILE_PATH}")
        object_feature_map = {}


def this_main(batch_image_paths, batch_prompts, batch_zengqiang_prompt, processor, final_output, batch_inference_states,
              device):
    """
    :param batch_image_paths: 当前批次的图片路径列表（非整体）
    :param batch_prompts: 文本提示词列表（如['truck','汽车轮子']）
    :param batch_zengqiang_prompt: 增强提示词列表（空值自动补全原提示词）
    :param processor: SAM3处理器
    :param final_output: 生成器返回的批次内占位列表（含无效图片的空字典）
    :param batch_inference_states: 批次内有效图片的inference_state列表
    :param device: 推理设备（cuda/cpu）
    :return: 批次内最终结果列表，格式与批次图片列表长度一致
    """
    # 1. 加载特征库 + 修复增强提示词补全逻辑
    load_feature_map()
    filled_zengqiang_prompts = []
    for idx, prompt in enumerate(batch_prompts):
        if batch_zengqiang_prompt[idx].strip():
            filled_zengqiang_prompts.append(batch_zengqiang_prompt[idx].strip())
        else:
            filled_zengqiang_prompts.append(prompt)

    # 2. 校验输入合法性
    if len(batch_inference_states) == 0:
        print("当前批次无有效图片")
        return final_output

    # 3. GPU批量推理核心逻辑（禁用梯度，节省显存）
    with torch.no_grad():
        valid_img_idx = 0  # 有效图片的索引（对应batch_inference_states）
        # 遍历批次内的final_output（已包含无效图片的占位）
        for placeholder_idx in range(len(final_output)):
            # 防止有效图片索引越界
            if valid_img_idx >= len(batch_inference_states):
                break

            # 修复：深拷贝state，避免污染原状态
            state = batch_inference_states[valid_img_idx]
            current_state = copy.deepcopy(state)
            img_masks = []
            img_boxes = []
            img_scores = []
            img_names = []

            # 遍历所有提示词，逐个推理
            for prompt_idx, prompt in enumerate(batch_prompts):
                try:
                    current_prompt = filled_zengqiang_prompts[prompt_idx]
                    # 每个提示词独立拷贝state，避免相互污染
                    prompt_state = copy.deepcopy(current_state)
                    # 情况1：提示词在特征库中（几何+语义提示）
                    if prompt in object_feature_map:
                        obj_feature = object_feature_map[prompt]
                        # 获取当前图片尺寸（从state提取，[宽, 高]）
                        current_img_size = [prompt_state["original_width"], prompt_state["original_height"]]

                        # 设置语义提示（优先用特征库的英文提示，否则用增强提示词）
                        en_prompt = obj_feature.get("en_semantic_prompt", current_prompt)
                        prompt_state = processor.set_text_prompt(state=prompt_state, prompt=en_prompt)

                        # 批量添加几何提示框
                        representative_boxes = obj_feature.get("representative_boxes", [])
                        original_img_size = obj_feature.get("ref_image_size", [1, 1])
                        for pixel_box in representative_boxes:
                            norm_box = convert_original_box_to_current_norm_box(
                                original_pixel_box=pixel_box,
                                original_img_size=original_img_size,
                                current_img_size=current_img_size
                            )
                            norm_box_tensor = torch.tensor([norm_box], dtype=torch.float32).to(device)
                            prompt_state = processor.add_geometric_prompt(
                                box=norm_box_tensor,
                                label=True,
                                state=prompt_state
                            )

                    # 情况2：提示词不在特征库中（仅语义提示）
                    else:
                        prompt_state = processor.set_text_prompt(state=prompt_state, prompt=current_prompt)

                    # 核心修复：添加SAM3推理步骤（没有这一步就没有masks/boxes/scores）
                    # 修复：正确提取推理结果（根据SAM3 processor的返回结构调整）
                    # 优先从predictions提取，兼容不同版本的SAM3
                    predictions = prompt_state
                    masks = predictions.get("masks", torch.tensor([]))
                    boxes = predictions.get("boxes", torch.tensor([]))
                    scores = predictions.get("scores", torch.tensor([]))

                    # 过滤空结果
                    if len(scores) == 0:
                        continue

                    # 张量转numpy（延迟转CPU，提升效率）
                    masks_np = masks.cpu().numpy().squeeze(axis=1) if len(masks.shape) > 1 else masks.cpu().numpy()
                    boxes_np = boxes.cpu().numpy().round(2)
                    scores_list = scores.cpu().tolist()

                    # 填充当前提示词的结果
                    for i in range(len(scores_list)):
                        img_masks.append(masks_np[i] if len(masks_np.shape) > 0 else masks_np)
                        img_boxes.append(boxes_np[i])
                        img_scores.append(scores_list[i])
                        img_names.append(prompt)

                except Exception as e:
                    print(f"⚠️  提示词[{prompt}]推理失败：{type(e).__name__} - {str(e)}")
                    continue

            # 填充有效图片的推理结果到final_output
            final_output[placeholder_idx] = {
                "boxes": img_boxes,
                "scores": img_scores,
                "name": img_names
            }
            valid_img_idx += 1
    # 4. 最终校验：保证输出长度与当前批次图片列表一致
    if len(final_output) < len(batch_image_paths):
        diff = len(batch_image_paths) - len(final_output)
        for _ in range(diff):
            final_output.append({ "boxes": [], "scores": [], "name": []})

    return final_output
