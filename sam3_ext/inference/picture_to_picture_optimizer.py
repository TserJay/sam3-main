import torch
import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ======================== 全局配置 ========================
FEATURE_FILE_PATH = "sam3_custom_objects.json"
object_feature_map = {}
BOX_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']


# ======================== 持久化工具函数 ========================
def convert_old_feature_map(old_map):
    new_map = {}
    for name, feature in old_map.items():
        new_map[name] = [feature]
    return new_map


def load_feature_map():
    global object_feature_map
    if os.path.exists(FEATURE_FILE_PATH):
        try:
            with open(FEATURE_FILE_PATH, "r", encoding="utf-8") as f:
                loaded_map = json.load(f)
            if loaded_map and isinstance(list(loaded_map.values())[0], dict):
                object_feature_map = convert_old_feature_map(loaded_map)
            else:
                object_feature_map = loaded_map
        except Exception as e:
            print(f"⚠️  加载特征文件失败（文件损坏），将创建新文件：{e}")
            object_feature_map = {}
    else:
        print(f"✅ 首次运行，未检测到特征文件，将创建：{FEATURE_FILE_PATH}")
        object_feature_map = {}


def save_feature_map():
    try:
        with open(FEATURE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(object_feature_map, f, ensure_ascii=False, indent=4)
        print(f"✅ 自定义物体特征已保存到：{FEATURE_FILE_PATH}")
    except Exception as e:
        print(f"❌ 保存特征文件失败：{e}")


# ======================== 辅助工具函数 ========================
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


def draw_bbox(ax, box_xyxy, color="red", linewidth=2, label=None, is_best=False):
    x1, y1, x2, y2 = box_xyxy
    width = x2 - x1
    height = y2 - y1
    line_width = linewidth + 1 if is_best else linewidth
    rect = Rectangle((x1, y1), width, height,
                     fill=False, color=color, linewidth=line_width, label=label)
    ax.add_patch(rect)
    if label:
        label_text = f"★{label}" if is_best else label
        ax.text(x1, y1 - 10, label_text, fontsize=8, color=color,
                bbox=dict(facecolor='white', alpha=0.8, pad=1))


# ======================== 核心识别函数（修改日志输出） ========================
def recognize_object(model, processor, image, mode, param):
    """
    双模式物体识别（仅保留最优结果的可视化，仍收集所有结果用于筛选）：
    - mode="custom"：param为自定义物体名称
    - mode="text"：param为文本描述
    返回：(all_results, best_result)
        - all_results：所有识别结果列表（内部使用）
        - best_result：置信度最高的结果
    """
    img_width, img_height = image.size
    all_results = []

    if mode == "custom":
        feature_list = object_feature_map[param]
        for feature_idx, feature in enumerate(feature_list):
            state_single = processor.add_geometric_prompt(
                box=feature["norm_box"],
                label=True,
                state=processor.set_image(image)
            )
            masks = state_single["masks"]
            boxes = state_single["boxes"]
            scores = state_single["scores"]

            if len(scores) > 0:
                # 仍收集所有结果（用于筛选最优），但日志只打印关键信息
                for idx in range(len(scores)):
                    mask = masks[idx].cpu().numpy()
                    mask = np.squeeze(mask)
                    box = boxes[idx].cpu().numpy().round(2)
                    score = scores[idx].cpu().item()
                    all_results.append({
                        "mask": mask,
                        "box": box,
                        "score": score,
                        "feature_idx": feature_idx,
                        "result_idx": idx,
                        "name": param
                    })

    elif mode == "text":
        state = processor.set_image(image)
        state = processor.set_text_prompt(
            prompt=param,
            state=state
        )
        masks = state["masks"]
        boxes = state["boxes"]
        scores = state["scores"]

        if len(scores) > 0:
            for idx in range(len(scores)):
                mask = masks[idx].cpu().numpy()
                mask = np.squeeze(mask)
                box = boxes[idx].cpu().numpy().round(2)
                score = scores[idx].cpu().item()
                all_results.append({
                    "mask": mask,
                    "box": box,
                    "score": score,
                    "feature_idx": -1,
                    "result_idx": idx,
                    "name": param
                })
            # 修改：只打印文本模式的最优结果
            best_score = max(scores).cpu().item()
    if len(all_results) == 0:
        return [], None

    # 筛选最优结果
    best_result = max(all_results, key=lambda x: x["score"])
    return all_results, best_result


# ======================== 增加图例取特征函数 ========================
def add_annotation(ref_path,object_name):
    ref_image_path = ref_path
    try:
        ref_image = Image.open(ref_image_path).convert("RGB")
        ref_width, ref_height = ref_image.size
    except FileNotFoundError:
        return False
    plt.figure(figsize=(10, 8))
    plt.imshow(ref_image)
    plt.title("请用鼠标框选物体（点击左上角、右下角），按Enter确认")
    clicks = plt.ginput(2, timeout=60)
    if len(clicks) < 2:
        print("❌ 未完成框选，标注取消！")
        plt.close()
        return False
    ref_box_xyxy = [clicks[0][0], clicks[0][1], clicks[1][0], clicks[1][1]]
    ref_box_cxcywh_norm = xyxy2cxcywh_norm(ref_box_xyxy, ref_width, ref_height)
    plt.close()


    new_feature = {
        "norm_box": ref_box_cxcywh_norm,
        "pixel_box": ref_box_xyxy,
        "ref_image_path": ref_image_path,
        "ref_image_size": [ref_width, ref_height],
        "annotate_time": str(np.datetime64('now'))
    }

    object_feature_map[object_name] = [new_feature]

    save_feature_map()
    return True


# ======================== 主流程 ========================
def this_main(target_image_paths,op_choice,obj_name):
    '''
    op_choice:选择操作，1. 新增/追加物体标注（多图提升精度）、2. 识别物体
    target_image_paths:目标识别文件路径,当op_choice=='1'时，该参数为需要添加的目标范式图片；当op_choice=='2'时，该参数为需要识别的图片
    obj_name:当op_choice=='1'时,为添加目标范式物体的名称；当op_choice=='2'时，为识别图片中的目标物体名称
    '''
    load_feature_map()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model()
    model.to(device)
    processor = Sam3Processor(
        model=model,
        resolution=1008,
        device=device,
        confidence_threshold=0.5
    )
    model.eval()


    if op_choice == "1":
        a=add_annotation(target_image_paths,obj_name)
        if a:
            return "添加完成"
        return "添加失败"

    elif op_choice == "2":
        target_name = obj_name
        if target_name in object_feature_map:
            recog_param = target_name
            mode = "custom"
        else:
            recog_param = target_name
            mode = "text"
        # 遍历目标图识别
        target_image = Image.open(target_image_paths).convert("RGB")

        # 调用识别函数（仍保留all_results用于筛选最优）
        all_results, best_result = recognize_object(model, processor, target_image, mode, recog_param)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 修改：调用仅显示最优锚框的可视化函数

        return best_result
