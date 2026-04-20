"""
检测服务 - 业务逻辑封装
从 new_API.py 迁移
"""
import gc
from datetime import datetime
import json
import os
import time

import sam3
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import platform

# 导入推理模块
from sam3_ext.inference.test_picture_to_picture_optimizer_batch_inference import this_main as a_main, xyxy2cxcywh_norm
from sam3_ext.inference.test_picture_to_picture_optimizer import this_main as b_main
from sam3_ext.inference.word_picture_batch_inference import this_main as c_main
from sam3_ext.inference.test_words_picture import words_to_picture


sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
feature_file_path = "sam3_custom_objects.json"  # 特征保存文件
tmp_file_path = "tmp_load.json"
object_feature_map = {}
tmp_feature_map = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16


class the_API:
    def __init__(self):
        global object_feature_map, tmp_feature_map
        self.bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        self.model = build_sam3_image_model(bpe_path=self.bpe_path)
        self.model.to(device)
        self.model.eval()
        self.processor = Sam3Processor(
            model=self.model,
            resolution=1008,
            device=device,
            confidence_threshold=0.5
        )
        tmp_feature_map = self.load_feature_map(file_path=tmp_file_path)
        object_feature_map = self.load_feature_map(file_path=feature_file_path)

    def use_one_feature_to_find_other_picture_batch(self, base_image_path, batch_states, this_batch_size, location,
                                                    img_width, img_height, zengqiang_prompt):
        time_now = str(datetime.now())
        obj_name = "tmp" + time_now
        self.add_annotation(base_image_path, obj_name, location, img_width, img_height, zengqiang_prompt,
                            save_path=tmp_file_path)
        all_result = a_main(target_name=obj_name, zengqiang_prompt=zengqiang_prompt, processor=self.processor,
                            batch_states=batch_states, this_batch_size=this_batch_size, device=device)
        return all_result

    def use_one_feature_to_find_other_picture_batch_imgFeature(self, target_image_paths, batch_size=batch_size):
        batch_images = []
        valid_paths = []
        for path in target_image_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                valid_paths.append(path)
            except FileNotFoundError:
                print(f"⚠️  目标图 {path} 不存在，跳过！")
        if not batch_images:
            print("❌ 无有效目标图，退出识别！")
            yield [], 0
            return
        for i in range(0, len(batch_images), batch_size):
            if i + batch_size <= len(batch_images):
                batch_slice = batch_images[i:i + batch_size]
            else:
                batch_slice = batch_images[i:]
            current_batch_size = len(batch_slice)
            batch_states = []
            for img in batch_slice:
                state = self.processor.set_image(img)
                batch_states.append(state)
            yield batch_states, current_batch_size

    def use_one_feature_to_find_other_picture(self, base_image_path, target_image, pro_set_img, location, img_width,
                                              img_height, zengqiang_prompt):
        time_now = str(datetime.now())
        obj_name = "tmp" + time_now
        self.add_annotation(base_image_path, obj_name, location, img_width, img_height, zengqiang_prompt,
                            save_path=tmp_file_path)
        all_result = b_main(obj_name, self.processor, target_image, pro_set_img, zengqiang_prompt, device=device)
        return all_result

    def use_one_feature_to_find_other_picture_imgFeature(self, target_image_path):
        target_image = Image.open(target_image_path).convert("RGB")
        pro_set_img = self.processor.set_image(target_image)
        return target_image, pro_set_img

    def use_words_list_to_find_other_picture(self, batch_image_paths, batch_prompts, batch_zengqiang_prompt,
                                             batch_inference_states, final_output):
        result = c_main(
            batch_image_paths,
            batch_prompts,
            batch_zengqiang_prompt,
            self.processor,
            final_output,
            batch_inference_states,
            device=device
        )
        return result

    def use_words_list_to_find_other_picture_imgFeature(self, image_paths, batch_size=batch_size):
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_image_paths = image_paths[batch_start:batch_end]
            batch_idx = batch_start // batch_size + 1
            final_output = []
            batch_images = []
            valid_image_indices = []

            for idx, path in enumerate(batch_image_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_image_indices.append(idx)
                    final_output.append({"mask": [], "boxes": [], "scores": [], "name": []})
                except Exception as e:
                    print(f"   ❌ 加载图片失败 {path}: {e} | 批次内索引：{idx} | 占位空字典")
                    final_output.append({"mask": [], "boxes": [], "scores": [], "name": []})

            batch_inference_states = []
            for img in batch_images:
                state = self.processor.set_image(img)
                batch_inference_states.append(state)
            yield batch_image_paths, batch_inference_states, final_output

    def use_one_word_to_find_aim_picture_object(self, your_prompt, inference_state, current_img_size):
        zengqiang_prompt = your_prompt if your_prompt not in object_feature_map else object_feature_map[your_prompt][
            "en_semantic_prompt"]
        output = words_to_picture(self.processor, inference_state, your_prompt, zengqiang_prompt, current_img_size)
        return output

    def use_one_word_to_find_aim_picture_object_imgFeature(self, img_path):
        image = Image.open(img_path)
        current_img_size = [image.width, image.height]
        inference_state = self.processor.set_image(image)
        return inference_state, current_img_size

    def plot_pic_get_location(self, ref_path):
        """框选物体，返回像素框和图片尺寸（原逻辑不变）"""
        ref_image_path = ref_path
        try:
            ref_image = Image.open(ref_image_path).convert("RGB")
            ref_width, ref_height = ref_image.size
        except FileNotFoundError:
            print(f"不存在文件{ref_image_path}")
            return False
        plt.figure(figsize=(10, 8))
        plt.imshow(ref_image)
        plt.title("请用鼠标框选物体（点击左上角、右下角），按Enter确认")
        clicks = plt.ginput(2, timeout=60)
        if len(clicks) < 2:
            print("❌ 未完成框选，标注取消！")
            plt.close()
            return False
        plt.close()
        ref_box_xyxy = [clicks[0][0], clicks[0][1], clicks[1][0], clicks[1][1]]
        return ref_box_xyxy, ref_width, ref_height

    def add_annotation(self, ref_path, object_name, ref_box_xyxy, ref_width, ref_height,
                       en_semantic_prompt, save_path=feature_file_path):
        """
        新增/更新物体标注（适配新结构：参考信息+代表性框+英文提示词）
        """
        ref_image_path = ref_path

        existing_map = self.load_feature_map(save_path)

        if object_name in existing_map:
            obj_info = existing_map[object_name]
            if ref_box_xyxy not in obj_info["representative_boxes"]:
                obj_info["representative_boxes"].append(ref_box_xyxy)
            if en_semantic_prompt != obj_info["en_semantic_prompt"] and en_semantic_prompt:
                obj_info["en_semantic_prompt"] = en_semantic_prompt
        else:
            obj_info = {
                "ref_image_path": ref_image_path,
                "ref_image_size": [ref_width, ref_height],
                "representative_boxes": [ref_box_xyxy],
                "en_semantic_prompt": en_semantic_prompt
            }
        existing_map[object_name] = obj_info

        self.save_feature_map(existing_map, save_path)
        print(f"✅ 标注完成！{object_name} - 英文提示词：{en_semantic_prompt}")
        return True

    def load_feature_map(self, file_path):
        object_feature_map = {}
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
        return object_feature_map

    def save_feature_map(self, object_feature_map, file_path):
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(object_feature_map, f, ensure_ascii=False, indent=4)
            print(f"✅ 特征已保存到：{file_path}")
        except Exception as e:
            print(f"❌ 保存失败：{e}")

    def convert_old_feature_map(self, old_map):
        new_map = {}
        for name, feature in old_map.items():
            new_map[name] = [feature]
        return new_map

    def remove_tmp_file(self, file_path=tmp_file_path):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"'{file_path}'删除成功.")
            except Exception as e:
                print(f"删除 '{file_path}'. 失败: {e}")
        else:
            print(f"'{file_path}' 未知.")


def plot_figure(output, picture_path, save_path=None):
    """
    在指定图片上绘制mask、检测框、置信度、物体名称
    """
    img = cv2.imdecode(np.fromfile(picture_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片：{picture_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_copy = img_rgb.copy()

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255)
    ]
    mask_alpha = 0.3

    detections = []

    if isinstance(output, dict) and 'scores' in output and 'boxes' in output:
        masks = ''
        if 'masks' in output:
            masks = output['masks'].cpu().numpy() if hasattr(output['masks'], 'cpu') else output['masks']
        elif 'mask' in output:
            masks = output['mask'].cpu().numpy() if hasattr(output['mask'], 'cpu') else output['mask']
        boxes = output['boxes'].cpu().numpy() if hasattr(output['boxes'], 'cpu') else output['boxes']
        scores = output['scores'].cpu().numpy() if hasattr(output['scores'], 'cpu') else output['scores']

        for i in range(len(boxes)):
            names = output.get('name', [])
            if isinstance(names, (list, np.ndarray)) and len(names) > i:
                name = names[i]
            else:
                name = f'object_{i}'

            det = {
                'mask': masks[i].squeeze() if len(masks) > i else None,
                'box': boxes[i] if len(boxes) > i else None,
                'score': scores[i] if len(scores) > i else 0.0,
                'name': name
            }
            detections.append(det)

    elif isinstance(output, list):
        def flatten_list(lst):
            for item in lst:
                if isinstance(item, list):
                    yield from flatten_list(item)
                elif isinstance(item, dict):
                    yield item

        for det in flatten_list(output):
            detections.append({
                'mask': det.get('mask', None),
                'box': det.get('box', None),
                'score': det.get('score', 0.0),
                'name': det.get('name', 'unknown')
            })

    elif isinstance(output, dict):
        detections.append({
            'mask': output.get('mask', None),
            'box': output.get('box', None),
            'score': output.get('score', 0.0),
            'name': output.get('name', 'unknown')
        })

    unique_names = list(set([det['name'] for det in detections if det['name']]))
    name_to_color = {}
    for idx, name in enumerate(unique_names):
        name_to_color[name] = colors[idx % len(colors)]

    for idx, det in enumerate(detections):
        if det['box'] is None and det['mask'] is None:
            continue

        color = name_to_color[det['name']]
        score = det['score']
        name = det['name']

        if det['box'] is not None and len(det['box']) == 4:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            pil_img = Image.fromarray(img_copy)
            draw = ImageDraw.Draw(pil_img)

            font_path = ""
            if platform.system() == "Windows":
                font_path = "C:/Windows/Fonts/simhei.ttf"
            elif platform.system() == "Linux":
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            elif platform.system() == "Darwin":
                font_path = "/System/Library/Fonts/PingFang.ttc"

            try:
                font = ImageFont.truetype(font_path, 20)
            except:
                font = ImageFont.load_default()
                print(f"警告：无法加载中文字体，路径{font_path}不存在，使用默认字体（可能仍无法显示中文）")

            text = f"{name} | {score:.2f}"

            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            bg_x1, bg_y1 = x1, y1 - text_h - 12
            bg_x2, bg_y2 = x1 + text_w, y1
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)

            draw.text((x1, bg_y1), text, font=font, fill=(255, 255, 255))

            img_copy = np.array(pil_img)

    plt.figure(figsize=(12, 8))
    plt.imshow(img_copy)
    plt.axis('off')
    plt.title(f"Detection Results - {picture_path}", fontsize=12)
    plt.tight_layout()

    if save_path:
        img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        cv2.imencode('.jpg', img_bgr)[1].tofile(save_path)
        print(f"标注图片已保存至：{save_path}")

    plt.show()


if __name__ == "__main__":
    print("开始创建模型...")
    a = the_API()
    print("初始化模型成功")
