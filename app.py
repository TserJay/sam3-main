import gc
import base64
from datetime import datetime
import json
import os
import time
import tempfile
import random

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np

# 导入您的原有模块
from test_picture_to_picture_optimizer_batch_inference import this_main as a_main, xyxy2cxcywh_norm
from test_picture_to_picture_optimizer import this_main as b_main
from word_picture_batch_inference import this_main as c_main
from test_words_picture import words_to_picture

import sam3
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import cv2
from PIL import Image
import platform

app = Flask(__name__)
CORS(app)  # 允许跨域请求

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
feature_file_path = "sam3_custom_objects.json"  # 特征保存文件
tmp_file_path = "tmp_load.json"
object_feature_map = {}
tmp_feature_map = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

# 图片保存目录
IMAGE_SAVE_DIR = "/mnt/pengyi-sam3/sam3-main/assets/images"

# 确保图片保存目录存在
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# 初始化API实例
print("开始创建模型...")
api_instance = None

def get_api_instance():
    global api_instance
    if api_instance is None:
        api_instance = The_API()
        print("初始化模型成功")
    return api_instance


class The_API:
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
        ref_image_path = ref_path
        try:
            ref_image = Image.open(ref_image_path).convert("RGB")
            ref_width, ref_height = ref_image.size
        except FileNotFoundError:
            print(f"不存在文件{ref_image_path}")
            return False
        # 注意：在API中无法直接使用plt.ginput，需要前端传递坐标
        return None

    def add_annotation(self, ref_path, object_name, ref_box_xyxy, ref_width, ref_height,
                       en_semantic_prompt, save_path=feature_file_path):
        # 将图片复制到永久保存目录
        if ref_path and os.path.exists(ref_path):
            # 生成唯一的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
            ext = os.path.splitext(ref_path)[1] or '.jpg'
            new_filename = f"{object_name}_{timestamp}_{random_str}{ext}"
            new_file_path = os.path.join(IMAGE_SAVE_DIR, new_filename)
            
            # 复制文件
            import shutil
            shutil.copy2(ref_path, new_file_path)
            
            # 更新图片路径为永久保存路径
            ref_image_path = new_file_path
            print(f"✅ 图片已保存到: {new_file_path}")
        else:
            ref_image_path = ref_path
            print(f"⚠️  警告：原图片路径无效，使用原路径: {ref_path}")

        existing_map = self.load_feature_map(save_path)

        if object_name in existing_map:
            obj_info = existing_map[object_name]
            if ref_box_xyxy not in obj_info["representative_boxes"]:
                obj_info["representative_boxes"].append(ref_box_xyxy)
            if en_semantic_prompt != obj_info["en_semantic_prompt"] and en_semantic_prompt:
                obj_info["en_semantic_prompt"] = en_semantic_prompt
            # 更新图片路径
            obj_info["ref_image_path"] = ref_image_path
            obj_info["ref_image_size"] = [ref_width, ref_height]
        else:
            obj_info = {
                "ref_image_path": ref_image_path,
                "ref_image_size": [ref_width, ref_height],
                "representative_boxes": [ref_box_xyxy],
                "en_semantic_prompt": en_semantic_prompt
            }
        existing_map[object_name] = obj_info

        self.save_feature_map(existing_map, save_path)
        print(f"✅ 标注完成！{object_name} - 英文提示词：{en_semantic_prompt} - 图片路径：{ref_image_path}")
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

    def remove_tmp_file(self, file_path=tmp_file_path):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"'{file_path}'删除成功.")
            except Exception as e:
                print(f"删除 '{file_path}'. 失败: {e}")
        else:
            print(f"'{file_path}' 未知.")


# 新增：Base64解码并保存到指定目录
def decode_and_save_base64_image(b64_str, save_dir=IMAGE_SAVE_DIR, object_name=None, suffix=".jpg"):
    """
    将Base64编码的图片字符串解码并保存到指定目录
    :param b64_str: Base64字符串（支持带前缀如data:image/jpeg;base64,）
    :param save_dir: 保存目录
    :param object_name: 对象名称，用于生成文件名
    :param suffix: 文件后缀
    :return: 保存的文件路径
    """
    try:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 移除Base64前缀（如果有）
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        
        # 解码Base64
        img_data = base64.b64decode(b64_str)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        
        if object_name:
            # 清理对象名称中的特殊字符
            safe_object_name = "".join(c for c in object_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
            filename = f"{safe_object_name}_{timestamp}_{random_str}{suffix}"
        else:
            filename = f"image_{timestamp}_{random_str}{suffix}"
        
        # 保存文件
        file_path = os.path.join(save_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(img_data)
        
        return file_path
    except Exception as e:
        raise ValueError(f"Base64解码或保存失败: {str(e)}")
    
# 新增：Base64解码为临时文件的辅助函数
def decode_base64_to_temp_file(b64_str, suffix=".jpg"):
    """
    将Base64编码的图片字符串解码并保存为临时文件
    :param b64_str: Base64字符串（支持带前缀如data:image/jpeg;base64,）
    :param suffix: 临时文件后缀
    :return: 临时文件路径
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


# 接口1: 添加标注（Base64版）
@app.route('/api/add_annotation', methods=['POST'])
def api_add_annotation():
    """
    入参给出一个物体的Base64图片和对应物体的位置坐标，并给它命名，保存物体特征
    JSON入参:
    {
        "image_base64": "Base64编码的图片字符串",
        "object_name": "物体名称",
        "ref_box_xyxy": [x1, y1, x2, y2],
        "en_semantic_prompt": "英文增强提示词"
    }
    """
    try:
        # 获取JSON参数
        req_data = request.get_json()
        if not req_data:
            return jsonify({'error': '请求体必须为JSON格式'}), 400

        image_base64 = req_data.get('image_base64')
        object_name = req_data.get('object_name')
        en_semantic_prompt = req_data.get('en_semantic_prompt')
        ref_box_xyxy = req_data.get('ref_box_xyxy')

        # 校验参数
        if not all([image_base64, object_name, en_semantic_prompt, ref_box_xyxy]):
            return jsonify({'error': '缺少必要参数（image_base64/object_name/ref_box_xyxy/en_semantic_prompt）'}), 400
        if len(ref_box_xyxy) != 4:
            return jsonify({'error': '坐标格式错误，应为[x1,y1,x2,y2]'}), 400

        # Base64解码并保存到图片目录
        image_path = decode_and_save_base64_image(image_base64, object_name=object_name)

        # 获取图片尺寸
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # 调用API
        api = get_api_instance()
        result = api.add_annotation(
            ref_path=image_path,
            object_name=object_name,
            ref_box_xyxy=ref_box_xyxy,
            ref_width=img_width,
            ref_height=img_height,
            en_semantic_prompt=en_semantic_prompt
        )

        # 注意：这里不再删除图片文件，因为已经保存到永久目录
        # os.unlink(image_path)  # 注释掉这行

        if result:
            return jsonify({
                'success': True,
                'message': f'标注完成！{object_name} - 英文提示词：{en_semantic_prompt}',
                'image_path': image_path
            })
        else:
            return jsonify({'error': '标注失败'}), 500

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 接口2: 使用物体名称查找图片中的物体（Base64版）
@app.route('/api/find_object_by_name', methods=['POST'])
def api_find_object_by_name():
    """
    使用一个物体名称，查找Base64图片里的对应的物体
    JSON入参:
    {
        "image_base64": "Base64编码的图片字符串",
        "object_name": "物体名称"
    }
    """
    try:
        # 获取JSON参数
        req_data = request.get_json()
        if not req_data:
            return jsonify({'error': '请求体必须为JSON格式'}), 400

        image_base64 = req_data.get('image_base64')
        object_name = req_data.get('object_name')

        # 校验参数
        if not all([image_base64, object_name]):
            return jsonify({'error': '缺少必要参数（image_base64/object_name）'}), 400

        # Base64解码为临时文件（其他接口仍然使用临时文件）
        # 注意：这里仍然使用临时文件，因为只是临时推理使用
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        img_data = base64.b64decode(image_base64)
        temp_file.write(img_data)
        temp_file.close()
        image_path = temp_file.name

        # 调用API
        api = get_api_instance()
        
        # 融合两个函数：先提取特征，然后推理
        begin_time = time.time() * 1000
        inference_state, current_img_size = api.use_one_word_to_find_aim_picture_object_imgFeature(image_path)
        output = api.use_one_word_to_find_aim_picture_object(object_name, inference_state, current_img_size)
        output["name"] = [object_name] * len(output["scores"])
        end_time = time.time() * 1000

        # 清理临时文件
        os.unlink(image_path)

        # 转换输出格式为可JSON序列化
        result = {
            'success': True,
            'inference_time_ms': end_time - begin_time,
            'results': []
        }

        # 处理输出结果
        if 'masks' in output and len(output['masks']) > 0:
            for i in range(len(output['scores'])):
                result_item = {
                    'score': float(output['scores'][i]),
                    'name': output['name'][i] if i < len(output['name']) else object_name
                }
                
                # 处理边界框
                if i < len(output['boxes']):
                    box = output['boxes'][i]
                    if hasattr(box, 'cpu'):
                        box = box.cpu().numpy()
                    result_item['box'] = box.tolist() if hasattr(box, 'tolist') else list(box)
                
                # 处理掩码（简化处理，只返回基本信息）
                if i < len(output['masks']):
                    mask = output['masks'][i]
                    if hasattr(mask, 'cpu'):
                        mask = mask.cpu().numpy()
                    result_item['mask_shape'] = mask.shape if hasattr(mask, 'shape') else []
                
                result['results'].append(result_item)

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 接口3: 用一张图片中的物体查找另一张图片中的物体（Base64版）
@app.route('/api/find_object_by_feature', methods=['POST'])
def api_find_object_by_feature():
    """
    在一张Base64图片中框选一个物体，查找另一张Base64图片里的对应物体
    JSON入参:
    {
        "base_image_base64": "基础图片的Base64字符串",
        "target_image_base64": "目标图片的Base64字符串",
        "ref_box_xyxy": [x1, y1, x2, y2],
        "en_semantic_prompt": "英文增强提示词"
    }
    """
    try:
        # 获取JSON参数
        req_data = request.get_json()
        if not req_data:
            return jsonify({'error': '请求体必须为JSON格式'}), 400

        base_image_base64 = req_data.get('base_image_base64')
        target_image_base64 = req_data.get('target_image_base64')
        ref_box_xyxy = req_data.get('ref_box_xyxy')
        en_semantic_prompt = req_data.get('en_semantic_prompt')

        # 校验参数
        if not all([base_image_base64, target_image_base64, ref_box_xyxy, en_semantic_prompt]):
            return jsonify({'error': '缺少必要参数（base_image_base64/target_image_base64/ref_box_xyxy/en_semantic_prompt）'}), 400
        if len(ref_box_xyxy) != 4:
            return jsonify({'error': '坐标格式错误，应为[x1,y1,x2,y2]'}), 400

        # Base64解码为临时文件
        base_image_path = decode_base64_to_temp_file(base_image_base64)
        target_image_path = decode_base64_to_temp_file(target_image_base64)

        # 获取基础图片尺寸
        with Image.open(base_image_path) as img:
            img_width, img_height = img.size

        # 调用API
        api = get_api_instance()
        
        # 提取目标图片特征
        target_image, pro_set_img = api.use_one_feature_to_find_other_picture_imgFeature(target_image_path)
        
        # 执行查找
        begin_time = time.time() * 1000
        alloutput = api.use_one_feature_to_find_other_picture(
            base_image_path, target_image, pro_set_img, 
            ref_box_xyxy, img_width, img_height, en_semantic_prompt
        )
        end_time = time.time() * 1000

        # 清理临时文件
        os.unlink(base_image_path)
        os.unlink(target_image_path)

        # 处理输出结果
        result = {
            'success': True,
            'inference_time_ms': end_time - begin_time,
            'results': []
        }

        if isinstance(alloutput, list):
            for item in alloutput:
                if isinstance(item, dict):
                    result_item = {
                        'score': float(item.get('score', 0.0)),
                        'name': item.get('name', 'unknown')
                    }
                    
                    if 'box' in item:
                        box = item['box']
                        if hasattr(box, 'cpu'):
                            box = box.cpu().numpy()
                        result_item['box'] = box.tolist() if hasattr(box, 'tolist') else list(box)
                    
                    if 'mask' in item:
                        mask = item['mask']
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu().numpy()
                        result_item['mask_shape'] = mask.shape if hasattr(mask, 'shape') else []
                    
                    result['results'].append(result_item)

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 接口4: 批量查找图片中的物体（Base64版）
@app.route('/api/batch_find_by_feature', methods=['POST'])
def api_batch_find_by_feature():
    """
    使用一张Base64图片中的物体，批量查找其他Base64图片中的对应物体
    JSON入参:
    {
        "base_image_base64": "基础图片的Base64字符串",
        "target_images_base64": ["目标图片1的Base64", "目标图片2的Base64", ...],
        "ref_box_xyxy": [x1, y1, x2, y2],
        "en_semantic_prompt": "英文增强提示词"
    }
    """
    try:
        # 获取JSON参数
        req_data = request.get_json()
        if not req_data:
            return jsonify({'error': '请求体必须为JSON格式'}), 400

        base_image_base64 = req_data.get('base_image_base64')
        target_images_base64 = req_data.get('target_images_base64')
        ref_box_xyxy = req_data.get('ref_box_xyxy')
        en_semantic_prompt = req_data.get('en_semantic_prompt')

        # 校验参数
        if not all([base_image_base64, target_images_base64, ref_box_xyxy, en_semantic_prompt]):
            return jsonify({'error': '缺少必要参数（base_image_base64/target_images_base64/ref_box_xyxy/en_semantic_prompt）'}), 400
        if len(ref_box_xyxy) != 4:
            return jsonify({'error': '坐标格式错误，应为[x1,y1,x2,y2]'}), 400
        if not isinstance(target_images_base64, list) or len(target_images_base64) == 0:
            return jsonify({'error': 'target_images_base64必须为非空列表'}), 400

        # Base64解码为临时文件
        base_image_path = decode_base64_to_temp_file(base_image_base64)
        target_image_paths = []
        for b64_str in target_images_base64:
            path = decode_base64_to_temp_file(b64_str)
            target_image_paths.append(path)

        # 获取基础图片尺寸
        with Image.open(base_image_path) as img:
            img_width, img_height = img.size

        # 调用API
        api = get_api_instance()
        
        # 批量提取特征
        gen = api.use_one_feature_to_find_other_picture_batch_imgFeature(target_image_paths)
        
        # 执行批量查找
        begin_time = time.time() * 1000
        all_results = []
        
        for batch_states, batch_size in gen:
            batch_result = api.use_one_feature_to_find_other_picture_batch(
                base_image_path, batch_states, batch_size, 
                ref_box_xyxy, img_width, img_height, en_semantic_prompt
            )
            all_results.extend(batch_result)
        
        end_time = time.time() * 1000

        # 清理临时文件
        os.unlink(base_image_path)
        for path in target_image_paths:
            os.unlink(path)

        # 处理输出结果
        result = {
            'success': True,
            'inference_time_ms': end_time - begin_time,
            'total_results': len(all_results),
            'results': []
        }

        for i, item in enumerate(all_results):
            if isinstance(item, dict):
                result_item = {
                    'image_index': i,
                    'score': float(item.get('score', 0.0)),
                    'name': item.get('name', 'unknown')
                }
                
                if 'box' in item:
                    box = item['box']
                    if hasattr(box, 'cpu'):
                        box = box.cpu().numpy()
                    result_item['box'] = box.tolist() if hasattr(box, 'tolist') else list(box)
                
                result['results'].append(result_item)

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 接口5: 批量查找多个提示词对应的物体（Base64版）
@app.route('/api/batch_find_by_words', methods=['POST'])
def api_batch_find_by_words():
    """
    使用多个提示词，批量查找多张Base64图片中的对应物体
    JSON入参:
    {
        "target_images_base64": ["目标图片1的Base64", "目标图片2的Base64", ...],
        "object_names": ["物体名称1", "物体名称2", ...],
        "en_semantic_prompts": ["提示词1", "提示词2", ...]  // 可选，与物体名称一一对应
    }
    """
    try:
        # 获取JSON参数
        req_data = request.get_json()
        if not req_data:
            return jsonify({'error': '请求体必须为JSON格式'}), 400

        target_images_base64 = req_data.get('target_images_base64')
        object_names = req_data.get('object_names')
        en_semantic_prompts = req_data.get('en_semantic_prompts', [])

        # 校验参数
        if not all([target_images_base64, object_names]):
            return jsonify({'error': '缺少必要参数（target_images_base64/object_names）'}), 400
        if not isinstance(target_images_base64, list) or len(target_images_base64) == 0:
            return jsonify({'error': 'target_images_base64必须为非空列表'}), 400
        if not isinstance(object_names, list) or len(object_names) == 0:
            return jsonify({'error': 'object_names必须为非空列表'}), 400

        # 处理提示词（补齐长度）
        if len(en_semantic_prompts) < len(object_names):
            en_semantic_prompts.extend([''] * (len(object_names) - len(en_semantic_prompts)))

        # Base64解码为临时文件
        target_image_paths = []
        for b64_str in target_images_base64:
            path = decode_base64_to_temp_file(b64_str)
            target_image_paths.append(path)

        # 调用API
        api = get_api_instance()
        
        # 批量提取特征
        gen = api.use_words_list_to_find_other_picture_imgFeature(target_image_paths)
        
        # 执行批量查找
        begin_time = time.time() * 1000
        all_results = []
        
        for batch_image_paths, batch_inference_states, final_output in gen:
            batch_result = api.use_words_list_to_find_other_picture(
                batch_image_paths, object_names, en_semantic_prompts,
                batch_inference_states, final_output
            )
            all_results.extend(batch_result)
        
        end_time = time.time() * 1000

        # 清理临时文件
        for path in target_image_paths:
            os.unlink(path)

        # 处理输出结果
        result = {
            'success': True,
            'inference_time_ms': end_time - begin_time,
            'total_images': len(all_results),
            'results': []
        }

        for i, img_result in enumerate(all_results):
            if isinstance(img_result, dict):
                img_data = {
                    'image_index': i,
                    'detections': []
                }
                
                # 检查是否有检测结果
                if 'scores' in img_result and len(img_result['scores']) > 0:
                    for j in range(len(img_result['scores'])):
                        detection = {
                            'score': float(img_result['scores'][j]),
                            'name': img_result['name'][j] if j < len(img_result['name']) else 'unknown'
                        }
                        
                        if j < len(img_result['boxes']):
                            box = img_result['boxes'][j]
                            if hasattr(box, 'cpu'):
                                box = box.cpu().numpy()
                            detection['box'] = box.tolist() if hasattr(box, 'tolist') else list(box)
                        
                        img_data['detections'].append(detection)
                
                result['results'].append(img_data)

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 健康检查接口
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'device': str(device)})


# 清理资源的接口
@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """清理临时文件和释放资源"""
    try:
        api = get_api_instance()
        api.remove_tmp_file()
        
        # 释放GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({'success': True, 'message': '资源清理完成'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 启动时初始化API
    get_api_instance()
    
    # 启动Flask服务
    print("Flask API服务启动中...（支持Base64+JSON请求）")
    app.run(host='0.0.0.0', port=5000, debug=True)