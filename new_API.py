import gc
from datetime import datetime
import json
import os
import time


from test_picture_to_picture_optimizer_batch_inference import this_main as a_main, \
    xyxy2cxcywh_norm  # 用一张图上的特征，去查找其它图特征,批量推理模式
from test_picture_to_picture_optimizer import this_main as b_main  #用一张图上的特征，去查找其它图特征,串行推理模式
from word_picture_batch_inference import this_main as c_main #用一个词列表，去查找目标图片文件夹里的所有词列表中的物体
from test_words_picture import words_to_picture  #用一个词，去查找目标图片文件里的物体

import sam3
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import platform


sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
feature_file_path = "sam3_custom_objects.json" #特征保存文件
tmp_file_path="tmp_load.json"
object_feature_map = {}
tmp_feature_map = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=16

class the_API:
    def __init__(self):
        global object_feature_map,tmp_feature_map
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
        tmp_feature_map=self.load_feature_map(file_path=tmp_file_path)
        object_feature_map=self.load_feature_map(file_path=feature_file_path)

    def use_one_feature_to_find_other_picture_batch(self,base_image_path,batch_states,this_batch_size,location,img_width,img_height,
                                                    zengqiang_prompt):
        # batch_states,this_batch_size=self.use_one_feature_to_find_other_picture_batch_imgFeature(target_image_paths)
        time_now=str(datetime.now())
        obj_name="tmp"+time_now
        self.add_annotation(base_image_path,obj_name,location,img_width,img_height,zengqiang_prompt,save_path=tmp_file_path)
        all_result=a_main(target_name=obj_name,zengqiang_prompt=zengqiang_prompt,processor=self.processor,
                      batch_states=batch_states,this_batch_size=this_batch_size,device=device) #result是个迭代器
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
            yield [], 0  # 修复：返回空值而非None，避免解包错误
            return
        # 修复：变量名改为current_batch_size，避免覆盖全局batch_size
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



    def use_one_feature_to_find_other_picture(self,base_image_path,target_image,pro_set_img,location,img_width,img_height,zengqiang_prompt):
        #base_image_path是特征提取文件,
        # target_image,pro_set_img = self.use_one_feature_to_find_other_picture_imgFeature(target_image_path)
        time_now = str(datetime.now())
        obj_name = "tmp" + time_now
        self.add_annotation(base_image_path, obj_name,location,img_width,img_height,zengqiang_prompt,save_path=tmp_file_path)
        all_result=b_main(obj_name,self.processor,target_image,pro_set_img,zengqiang_prompt,device=device) #obj_name是特征图片的文件路径
        return all_result #{"mask"\"box"\"score"\"feature_idx"\"result_idx"\"name"}
    def use_one_feature_to_find_other_picture_imgFeature(self,target_image_path): #先进行特征提取
        target_image = Image.open(target_image_path).convert("RGB")
        pro_set_img=self.processor.set_image(target_image)
        return target_image,pro_set_img

    def use_words_list_to_find_other_picture(self, batch_image_paths, batch_prompts, batch_zengqiang_prompt,
                                             batch_inference_states, final_output):
        """# batch_image_paths, batch_inference_states, final_output=self.use_words_list_to_find_other_picture_imgFeature(image_paths)"""
        result = c_main(
            batch_image_paths,
            batch_prompts,
            batch_zengqiang_prompt,
            self.processor,
            final_output,
            batch_inference_states,
            device=device
        )
        return result #{"mask": [], "boxes": [], "scores": [],"name": []}

    def use_words_list_to_find_other_picture_imgFeature(self, image_paths, batch_size=batch_size):
        """
        生成器：按批次处理图片，返回每个批次的路径/state/占位列表
        核心：final_output保证批次内图片索引一一对应，后续由this_main填充有效结果
        """
        for batch_start in range(0, len(image_paths), batch_size):
            # 1. 截取当前批次的图片路径（固定逻辑）
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_image_paths = image_paths[batch_start:batch_end]
            batch_idx = batch_start // batch_size + 1
            # 2. 初始化批次内的final_output（核心：长度=批次图片数，每个位置先占位）
            final_output = []
            batch_images = []
            valid_image_indices = []  # 记录有效图片在批次内的索引（比如[0,2]表示批次第0、2张有效）

            # 3. 加载图片 + 初始化final_output（保证索引一一对应）
            for idx, path in enumerate(batch_image_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_image_indices.append(idx)  # 记录有效图片的批次内索引
                    # 有效图片：先初始化空字典（后续this_main填充结果）
                    final_output.append({"mask": [], "boxes": [], "scores": [], "name": []})
                except Exception as e:
                    print(f"   ❌ 加载图片失败 {path}: {e} | 批次内索引：{idx} | 占位空字典")
                    # 无效图片：直接占位空字典（后续不修改）
                    final_output.append({"mask": [], "boxes": [], "scores": [], "name": []})

            # 4. 生成有效图片的inference_state（仅处理加载成功的图片）
            batch_inference_states = []
            for img in batch_images:
                state = self.processor.set_image(img)
                batch_inference_states.append(state)
            yield batch_image_paths, batch_inference_states, final_output

    def use_one_word_to_find_aim_picture_object(self,your_prompt,inference_state,current_img_size):
        # inference_state,current_img_size=self.use_one_word_to_find_aim_picture_object_imgFeature(img_path)
        zengqiang_prompt = your_prompt if your_prompt not in object_feature_map else object_feature_map[your_prompt][
            "en_semantic_prompt"]
        output=words_to_picture(self.processor,inference_state,your_prompt,zengqiang_prompt,current_img_size)
        return output #output["masks"], output["boxes"], output["scores"]
    def use_one_word_to_find_aim_picture_object_imgFeature(self,img_path): #先进行特征提取
        image = Image.open(img_path)
        current_img_size = [image.width, image.height]
        inference_state = self.processor.set_image(image)
        return inference_state,current_img_size


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
        return ref_box_xyxy, ref_width, ref_height  # ref_box_xyxy=[x1,y1,x2,y2]

    def add_annotation(self, ref_path, object_name, ref_box_xyxy, ref_width, ref_height,
                       en_semantic_prompt, save_path=feature_file_path):
        """
        新增/更新物体标注（适配新结构：参考信息+代表性框+英文提示词）
        :param ref_path: 参考图路
        :param object_name: 中文物体名（如“定制红色金属零件”）
        :param ref_box_xyxy: 框选的像素框（从plot_pic_get_location获取）
        :param ref_width/ref_height: 参考图尺寸（从plot_pic_get_location获取）
        :param en_semantic_prompt: 手动传入的英文视觉特征提示词（核心）
        :param save_path: 特征文件保存路径
        :return: 标注是否成功
        """
        ref_image_path = ref_path

        # 1. 加载已有特征库
        existing_map = self.load_feature_map(save_path)

        # 3. 适配新JSON结构：更新/新增物体信息
        if object_name in existing_map:
            # 已有该物体：仅添加新的代表性框（不重复存储参考图信息）
            obj_info = existing_map[object_name]
            # 避免重复添加相同框
            if ref_box_xyxy not in obj_info["representative_boxes"]:
                obj_info["representative_boxes"].append(ref_box_xyxy)
            # 若手动传入了新的英文提示词，更新
            if en_semantic_prompt != obj_info["en_semantic_prompt"] and en_semantic_prompt:
                obj_info["en_semantic_prompt"] = en_semantic_prompt
        else:
            # 新增物体：初始化完整结构（核心修改点）
            obj_info = {
                "ref_image_path": ref_image_path,  # 参考图路径
                "ref_image_size": [ref_width, ref_height],  # 参考图尺寸
                "representative_boxes": [ref_box_xyxy],  # 代表性框（视觉模板）
                "en_semantic_prompt": en_semantic_prompt  # 英文视觉特征提示词
            }
        existing_map[object_name] = obj_info

        # 4. 保存更新后的特征库
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
    def convert_old_feature_map(self,old_map):
        new_map = {}
        for name, feature in old_map.items():
            new_map[name] = [feature]
        return new_map
    def remove_tmp_file(self,file_path=tmp_file_path):
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
    :param output: 推理输出结果（支持多格式：字典/列表/嵌套列表）
    :param picture_path: 图片文件路径
    :param save_path: 可选，保存标注后图片的路径（如"result.jpg"）
    """
    # -------------------------- 1. 读取并预处理图片 --------------------------
    # 读取图片（处理中文路径）
    img = cv2.imdecode(np.fromfile(picture_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片：{picture_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转RGB用于matplotlib显示
    img_copy = img_rgb.copy()  # 用于绘制mask和框的副本

    # -------------------------- 2. 定义颜色配置 --------------------------
    # 不同检测结果的颜色（循环使用）
    colors = [
        (255, 0, 0),  # 红
        (0, 255, 0),  # 绿
        (0, 0, 255),  # 蓝
        (255, 255, 0),  # 黄
        (255, 0, 255),  # 紫
        (0, 255, 255)  # 青
    ]
    mask_alpha = 0.3  # mask透明度

    # -------------------------- 3. 解析输出结果 --------------------------
    # 统一格式：将各种输出转为 [{'mask': ..., 'box': ..., 'score': ..., 'name': ...}, ...]
    detections = []

    # 情况1：output是顶层结果字典（如第一个推理输出）
    if isinstance(output, dict) and 'scores' in output and 'boxes' in output:
        masks = ''
        if 'masks' in output:
            masks = output['masks'].cpu().numpy() if hasattr(output['masks'], 'cpu') else output['masks']
        elif 'mask' in output:
            masks = output['mask'].cpu().numpy() if hasattr(output['mask'], 'cpu') else output['mask']
        boxes = output['boxes'].cpu().numpy() if hasattr(output['boxes'], 'cpu') else output['boxes']
        scores = output['scores'].cpu().numpy() if hasattr(output['scores'], 'cpu') else output['scores']

        for i in range(len(boxes)):
            # 处理name为数组/列表的情况，取对应索引的名称
            names = output.get('name', [])
            if isinstance(names, (list, np.ndarray)) and len(names) > i:
                name = names[i]  # 取数组中对应索引的单个名称
            else:
                name = f'object_{i}'  # 兜底默认名称

            det = {
                'mask': masks[i].squeeze() if len(masks) > i else None,
                'box': boxes[i] if len(boxes) > i else None,
                'score': scores[i] if len(scores) > i else 0.0,
                'name': name
            }
            detections.append(det)

    # 情况2：output是列表（如alloutput/bestoutput）
    elif isinstance(output, list):
        # 递归解析嵌套列表
        def flatten_list(lst):
            for item in lst:
                if isinstance(item, list):
                    yield from flatten_list(item)
                elif isinstance(item, dict):
                    yield item

        for det in flatten_list(output):
            # 补全缺失字段
            detections.append({
                'mask': det.get('mask', None),
                'box': det.get('box', None),
                'score': det.get('score', 0.0),
                'name': det.get('name', 'unknown')
            })

    # 情况3：output是单个字典（如bestoutput单结果）
    elif isinstance(output, dict):
        detections.append({
            'mask': output.get('mask', None),
            'box': output.get('box', None),
            'score': output.get('score', 0.0),
            'name': output.get('name', 'unknown')
        })

    # -------------------------- 新增：创建名称-颜色映射字典 --------------------------
    # 提取所有唯一的物体名称，为每个名称分配固定颜色
    unique_names = list(set([det['name'] for det in detections if det['name']]))
    name_to_color = {}
    for idx, name in enumerate(unique_names):
        name_to_color[name] = colors[idx % len(colors)]  # 循环分配颜色

    # -------------------------- 4. 绘制标注 --------------------------
    for idx, det in enumerate(detections):
        # 跳过空结果
        if det['box'] is None and det['mask'] is None:
            continue

        # 修复问题1：相同名称使用相同颜色
        color = name_to_color[det['name']]  # 根据名称取固定颜色
        score = det['score']
        name = det['name']

        # #4.1 绘制mask（如果有）
        # print(det)
        # if det['mask'] is not None and det['mask'].size > 0:
        #     mask = det['mask']
        #     # 调整mask尺寸匹配图片
        #     if mask.shape[:2] != img_copy.shape[:2]:
        #         mask = cv2.resize(mask.astype(np.uint8), (img_copy.shape[1], img_copy.shape[0])).astype(bool)
        #
        #     # 叠加mask到图片
        #     mask_rgb = np.zeros_like(img_copy)
        #     mask_rgb[mask] = color
        #     img_copy = cv2.addWeighted(img_copy, 1, mask_rgb, mask_alpha, 0)

        # 4.2 绘制检测框和文本（如果有box）
        if det['box'] is not None and len(det['box']) == 4:
            x1, y1, x2, y2 = map(int, det['box'])
            # 绘制矩形框（保留原OpenCV绘制，不影响）
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # 使用PIL绘制中文文本（支持中文）
            # 1. 转换OpenCV图像为PIL Image
            pil_img = Image.fromarray(img_copy)
            draw = ImageDraw.Draw(pil_img)

            # 2. 加载中文字体（兼容Windows/Linux/macOS）
            font_path = ""
            if platform.system() == "Windows":
                font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
            elif platform.system() == "Linux":
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 通用字体
            elif platform.system() == "Darwin":  # macOS
                font_path = "/System/Library/Fonts/PingFang.ttc"  # 苹方

            # 兜底：如果指定字体不存在，使用默认字体
            try:
                # 调大字体大小（从0-20，可根据需求再调整）
                font = ImageFont.truetype(font_path, 20)  # 字体大小从14→20
            except:
                font = ImageFont.load_default()
                print(f"警告：无法加载中文字体，路径{font_path}不存在，使用默认字体（可能仍无法显示中文）")

            # 3. 准备文本内容
            text = f"{name} | {score:.2f}"

            # 4. 计算文本尺寸（替代cv2.getTextSize）
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # 5. 绘制文本背景（矩形）- 适配更大的字体
            bg_x1, bg_y1 = x1, y1 - text_h - 12  # 微调背景框位置，适配大字体
            bg_x2, bg_y2 = x1 + text_w, y1
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)

            # 6. 绘制文本（白色字体）
            draw.text((x1, bg_y1), text, font=font, fill=(255, 255, 255))

            # 7. 转换回OpenCV格式
            img_copy = np.array(pil_img)

    # -------------------------- 5. 显示/保存图片 --------------------------
    plt.figure(figsize=(12, 8))
    plt.imshow(img_copy)
    plt.axis('off')  # 关闭坐标轴
    plt.title(f"Detection Results - {picture_path}", fontsize=12)
    plt.tight_layout()

    # 保存图片（支持中文路径）
    if save_path:
        img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        cv2.imencode('.jpg', img_bgr)[1].tofile(save_path)
        print(f"标注图片已保存至：{save_path}")

    # 显示图片
    plt.show()

if __name__ == "__main__": #接口测试
    print("开始创建模型...")
    a=the_API()
    print("初始化模型成功")
    print("=======================================================")

# ==================================从图片中框选一个物体，并给它命名，需要添加该物体的特征，作为增强提示词保存==================================
    pic_path =r"assets/images/truck.jpg" #图片位置
    object_name="汽车轮子" #需要添加的物体名称
    en_prompt="car wheel, truck tire, black rubber tire, large heavy-duty tire"  #需要添加的增强提示词,必须填写
    # ref_box_xyxy, ref_width, ref_height=a.plot_pic_get_location(pic_path)  #画图获取目标物体所在图片的位置(x1,y1,x2,y2)、宽、高
    ref_box_xyxy, ref_width, ref_height =[np.float64(433.8225806451614), np.float64(643.6290322580645), np.float64(663.758064516129), np.float64(845.6935483870968)],1800,1200
    a.add_annotation(pic_path,object_name,ref_box_xyxy, ref_width, ref_height,
                     en_prompt)
    print("新物体添加成功======================================================")

    #
    # print("添加法兰盘测试↓================================")
    # pic_path = r"assets/images/法兰盘.png"  # 图片位置
    # object_name = "flange"  # 需要添加的物体名称，此处为法兰盘
    # en_prompt = ("large industrial black circular flange with X-shaped reinforcement structure,"
    #              " located at the end of a long composite material component (wind turbine blade) on a factory production line, "
    #              "heavy-duty metal flange, factory workshop setting, surrounded by yellow safety stairs and workers")  # 需要添加的增强提示词,必须填写
    # ref_box_xyxy, ref_width, ref_height=a.plot_pic_get_location(pic_path)  #画图获取目标物体所在图片的位置(x1,y1,x2,y2)、宽、高
    # a.add_annotation(pic_path, object_name, ref_box_xyxy, ref_width, ref_height,
    #                  en_prompt)
    # print("新物体添加成功======================================================")

#
#==========================================使用一个物体名称，去查找图片里的对应的物体======================================================
    aim_picture=r"assets/images/truck.jpg" #目标图片
    your_prompt = "汽车轮子" #需要查找的物体
    inference_state,current_img_size=a.use_one_word_to_find_aim_picture_object_imgFeature(aim_picture) #先提取特征，特征提取与推理相分离
    print(f"图片特征提取成功,pic_feature.size={current_img_size}")
    begin_time=time.time()*1000
    output=a.use_one_word_to_find_aim_picture_object(your_prompt,inference_state,current_img_size);output["name"]=[your_prompt]*len(output["scores"]) #推理得出结果
    end_time=time.time()*1000
    print(f"推理时间：{end_time-begin_time} ms")
    # plot_figure(output, aim_picture)
    # print(output)
    # print(output["masks"])
    # print(output["boxes"])
    # print(output["scores"])
    # print(output["name"])
    # print(f"{aim_picture} 的 {your_prompt} 物体查找完成==========================================")



# ==========================================在一张图片中框选一个物体，并输入该物体的特征，去查找另一张图片里的对应的物体======================================================
    base_picture = r'assets/images/test_image.jpg' #需要添加查找的物体所在的图片
    aim_picture=r'assets/images/a_people.png'#目标查找图片
    en_promt="person,people" #增强提示词描述，必填
    target_image,pro_set_img = a.use_one_feature_to_find_other_picture_imgFeature(aim_picture) #另一张图片特征提取
    print("图片特征提取成功")
    # postion, width, height=a.plot_pic_get_location(base_picture) #使用画图方式获取位置、宽、高
    postion, width, height =[np.float64(496.6354838709678), np.float64(308.29999999999995), np.float64(587.4741935483871), np.float64(645.2290322580645)],1280,720
    begin_time = time.time() * 1000
    alloutput=a.use_one_feature_to_find_other_picture(base_picture,target_image,pro_set_img,postion, width, height,en_promt)
    end_time = time.time() * 1000
    print(f"推理时间：{end_time - begin_time}")
    # plot_figure(alloutput,aim_picture) #画图显示
    print(alloutput) #数组字典，每个数组包括alloutput[mask]、alloutput[box]、alloutput[score]、alloutput[name]
    print("============================================")





#===============================================使用一张图片中的某一个物体，去查找其它图片中相对应的物体（批量查找），该物体无需命名===================================
    other_picture_batch = [r"assets/images/truck.jpg",
                           r"assets/images/car3.png",
                           r"assets/images/car2.jpg"]
    zengqiang_prompt="car wheel, truck tire, black rubber tire, large heavy-duty tire" #增强提示词，必须要有
    base_picture=r"assets/images/car1.jpg"
    gen = a.use_one_feature_to_find_other_picture_batch_imgFeature(other_picture_batch) #gen是一个迭代器
    print("图片特征提取成功")
    # postion, width, height= a.plot_pic_get_location(base_picture) #使用画图方式获取位置、宽、高
    postion, width, height= [np.float64(317.22321428571433), np.float64(543.392857142857), np.float64(358.794642857143), np.float64(588.4285714285713)],800,1067
    begin_time = time.time() * 1000
    for ba,this in gen:
        all_out=a.use_one_feature_to_find_other_picture_batch(base_picture,ba,this, postion, width, height,zengqiang_prompt)
        # print(all_out)


        # for idx in range(len(other_picture_batch)): 画图显示
        #     plot_figure(all_out[idx],other_picture_batch[idx])
    end_time = time.time() * 1000
    print(f"推理时间：{end_time - begin_time}")
    print("===========================================================================================================")




#=======================================使用多个提示词，在多张图片里查找（批量查找）所有提示词中的物体========================================
    other_picture_batch = [r"assets/images/truck.jpg",
                           r"assets/images/car3.png",
                           r"assets/images/car2.jpg",
                           r"assets/images/法兰盘.png"]
    batch_prompt=['truck','汽车轮子',"flange"]  #一次性查多个物体
    batch_zengqiang_prompt=['',"car wheel, truck tire, black rubber tire, large heavy-duty tire",'']  #批量增强提示词，需要和物体名称一一对应，
                                                                                                    # 如果用户没有输入物体对应的增强提示词，则为空
    gen=a.use_words_list_to_find_other_picture_imgFeature(other_picture_batch)
    print("特征提取成功")
    begin_time = time.time() * 1000
    for batch_image_paths, pic_feature, final_ in gen:
        batch_output = a.use_words_list_to_find_other_picture(batch_image_paths,batch_prompt,batch_zengqiang_prompt,pic_feature,final_)
        # print(batch_output)
        # for out in batch_output:
        #     print(out['mask'])
        #     print(out['boxes'])
        #     print(out['scores'])


        # for idx in range(len(batch_image_paths)):   #画图显示
        #     plot_figure(batch_output[idx],batch_image_paths[idx])
    end_time = time.time() * 1000
    print(f"推理时间：{end_time - begin_time}")
    print("==================================================================================")

#==============================每次退出程序时删除临时文件和关闭模型算法释放资源===============================================
    a.remove_tmp_file() #删除临时文件
    del a;gc.collect() #关闭算法模型
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
