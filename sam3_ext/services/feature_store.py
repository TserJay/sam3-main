"""
特征存储管理服务
"""
import json
import os
from typing import Dict, Any, Optional


class FeatureStore:
    """特征存储管理类"""
    
    def __init__(self, feature_file_path: str = "configs/features.json", tmp_file_path: str = "configs/tmp_features.json"):
        self.feature_file_path = feature_file_path
        self.tmp_file_path = tmp_file_path
        self._feature_map: Dict[str, Any] = {}
        self._tmp_feature_map: Dict[str, Any] = {}
    
    def load(self, use_tmp: bool = False) -> Dict[str, Any]:
        """
        加载特征库
        
        Args:
            use_tmp: 是否加载临时特征文件
        
        Returns:
            特征映射字典
        """
        file_path = self.tmp_file_path if use_tmp else self.feature_file_path
        feature_map = {}
        
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    loaded_map = json.loads(content) if content else {}
                feature_map = loaded_map
            except Exception as e:
                print(f"加载特征文件失败（文件损坏），创建新文件：{e}")
                self._create_empty_file(file_path)
        else:
            print(f"创建特征文件：{file_path}")
            self._create_empty_file(file_path)
        
        if use_tmp:
            self._tmp_feature_map = feature_map
        else:
            self._feature_map = feature_map
        
        return feature_map
    
    def save(self, feature_map: Dict[str, Any], use_tmp: bool = False) -> bool:
        """
        保存特征库
        
        Args:
            feature_map: 特征映射字典
            use_tmp: 是否保存到临时文件
        
        Returns:
            是否保存成功
        """
        file_path = self.tmp_file_path if use_tmp else self.feature_file_path
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(feature_map, f, ensure_ascii=False, indent=4)
            print(f"✅ 特征已保存到：{file_path}")
            return True
        except Exception as e:
            print(f"❌ 保存失败：{e}")
            return False
    
    def add_annotation(
        self,
        object_name: str,
        ref_image_path: str,
        ref_box_xyxy: list,
        ref_width: int,
        ref_height: int,
        en_semantic_prompt: str,
        use_tmp: bool = False
    ) -> bool:
        """
        添加/更新物体标注
        
        Args:
            object_name: 物体名称
            ref_image_path: 参考图路径
            ref_box_xyxy: 框选的像素框
            ref_width: 参考图宽度
            ref_height: 参考图高度
            en_semantic_prompt: 英文增强提示词
            use_tmp: 是否使用临时文件
        
        Returns:
            是否添加成功
        """
        feature_map = self.load(use_tmp=use_tmp)
        
        if object_name in feature_map:
            obj_info = feature_map[object_name]
            if ref_box_xyxy not in obj_info.get("representative_boxes", []):
                if "representative_boxes" not in obj_info:
                    obj_info["representative_boxes"] = []
                obj_info["representative_boxes"].append(ref_box_xyxy)
            if en_semantic_prompt and en_semantic_prompt != obj_info.get("en_semantic_prompt"):
                obj_info["en_semantic_prompt"] = en_semantic_prompt
            obj_info["ref_image_path"] = ref_image_path
            obj_info["ref_image_size"] = [ref_width, ref_height]
        else:
            obj_info = {
                "ref_image_path": ref_image_path,
                "ref_image_size": [ref_width, ref_height],
                "representative_boxes": [ref_box_xyxy],
                "en_semantic_prompt": en_semantic_prompt
            }
        
        feature_map[object_name] = obj_info
        return self.save(feature_map, use_tmp=use_tmp)
    
    def get_object_feature(self, object_name: str, use_tmp: bool = False) -> Optional[Dict[str, Any]]:
        """
        获取物体特征信息
        
        Args:
            object_name: 物体名称
            use_tmp: 是否从临时文件获取
        
        Returns:
            物体特征信息，不存在返回None
        """
        feature_map = self.load(use_tmp=use_tmp)
        return feature_map.get(object_name)
    
    def remove_tmp_file(self) -> bool:
        """删除临时特征文件"""
        if os.path.exists(self.tmp_file_path):
            try:
                os.remove(self.tmp_file_path)
                print(f"'{self.tmp_file_path}' 删除成功.")
                return True
            except Exception as e:
                print(f"删除 '{self.tmp_file_path}' 失败: {e}")
                return False
        return True
    
    def _create_empty_file(self, file_path: str):
        """创建空的特征文件"""
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
    
    @staticmethod
    def convert_old_feature_map(old_map: Dict[str, Any]) -> Dict[str, Any]:
        """转换旧版特征格式"""
        new_map = {}
        for name, feature in old_map.items():
            new_map[name] = [feature] if isinstance(feature, dict) else feature
        return new_map
