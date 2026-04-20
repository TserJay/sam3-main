import json
import os
import gc
import time
import uuid
import threading
import numpy as np
from flask import Flask, request, jsonify
from new_API import the_API
import torch

# ===================== 1. 自定义JSON编码器（增强版） =====================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # 处理所有numpy类型
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.uint8):
            return int(obj)
        # 处理torch张量（若有）
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        # 处理嵌套列表/字典
        elif isinstance(obj, list):
            return [self.default(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self.default(v) for k, v in obj.items()}
        # 其他类型默认处理
        else:
            return super(NumpyEncoder, self).default(obj)

# ===================== 2. Flask初始化（强制绑定编码器） =====================
app = Flask(__name__)
# 强制设置编码器（优先级最高）
app.json_encoder = NumpyEncoder
# 全局禁用Flask的JSON_AS_ASCII（避免中文乱码，顺带确保编码器生效）
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 允许100MB请求体（图片特征可能大）
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # 关闭美化，提升响应速度

# ===================== 3. 全局变量与缓存 =====================
sam3_api = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_cache = {}
cache_lock = threading.Lock()
CACHE_EXPIRE_SECONDS = 30

# ===================== 4. 模型初始化 =====================
model_initialized = False
model_init_lock = threading.Lock()



@app.after_request
def add_long_connection_headers(response):
    response.headers['Connection'] = 'keep-alive'
    response.headers['Keep-Alive'] = 'timeout=1800, max=1000'
    response.headers['X-Timeout'] = '1800'  # 告诉客户端超时阈值
    return response


@app.before_request
def init_sam3():
    global sam3_api,model_initialized
    with model_init_lock:
        if not model_initialized:
            print("开始初始化SAM3模型...")
            sam3_api = the_API()
            model_initialized = True
            print("SAM3模型初始化完成！")
# ===================== 5. 缓存清理 =====================
def clean_expired_cache():
    with cache_lock:
        current_time = time.time()
        expired_keys = [k for k, (_, et) in feature_cache.items() if current_time > et]
        for k in expired_keys:
            del feature_cache[k]
    threading.Timer(60, clean_expired_cache).start()
clean_expired_cache()

# ===================== 6. 辅助函数：手动转换ndarray为列表（兜底） =====================
def convert_ndarray_to_list(data):
    """递归转换所有ndarray为列表"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist()
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    elif isinstance(data, dict):
        return {k: convert_ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, (np.integer, np.floating, np.bool_)):
        return data.item()
    else:
        return data

# ===================== 7. 特征提取接口 =====================
@app.route('/api/extract_image_features', methods=['POST'])
def extract_image_features():
    try:
        data = request.json
        other_picture_batch = data.get('other_picture_batch')
        if not other_picture_batch:
            return jsonify({"code": 400, "msg": "图片路径列表不能为空"}), 400

        gen = sam3_api.use_words_list_to_find_other_picture_imgFeature(other_picture_batch)
        batch_features = []
        for batch_image_paths, batch_inference_states, final_output in gen:
            batch_features.append((batch_image_paths, batch_inference_states, final_output))
        
        feature_id = str(uuid.uuid4())
        expire_time = time.time() + CACHE_EXPIRE_SECONDS
        with cache_lock:
            feature_cache[feature_id] = (batch_features, expire_time)
        return jsonify({
            "code": 200,
            "msg": "特征提取成功",
            "feature_id": feature_id,
            "batch_count": len(batch_features),
            "expire_seconds": CACHE_EXPIRE_SECONDS
        })
    except Exception as e:
        return jsonify({"code": 500, "msg": f"特征提取失败：{str(e)}"}), 500

# ===================== 8. 推理接口（核心：手动兜底转换） =====================
@app.route('/api/infer_by_batch_words', methods=['POST'])
def infer_by_batch_words():
    try:
        data = request.json
        feature_id = data.get('feature_id')
        batch_prompt = data.get('batch_prompt')
        batch_zengqiang_prompt = data.get('batch_zengqiang_prompt', ['']*len(batch_prompt))
        if not feature_id:
            return jsonify({"code": 400, "msg": "特征ID不能为空"}), 400
        if not batch_prompt:
            return jsonify({"code": 400, "msg": "词列表不能为空"}), 400
        
        with cache_lock:
            if feature_id not in feature_cache:
                return jsonify({"code": 404, "msg": "特征ID不存在或已过期"}), 404
            batch_features, _ = feature_cache[feature_id]
        # 批量推理
        begin_time = time.time() * 1000
        all_results = []
        for batch_image_paths, pic_feature, final_ in batch_features:
            batch_output = sam3_api.use_words_list_to_find_other_picture(
                batch_image_paths, batch_prompt, batch_zengqiang_prompt, pic_feature, final_
            )
            # ========== 关键：手动转换所有ndarray为列表（兜底） ==========
            batch_output = convert_ndarray_to_list(batch_output)
            all_results.append(batch_output)
            print(all_results)
            print("+"*30)
        end_time = time.time() * 1000

        # ========== 最终返回：用自定义编码器序列化 ==========
        return jsonify({
            "code": 200,
            "msg": "推理成功",
            "inference_time_ms": end_time - begin_time,
            "batch_count": len(all_results),
            "result": all_results
        })
    except Exception as e:
        # 打印详细错误（便于定位）
        import traceback
        print(f"推理错误详情：{traceback.format_exc()}")
        return jsonify({"code": 500, "msg": f"推理失败：{str(e)}"}), 500

# ===================== 9. 健康检查 =====================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"code": 200, "msg": "SAM3 service is running"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
