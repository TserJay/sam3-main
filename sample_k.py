"""
SAM3 推理示例脚本
实现两种推理模式：
1. 多张推理/batch inference --- 不同frame帧关注同样的类别
2. 单张多类别推理 --- 同一frame帧关注不同的类别

包含推理速度和显存占用测量功能
"""

import os
import time
import copy
import gc
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

# ======================== 配置 ========================
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 全局模型实例（延迟加载）
_model = None
_processor = None


def get_model_and_processor():
    """获取模型和处理器单例"""
    global _model, _processor
    if _model is None:
        print("正在加载模型...")
        bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        _model = build_sam3_image_model(bpe_path=bpe_path)
        _model.to(device)
        _model.eval()
        _processor = Sam3Processor(
            model=_model,
            resolution=1008,
            device=device,
            confidence_threshold=0.5
        )
        print("模型加载完成")
    return _model, _processor


def get_gpu_memory_info() -> Dict[str, float]:
    """获取GPU显存使用情况（MB）"""
    if not torch.cuda.is_available():
        return {"allocated": 0, "cached": 0, "total": 0}
    
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    cached = torch.cuda.memory_reserved() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    
    return {
        "allocated": round(allocated, 2),
        "cached": round(cached, 2),
        "total": round(total, 2)
    }


def clear_gpu_cache():
    """清理GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class InferenceBenchmark:
    """推理性能基准测试工具"""
    
    def __init__(self):
        self.results = []
    
    def measure(self, name: str, func, *args, **kwargs):
        """测量函数执行时间和显存"""
        clear_gpu_cache()
        mem_before = get_gpu_memory_info()
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        mem_after = get_gpu_memory_info()
        
        benchmark_result = {
            "name": name,
            "time_ms": round((end_time - start_time) * 1000, 2),
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
            "memory_delta_mb": round(mem_after["allocated"] - mem_before["allocated"], 2)
        }
        self.results.append(benchmark_result)
        
        print(f"\n[{name}]")
        print(f"  推理时间: {benchmark_result['time_ms']:.2f} ms")
        print(f"  显存增量: {benchmark_result['memory_delta_mb']:.2f} MB")
        print(f"  显存占用: {mem_after['allocated']:.2f} / {mem_after['total']:.2f} MB")
        
        return result, benchmark_result
    
    def summary(self):
        """输出汇总报告"""
        print("\n" + "=" * 60)
        print("推理性能汇总报告")
        print("=" * 60)
        
        total_time = 0
        for r in self.results:
            print(f"\n{r['name']}:")
            print(f"  时间: {r['time_ms']:.2f} ms")
            print(f"  显存增量: {r['memory_delta_mb']:.2f} MB")
            total_time += r['time_ms']
        
        print(f"\n总推理时间: {total_time:.2f} ms")
        print("=" * 60)


# ======================== 模式1: 多张推理/Batch Inference ========================
# 不同frame帧关注同样的类别

def batch_inference_same_category(
    image_paths: List[str],
    prompt: str,
    batch_size: int = 4,
    benchmark: Optional[InferenceBenchmark] = None
) -> Tuple[List[Dict], Dict]:
    """
    多张图片批量推理 - 同一类别
    
    场景：在多帧视频中检测同一类物体（如"person"、"car"等）
    
    Args:
        image_paths: 图片路径列表
        prompt: 文本提示词（如"person"）
        batch_size: 批次大小
        benchmark: 性能测试工具实例
    
    Returns:
        results: 每张图片的推理结果列表
        stats: 性能统计信息
    """
    _, processor = get_model_and_processor()
    
    all_results = []
    total_inference_time = 0
    total_images = len(image_paths)
    
    print(f"\n{'='*60}")
    print(f"模式1: 多张推理/Batch Inference - 同一类别")
    print(f"  类别: {prompt}")
    print(f"  图片数量: {total_images}")
    print(f"  批次大小: {batch_size}")
    print(f"{'='*60}")
    
    clear_gpu_cache()
    mem_start = get_gpu_memory_info()
    overall_start = time.perf_counter()
    
    # 分批处理
    for batch_idx in range(0, total_images, batch_size):
        batch_end = min(batch_idx + batch_size, total_images)
        batch_paths = image_paths[batch_idx:batch_end]
        current_batch_size = len(batch_paths)
        
        print(f"\n处理批次 {batch_idx//batch_size + 1}: {current_batch_size} 张图片")
        
        # 加载批次图片
        batch_images = []
        valid_indices = []
        for idx, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                print(f"  警告: 无法加载图片 {path}: {e}")
        
        if not batch_images:
            continue
        
        # 批量设置图片特征
        batch_start_time = time.perf_counter()
        batch_states = []
        for img in batch_images:
            state = processor.set_image(img)
            batch_states.append(state)
        
        # 批量推理
        batch_results = []
        for i, state in enumerate(batch_states):
            # 深拷贝避免污染
            current_state = copy.deepcopy(state)
            
            # 设置文本提示
            current_state = processor.set_text_prompt(prompt=prompt, state=current_state)
            
            # 提取结果
            masks = current_state.get("masks", torch.tensor([]))
            boxes = current_state.get("boxes", torch.tensor([]))
            scores = current_state.get("scores", torch.tensor([]))
            
            result = {
                "image_path": batch_paths[i],
                "masks": masks.cpu().numpy() if len(masks) > 0 else [],
                "boxes": boxes.cpu().numpy() if len(boxes) > 0 else [],
                "scores": scores.cpu().tolist() if len(scores) > 0 else [],
                "prompt": prompt
            }
            batch_results.append(result)
        
        batch_end_time = time.perf_counter()
        batch_time = (batch_end_time - batch_start_time) * 1000
        total_inference_time += batch_time
        
        print(f"  批次推理时间: {batch_time:.2f} ms")
        print(f"  平均每张: {batch_time/current_batch_size:.2f} ms")
        
        all_results.extend(batch_results)
    
    overall_end = time.perf_counter()
    mem_end = get_gpu_memory_info()
    
    stats = {
        "mode": "batch_inference_same_category",
        "prompt": prompt,
        "total_images": total_images,
        "batch_size": batch_size,
        "total_time_ms": round((overall_end - overall_start) * 1000, 2),
        "inference_time_ms": round(total_inference_time, 2),
        "avg_time_per_image_ms": round(total_inference_time / total_images, 2) if total_images > 0 else 0,
        "memory_used_mb": mem_end["allocated"],
        "memory_delta_mb": round(mem_end["allocated"] - mem_start["allocated"], 2)
    }
    
    print(f"\n模式1 统计:")
    print(f"  总时间: {stats['total_time_ms']:.2f} ms")
    print(f"  推理时间: {stats['inference_time_ms']:.2f} ms")
    print(f"  平均每张: {stats['avg_time_per_image_ms']:.2f} ms")
    print(f"  显存占用: {stats['memory_used_mb']:.2f} MB")
    
    return all_results, stats


# ======================== 模式2: 单张多类别推理 ========================
# 同一frame帧关注不同的类别

def single_image_multi_category(
    image_path: str,
    prompts: List[str],
    benchmark: Optional[InferenceBenchmark] = None
) -> Tuple[Dict, Dict]:
    """
    单张图片多类别推理
    
    场景：在同一帧中检测多种不同物体（如同时检测"person"、"car"、"dog"等）
    
    Args:
        image_path: 图片路径
        prompts: 文本提示词列表
        benchmark: 性能测试工具实例
    
    Returns:
        result: 推理结果
        stats: 性能统计信息
    """
    _, processor = get_model_and_processor()
    
    print(f"\n{'='*60}")
    print(f"模式2: 单张多类别推理")
    print(f"  图片: {image_path}")
    print(f"  类别数量: {len(prompts)}")
    print(f"  类别: {prompts}")
    print(f"{'='*60}")
    
    clear_gpu_cache()
    mem_start = get_gpu_memory_info()
    overall_start = time.perf_counter()
    
    # 加载图片
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"错误: 无法加载图片 {image_path}: {e}")
        return {}, {}
    
    # 设置图片特征（只做一次）
    feature_start = time.perf_counter()
    base_state = processor.set_image(image)
    feature_end = time.perf_counter()
    feature_time = (feature_end - feature_start) * 1000
    
    print(f"\n图片特征提取时间: {feature_time:.2f} ms")
    
    # 对每个类别进行推理
    all_detections = {
        "image_path": image_path,
        "image_size": [image.width, image.height],
        "detections": []
    }
    
    inference_times = []
    
    for prompt_idx, prompt in enumerate(prompts):
        prompt_start = time.perf_counter()
        
        # 深拷贝状态，避免污染
        current_state = copy.deepcopy(base_state)
        
        # 设置文本提示
        current_state = processor.set_text_prompt(prompt=prompt, state=current_state)
        
        # 提取结果
        masks = current_state.get("masks", torch.tensor([]))
        boxes = current_state.get("boxes", torch.tensor([]))
        scores = current_state.get("scores", torch.tensor([]))
        
        prompt_end = time.perf_counter()
        prompt_time = (prompt_end - prompt_start) * 1000
        inference_times.append(prompt_time)
        
        detection = {
            "prompt": prompt,
            "masks": masks.cpu().numpy() if len(masks) > 0 else [],
            "boxes": boxes.cpu().numpy() if len(boxes) > 0 else [],
            "scores": scores.cpu().tolist() if len(scores) > 0 else [],
            "inference_time_ms": round(prompt_time, 2)
        }
        all_detections["detections"].append(detection)
        
        print(f"  类别 [{prompt}]: {len(scores)} 个目标, 耗时 {prompt_time:.2f} ms")
    
    overall_end = time.perf_counter()
    mem_end = get_gpu_memory_info()
    
    total_inference_time = sum(inference_times)
    
    stats = {
        "mode": "single_image_multi_category",
        "image_path": image_path,
        "num_categories": len(prompts),
        "prompts": prompts,
        "feature_time_ms": round(feature_time, 2),
        "total_time_ms": round((overall_end - overall_start) * 1000, 2),
        "total_inference_time_ms": round(total_inference_time, 2),
        "avg_time_per_category_ms": round(total_inference_time / len(prompts), 2) if prompts else 0,
        "memory_used_mb": mem_end["allocated"],
        "memory_delta_mb": round(mem_end["allocated"] - mem_start["allocated"], 2)
    }
    
    print(f"\n模式2 统计:")
    print(f"  特征提取: {stats['feature_time_ms']:.2f} ms")
    print(f"  总推理时间: {stats['total_inference_time_ms']:.2f} ms")
    print(f"  平均每类别: {stats['avg_time_per_category_ms']:.2f} ms")
    print(f"  显存占用: {stats['memory_used_mb']:.2f} MB")
    
    return all_detections, stats


# ======================== 部署方案分析 ========================

def analyze_deployment_options(
    batch_stats: Dict,
    multi_category_stats: Dict,
    target_fps: float = 30.0,
    num_categories: int = 3
) -> Dict:
    """
    分析部署方案
    
    Args:
        batch_stats: 批量推理统计
        multi_category_stats: 多类别推理统计
        target_fps: 目标帧率
        num_categories: 目标类别数量
    
    Returns:
        部署建议
    """
    print(f"\n{'='*60}")
    print("部署方案分析")
    print(f"{'='*60}")
    
    # 计算实际性能
    batch_fps = 1000 / batch_stats["avg_time_per_image_ms"] if batch_stats["avg_time_per_image_ms"] > 0 else 0
    multi_cat_fps = 1000 / multi_category_stats["total_inference_time_ms"] if multi_category_stats["total_inference_time_ms"] > 0 else 0
    
    print(f"\n目标帧率: {target_fps} FPS")
    print(f"目标类别数: {num_categories}")
    
    print(f"\n模式1 (批量推理) 性能:")
    print(f"  平均每张: {batch_stats['avg_time_per_image_ms']:.2f} ms")
    print(f"  等效帧率: {batch_fps:.2f} FPS")
    print(f"  显存占用: {batch_stats['memory_used_mb']:.2f} MB")
    
    print(f"\n模式2 (多类别推理) 性能:")
    print(f"  特征提取: {multi_category_stats['feature_time_ms']:.2f} ms")
    print(f"  总推理时间: {multi_category_stats['total_inference_time_ms']:.2f} ms")
    print(f"  等效帧率: {multi_cat_fps:.2f} FPS")
    print(f"  显存占用: {multi_category_stats['memory_used_mb']:.2f} MB")
    
    # 部署建议
    recommendations = []
    
    # 场景1: 视频流处理（多帧同类别）
    if batch_fps >= target_fps:
        recommendations.append({
            "scenario": "视频流处理（多帧同类别）",
            "feasible": True,
            "method": "批量推理模式",
            "expected_fps": batch_fps,
            "notes": f"可满足 {target_fps} FPS 要求，建议 batch_size={batch_stats.get('batch_size', 4)}"
        })
    else:
        recommendations.append({
            "scenario": "视频流处理（多帧同类别）",
            "feasible": False,
            "method": "批量推理模式",
            "expected_fps": batch_fps,
            "notes": f"无法满足 {target_fps} FPS，建议：1) 减小分辨率 2) 使用TensorRT加速 3) 多GPU并行"
        })
    
    # 场景2: 单帧多目标检测
    if multi_cat_fps >= target_fps:
        recommendations.append({
            "scenario": "单帧多目标检测",
            "feasible": True,
            "method": "多类别推理模式",
            "expected_fps": multi_cat_fps,
            "notes": f"可满足 {target_fps} FPS 要求，支持 {num_categories} 个类别同时检测"
        })
    else:
        recommendations.append({
            "scenario": "单帧多��标检测",
            "feasible": False,
            "method": "多类别推理模式",
            "expected_fps": multi_cat_fps,
            "notes": f"无法满足 {target_fps} FPS，建议：1) 减少类别数 2) 使用模型量化 3) 异步流水线处理"
        })
    
    # 显存分析
    gpu_total = get_gpu_memory_info()["total"]
    memory_usage = max(batch_stats["memory_used_mb"], multi_category_stats["memory_used_mb"])
    memory_ratio = memory_usage / gpu_total if gpu_total > 0 else 0
    
    print(f"\n显存分析:")
    print(f"  GPU总显存: {gpu_total:.2f} MB")
    print(f"  推理占用: {memory_usage:.2f} MB ({memory_ratio*100:.1f}%)")
    
    if memory_ratio > 0.8:
        print("  警告: 显存占用过高，可能导致OOM")
    elif memory_ratio > 0.5:
        print("  注意: 显存占用中等，建议监控")
    else:
        print("  状态: 显存占用正常")
    
    print(f"\n部署建议:")
    for i, rec in enumerate(recommendations, 1):
        status = "✅ 可行" if rec["feasible"] else "❌ 需优化"
        print(f"\n{i}. {rec['scenario']}: {status}")
        print(f"   方法: {rec['method']}")
        print(f"   预期帧率: {rec['expected_fps']:.2f} FPS")
        print(f"   说明: {rec['notes']}")
    
    return {
        "batch_fps": batch_fps,
        "multi_category_fps": multi_cat_fps,
        "memory_usage_mb": memory_usage,
        "memory_ratio": memory_ratio,
        "recommendations": recommendations
    }


# ======================== 主函数 ========================

def main():
    """主函数：演示两种推理模式并分析部署方案"""
    
    print("=" * 60)
    print("SAM3 推理示例 - 性能测试与部署分析")
    print("=" * 60)
    
    # 初始化性能测试工具
    benchmark = InferenceBenchmark()
    
    # 准备测试图片
    test_images_dir = f"{sam3_root}/assets/images"
    
    # 获取测试图片列表
    image_extensions = ['.jpg', '.jpeg', '.png']
    test_images = []
    if os.path.exists(test_images_dir):
        for f in os.listdir(test_images_dir):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                test_images.append(os.path.join(test_images_dir, f))
    
    if not test_images:
        print("警告: 未找到测试图片，使用示例图片")
        # 使用默认测试图片
        test_images = [f"{sam3_root}/assets/images/test_image.jpg"]
    
    # 限制测试图片数量
    test_images = test_images[:8]
    print(f"\n测试图片数量: {len(test_images)}")
    
    # ==================== 模式1测试 ====================
    print("\n" + "=" * 60)
    print("开始模式1测试: 多张推理/Batch Inference")
    print("=" * 60)
    
    batch_results, batch_stats = batch_inference_same_category(
        image_paths=test_images,
        prompt="person",  # 检测人物
        batch_size=4,
        benchmark=benchmark
    )
    
    # ==================== 模式2测试 ====================
    print("\n" + "=" * 60)
    print("开始模式2测试: 单张多类别推理")
    print("=" * 60)
    
    # 使用第一张测试图片
    test_image = test_images[0] if test_images else f"{sam3_root}/assets/images/test_image.jpg"
    
    multi_category_results, multi_category_stats = single_image_multi_category(
        image_path=test_image,
        prompts=["person", "car", "dog"],  # 同时检测多种物体
        benchmark=benchmark
    )
    
    # ==================== 部署方案分析 ====================
    deployment_analysis = analyze_deployment_options(
        batch_stats=batch_stats,
        multi_category_stats=multi_category_stats,
        target_fps=30.0,
        num_categories=3
    )
    
    # ==================== 汇总报告 ====================
    print("\n" + "=" * 60)
    print("最终汇总报告")
    print("=" * 60)
    
    print(f"\n模式1 (批量推理):")
    print(f"  处理图片: {batch_stats['total_images']} 张")
    print(f"  批次大小: {batch_stats['batch_size']}")
    print(f"  平均耗时: {batch_stats['avg_time_per_image_ms']:.2f} ms/张")
    print(f"  等效帧率: {deployment_analysis['batch_fps']:.2f} FPS")
    
    print(f"\n模式2 (多类别推理):")
    print(f"  类别数量: {multi_category_stats['num_categories']}")
    print(f"  总耗时: {multi_category_stats['total_inference_time_ms']:.2f} ms")
    print(f"  平均每类别: {multi_category_stats['avg_time_per_category_ms']:.2f} ms")
    print(f"  等效帧率: {deployment_analysis['multi_category_fps']:.2f} FPS")
    
    print(f"\n显存使用:")
    print(f"  占用: {deployment_analysis['memory_usage_mb']:.2f} MB")
    print(f"  比例: {deployment_analysis['memory_ratio']*100:.1f}%")
    
    # 输出JSON格式的结果（便于程序解析）
    import json
    final_results = {
        "batch_inference": {
            "stats": batch_stats,
            "fps": deployment_analysis["batch_fps"]
        },
        "multi_category_inference": {
            "stats": multi_category_stats,
            "fps": deployment_analysis["multi_category_fps"]
        },
        "deployment_analysis": deployment_analysis
    }
    
    print("\n" + "=" * 60)
    print("JSON结果:")
    print(json.dumps(final_results, indent=2, ensure_ascii=False, default=str))
    
    return final_results


if __name__ == "__main__":
    # 启用TF32加速（适用于Ampere架构GPU）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 使用bfloat16精度（可选，可提升速度）
    # with torch.autocast("cuda", dtype=torch.bfloat16):
    #     main()
    
    main()