import os
import time
import torch
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

import sam3
from sam3 import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.visualization_utils import plot_results

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")


def create_empty_datapoint():
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint, pil_image):
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint, text_query, query_id):
    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )


def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def get_gpu_memory_summary():
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        }
    return None


def measure_inference_speed(func, *args, warmup=2, runs=5, **kwargs):
    for _ in range(warmup):
        func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "result": result,
    }


def batch_inference_same_category():
    print("=" * 60)
    print("1. 多张推理/batch inference - 不同frame帧关注同样的类别")
    print("=" * 60)

    print("\n[1] Building model...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    bpe_path = f"assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008, max_size=1008, square=True, consistent_transform=False
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False,
    )

    print(f"\n[2] Preparing batch with same category prompt...")
    print(f"    GPU Memory before loading images: {get_memory_usage():.2f} GB")

    image_path = f"To_Base64/truck.jpg"
    base_image = Image.open(image_path)

    categories = ["person", "car", "bicycle"]

    datapoints = []
    query_ids = []
    query_id_counter = 1

    for i, category in enumerate(categories):
        dp = create_empty_datapoint()
        set_image(dp, base_image)
        add_text_prompt(dp, category, query_id_counter)
        dp = transform(dp)
        datapoints.append(dp)
        query_ids.append(query_id_counter)
        print(f"    Frame {i + 1}: category='{category}', query_id={query_id_counter}")
        query_id_counter += 1

    print(f"\n[3] Memory after preparing datapoints: {get_memory_usage():.2f} GB")

    print("\n[4] Collating batch and moving to GPU...")
    batch = collate(datapoints, dict_key="dummy")["dummy"]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
    print(f"    GPU Memory after moving to GPU: {get_memory_usage():.2f} GB")

    print("\n[5] Running batch inference...")
    print(f"    Batch size: {len(datapoints)} images")

    result = measure_inference_speed(model, batch, warmup=2, runs=5)

    print(f"\n[6] Inference Speed Results:")
    print(f"    Mean time: {result['mean'] * 1000:.2f} ms per batch")
    print(f"    Std time:  {result['std'] * 1000:.2f} ms")
    print(f"    Min time:  {result['min'] * 1000:.2f} ms")
    print(f"    Max time:  {result['max'] * 1000:.2f} ms")
    print(f"    Per image: {result['mean()'] * 1000 / len(datapoints):.2f} ms")

    print(f"\n[7] Memory Usage:")
    mem_summary = get_gpu_memory_summary()
    print(f"    Currently allocated: {mem_summary['allocated']:.2f} GB")
    print(f"    Reserved: {mem_summary['reserved']:.2f} GB")
    print(f"    Peak allocated: {mem_summary['max_allocated']:.2f} GB")

    print("\n[8] Processing results...")
    output = result["result"]
    processed_results = postprocessor.process_results(output, batch.find_metadatas)

    for i, (cat, qid) in enumerate(zip(categories, query_ids)):
        if qid in processed_results:
            result_data = processed_results[qid]
            n_objects = len(result_data.get("scores", []))
            print(f"    Frame {i + 1} ('{cat}'): found {n_objects} object(s)")

    return {
        "inference_time": result,
        "memory": mem_summary,
        "results": processed_results,
        "batch_size": len(datapoints),
    }


def single_image_multi_category():
    print("\n" + "=" * 60)
    print("2. 单张多类别推理 - 同一frame帧关注不同的类别")
    print("=" * 60)

    print("\n[1] Building model...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.inference_mode().__enter__()

    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=1008, max_size=1008, square=True, consistent_transform=False
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.5,
        to_cpu=False,
    )

    print(f"\n[2] Preparing single image with multiple category prompts...")
    print(f"    GPU Memory before loading: {get_memory_usage():.2f} GB")

    image_path = f"{sam3_root}/assets/images/test_image.jpg"
    base_image = Image.open(image_path)

    categories = ["person", "car", "bicycle", "dog"]

    dp = create_empty_datapoint()
    set_image(dp, base_image)

    query_ids = []
    query_id_counter = 100

    for cat in categories:
        add_text_prompt(dp, cat, query_id_counter)
        query_ids.append(query_id_counter)
        print(f"    Category: '{cat}', query_id={query_id_counter}")
        query_id_counter += 1

    dp = transform(dp)

    print(f"\n[3] Memory after preparing datapoint: {get_memory_usage():.2f} GB")

    print("\n[4] Collating and moving to GPU...")
    batch = collate([dp], dict_key="dummy")["dummy"]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
    print(f"    GPU Memory after moving to GPU: {get_memory_usage():.2f} GB")

    print("\n[5] Running inference...")
    print(f"    Number of categories: {len(categories)}")

    result = measure_inference_speed(model, batch, warmup=2, runs=5)

    print(f"\n[6] Inference Speed Results:")
    print(f"    Mean time: {result['mean'] * 1000:.2f} ms")
    print(f"    Std time:  {result['std'] * 1000:.2f} ms")
    print(f"    Min time:  {result['min'] * 1000:.2f} ms")
    print(f"    Max time:  {result['max'] * 1000:.2f} ms")

    print(f"\n[7] Memory Usage:")
    mem_summary = get_gpu_memory_summary()
    print(f"    Currently allocated: {mem_summary['allocated']:.2f} GB")
    print(f"    Reserved: {mem_summary['reserved']:.2f} GB")
    print(f"    Peak allocated: {mem_summary['max_allocated']:.2f} GB")

    print("\n[8] Processing results...")
    output = result["result"]
    processed_results = postprocessor.process_results(output, batch.find_metadatas)

    for cat, qid in zip(categories, query_ids):
        if qid in processed_results:
            result_data = processed_results[qid]
            n_objects = len(result_data.get("scores", []))
            print(f"    '{cat}': found {n_objects} object(s)")

    return {
        "inference_time": result,
        "memory": mem_summary,
        "results": processed_results,
        "n_categories": len(categories),
    }


def compare_approaches():
    print("\n" + "=" * 60)
    print("3. 方案对比分析")
    print("=" * 60)

    batch_result = batch_inference_same_category()
    single_result = single_image_multi_category()

    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)

    print(f"\nbatch inference ({batch_result['batch_size']}张图, 不同类别):")
    print(f"  推理时间: {batch_result['inference_time']['mean'] * 1000:.2f} ms/batch")
    print(
        f"  单张时间: {batch_result['inference_time']['mean'] * 1000 / batch_result['batch_size']:.2f} ms"
    )
    print(f"  峰值显存: {batch_result['memory']['max_allocated']:.2f} GB")

    print(
        f"\nsingle image multi-category (1张图, {single_result['n_categories']}个类别):"
    )
    print(f"  推理时间: {single_result['inference_time']['mean'] * 1000:.2f} ms")
    print(f"  峰值显存: {single_result['memory']['max_allocated']:.2f} GB")


if __name__ == "__main__":
    print("SAM3 Inference Sample")
    print("功能: 多张推理/batch inference & 单张多类别推理")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. GPU memory tracking will be limited.")

    compare_approaches()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
