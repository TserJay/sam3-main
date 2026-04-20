from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from typing import List
import torch
import cv2

GLOBAL_COUNTER = 1


def create_empty_datapoint():
    """ A datapoint is a single image on which we can apply several queries at once. """
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint, pil_image):
    """ Add the image to be processed to the datapoint """
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint, text_query):
    """ Add a text query to the datapoint """

    global GLOBAL_COUNTER
    # in this function, we require that the image is already set.
    # that's because we'll get its size to figure out what dimension to resize masks and boxes
    # In practice you're free to set any size you want, just edit the rest of the function
    assert len(datapoint.images) == 1, "please set the image first"

    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],  # unused for inference
            is_exhaustive=True,  # unused for inference
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1


def add_visual_prompt(datapoint, boxes: List[List[float]], labels: List[bool], text_prompt="visual"):
    """ Add a visual query to the datapoint.
    The bboxes are expected in XYXY format (top left and bottom right corners)
    For each bbox, we expect a label (true or false). The model tries to find boxes that ressemble the positive ones while avoiding the negative ones
    We can also give a text_prompt as an additional hint. It's not mandatory, leave it to "visual" if you want the model to solely rely on the boxes.

    Note that the model expects the prompt to be consistent. If the text reads "elephant" but the provided boxe points to a dog, the results will be undefined.
    """

    global GLOBAL_COUNTER
    # in this function, we require that the image is already set.
    # that's because we'll get its size to figure out what dimension to resize masks and boxes
    # In practice you're free to set any size you want, just edit the rest of the function
    assert len(datapoint.images) == 1, "please set the image first"
    assert len(boxes) > 0, "please provide at least one box"
    assert len(boxes) == len(labels), f"Expecting one label per box. Found {len(boxes)} boxes but {len(labels)} labels"
    for b in boxes:
        assert len(b) == 4, f"Boxes must have 4 coordinates, found {len(b)}"

    labels = torch.tensor(labels, dtype=torch.bool).view(-1)
    if not labels.any().item() and text_prompt == "visual":
        print(
            "Warning: you provided no positive box, nor any text prompt. The prompt is ambiguous and the results will be undefined")
    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_prompt,
            image_id=0,
            object_ids_output=[],  # unused for inference
            is_exhaustive=True,  # unused for inference
            query_processing_order=0,
            input_bbox=torch.tensor(boxes, dtype=torch.float).view(-1, 4),
            input_bbox_label=labels,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1


if __name__ == '__main__':
    from sam3.eval.postprocessors import PostProcessImage

    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        # if this number is positive, the processor will return topk. For this demo we instead limit by confidence, see below
        iou_type="segm",  # we want masks
        use_original_sizes_box=True,  # our boxes should be resized to the image size
        use_original_sizes_mask=True,  # our masks should be resized to the image size
        convert_mask_to_rle=False,
        # the postprocessor supports efficient conversion to RLE format. In this demo we prefer the binary format for easy plotting
        detection_threshold=0.5,  # Only return confident detections
        to_cpu=False,
    )
    from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI

    from sam3.model.position_encoding import PositionEmbeddingSine

    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    import torch
    #################################### For Image ####################################
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    import time

    # Load the model
    # model = build_sam3_image_model(checkpoint_path=r'/mnt/mys40/sam3/checkpoints/sam3.pt',load_from_HF=False)
    model = build_sam3_image_model(checkpoint_path=r'checkpoints/sam3.pt')
    processor = Sam3Processor(model)

    start = time.time()
    # Load an image
    img1 = Image.open("/mnt/mys40/sam3/frame_0020.jpg")
    datapoint1 = create_empty_datapoint()
    set_image(datapoint1, img1)
    id1 = add_text_prompt(datapoint1, "yellow object")
    id2 = add_text_prompt(datapoint1, "white object")

    datapoint1 = transform(datapoint1)

    # Collate then move to cuda
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.model.utils.misc import copy_data_to_device

    batch = collate([datapoint1], dict_key="dummy")["dummy"]
    batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)

    # Forward. Note that the first forward will be very slow due to compilation
    output = model(batch)
    processed_results = postprocessor.process_results(output, batch.find_metadatas)

    end = time.time()

    print("推理时间：", end - start)
    # print(processed_results[id1])
    # print(processed_results[id2])

    # 图像可视化
    # from sam3.visualization_utils import plot_results
    # plot_results(img1, processed_results[id1])