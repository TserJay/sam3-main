"""推理模块"""

from .test_picture_to_picture_optimizer_batch_inference import this_main as batch_inference_main
from .test_picture_to_picture_optimizer import this_main as single_inference_main
from .test_words_picture import words_to_picture
from .word_picture_batch_inference import this_main as word_batch_inference_main
from .picture_to_picture_optimizer import this_main as picture_to_picture_main

__all__ = [
    "batch_inference_main",
    "single_inference_main", 
    "words_to_picture",
    "word_batch_inference_main",
    "picture_to_picture_main",
]
