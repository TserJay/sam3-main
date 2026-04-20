"""
SAM3 扩展模块 - 业务封装层
基于 sam3 核心库的二次开发接口
"""

from .inference import BatchDetector, SingleDetector, WordDetector, WordBatchDetector
from .services import DetectorService, FeatureStore
from .api import create_app

__version__ = "1.0.0"
__all__ = [
    "BatchDetector",
    "SingleDetector", 
    "WordDetector",
    "WordBatchDetector",
    "DetectorService",
    "FeatureStore",
    "create_app",
]
