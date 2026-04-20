# SAM3 模型导出工具

## 文件说明

| 文件 | 功能 | 成功率 |
|------|------|--------|
| `export_guaranteed.py` | 导出完整模型权重 (.pth) | 100% |
| `export_onnx_modules.py` | 分模块导出 ONNX | 视模块而定 |

## 方案一：导出完整权重（推荐）

```bash
python tools/export_guaranteed.py
```

**导出内容：**
- `sam3_full_weights.pth` - 完整模型权重
- `sam3_config.json` - 模型配置
- `sam3_backbone_traced.pt` - TorchScript（可能）

## 方案二：分模块导出 ONNX

```bash
# 导出所有模块
python tools/export_onnx_modules.py --output-dir ./onnx_models

# 只导出图像编码器（成功率最高）
python tools/export_onnx_modules.py --skip-text --skip-decoder

# 只导出文本编码器
python tools/export_onnx_modules.py --skip-vision --skip-decoder
```

**三个模块：**

| 模块 | 文件 | 成功率 | 说明 |
|------|------|--------|------|
| Vision Encoder | `sam3_vision_encoder_bs1.onnx` | 高 | ViT + FPN |
| Text Encoder | `sam3_text_encoder_bs1.onnx` | 中 | 文本编码 |
| Fusion Decoder | `sam3_fusion_decoder_bs1.onnx` | 低 | Transformer + Head |

## 加载导出的模型

### 加载权重
```python
import torch
from sam3.model_builder import build_sam3_image_model

model = build_sam3_image_model(bpe_path="path/to/bpe")
checkpoint = torch.load('exported_sam3/sam3_full_weights.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 加载 ONNX
```python
import onnxruntime as ort

# 加载图像编码器
vision_session = ort.InferenceSession("onnx_models/sam3_vision_encoder_bs1.onnx")
vision_features = vision_session.run(None, {"image": image_numpy})

# 加载文本编码器
text_session = ort.InferenceSession("onnx_models/sam3_text_encoder_bs1.onnx")
text_features = text_session.run(None, {"input_ids": ids, "attention_mask": mask})
```

## 关于 TensorRT

SAM3 模型**不能直接**转换为 TensorRT，原因：

1. Deformable Attention 算子不支持
2. 动态输出数量
3. 自定义 CUDA 算子

### 推荐的加速方案

```
PyTorch → ONNX Runtime (1.5-2x) → Torch-TensorRT (2-3x)
```
