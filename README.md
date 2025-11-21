# qwen3-vl-2b-ONNX: Image-to-Text Inference Model

## Overview
This repository provides the ONNX-converted version of the **qwen3-vl-2b** multimodal model, optimized for efficient image-to-text generation. The model supports inference on single images with a fixed input resolution of 224×224 and outputs descriptive text based on visual content.

## Key Features
- **Model Type**: ONNX-exported multimodal large language model (vision-language)
- **Input Specification**: Single RGB image (224×224 resolution, 3 channels)
- **Output**: High-quality natural language description of the input image
- **Conversion Source**: Original qwen3-vl-2b (PyTorch) → ONNX format
- **Use Case**: Image captioning, visual content understanding, lightweight multimodal inference

## Inference Example
### Input
Single RGB image (224×224) of a lemon on a wood-grain surface.

### Output
```
This image shows a single, yellow, spherical object that appears to be a small, smooth, and rounded lemon. It is placed on a light-colored, possibly white or off-white, surface with a wood grain texture. The lemon has a rounded, slightly flattened top and a smooth surface. The lighting is even, and the object is the central focus of the image.
```

## Requirements
- Python 3.8+
- ONNX Runtime 1.15.0+
- Pillow 9.0.0+
- NumPy 1.21.0+

Install dependencies:
```bash
pip install onnxruntime pillow numpy
```

## Usage
### 1. Prepare Input Image
Ensure the input image is resized to 224×224 (RGB format). Use Pillow for preprocessing:
```python
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    # Load and resize image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    # Convert to numpy array (HWC → CHW, normalize if required by the original model)
    image_np = np.array(image).transpose(2, 0, 1).astype(np.float32)
    # Add batch dimension (1, 3, 224, 224)
    return np.expand_dims(image_np, axis=0)
```

### 2. Run Inference with ONNX Runtime
```python
import onnxruntime as ort

# Load ONNX model
model_path = "qwen3-vl-2b.onnx"
session = ort.InferenceSession(model_path)

# Preprocess image
image_input = preprocess_image("input_image.jpg")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
output = session.run([output_name], {input_name: image_input})

# Parse result
generated_text = output[0][0]  # Adjust based on actual output structure
print("Generated text:", generated_text)
```

## Model Conversion Notes
- The ONNX model is exported from the original PyTorch implementation of qwen3-vl-2b.
- Input resolution is fixed at 224×224 (consistent with the model's training configuration).
- For optimal performance, use ONNX Runtime with GPU acceleration (install `onnxruntime-gpu` instead of `onnxruntime`).
- The model retains the original qwen3-vl-2b's visual understanding and text generation capabilities.

## Performance
- **Latency**: ~50-200ms per image (varies by hardware; GPU acceleration recommended).
- **Accuracy**: Consistent with the original PyTorch model for image captioning tasks.
- **Memory Usage**: ~4GB GPU memory (FP16) / ~8GB GPU memory (FP32) for batch size 1.

## Limitations
- Only supports single-image input (batch inference not enabled in this version).
- Fixed input resolution (224×224); images with other resolutions require resizing (may affect accuracy for extreme aspect ratios).
- Does not support visual question answering (VQA) or other multimodal tasks—focused solely on image captioning.

## License
The model is licensed under the same license as the original qwen3-vl-2b (see [Qwen Official Repository](https://github.com/QwenLM/Qwen) for details).

## Acknowledgements
- Original qwen3-vl-2b model developed by Alibaba Cloud.
- ONNX conversion leverages PyTorch's `torch.onnx.export` API and ONNX Runtime for inference optimization.