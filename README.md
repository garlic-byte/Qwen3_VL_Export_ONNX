# qwen3-vl-2b-ONNX: Image-to-Text Inference Model

## Overview
This repository provides the ONNX-converted version of the **qwen3-vl-2b** multimodal model, optimized for efficient image-to-text generation. The model supports inference on single images with a fixed input resolution of 224×224 and outputs descriptive text based on visual content.

## Key Features
- **Model Type**: ONNX-exported multimodal large language model (vision-language)
- **Input Specification**: Single RGB image (224×224 resolution, 3 channels)
- **Output**: High-quality natural language description of the input image
- **Conversion Source**: Original qwen3-vl-2b (PyTorch) → ONNX format

## Inference Example
### Input
![Image](demo_data/input1.jpg)
- **Version**: Single RGB image (224×224) of a lemon.
- **Language**: Describe this image.

### Output
```
This image shows a single, yellow, spherical object that appears to be a small, smooth, and rounded lemon. It is placed on a light-colored, possibly white or off-white, surface with a wood grain texture. The lemon has a rounded, slightly flattened top and a smooth surface. The lighting is even, and the object is the central focus of the image.
```

## Requirements
- Will be supplemented.

## Next task
- Adapt images of different sizes
- Comparison of Test Torch and ONNX inference Speed
- Convert ONNX to TensorRT to further improve inference speed
- Convert more models from Torch to ONNX


## Usage
### 1. Download Qwen3-VL
```bash
# Download the model
hf download Qwen/Qwen3-VL-2B-Instruct
```

### 2. Conert Torch to ONNX
```bash
python qwen3_vl_export_onnx.py
```

### 3. Run Inference with ONNX Runtime
```bash
python inference_onnx.py
```

## Model Conversion Notes
- The ONNX model is exported from the original PyTorch implementation of qwen3-vl-2b.
- Input resolution is fixed at 224×224 (consistent with the model's training configuration).
- For optimal performance, use ONNX Runtime with GPU acceleration (install `onnxruntime-gpu` instead of `onnxruntime`).
- The model retains the original qwen3-vl-2b's visual understanding and text generation capabilities.

## Performance
- Will be supplemented.
- **Latency**: 
- **Accuracy**: 
- **Memory Usage**:
- 

## License
The model is licensed under the same license as the original qwen3-vl-2b (see [Qwen Official Repository](https://github.com/QwenLM/Qwen) for details).

## Acknowledgements
- Original qwen3-vl-2b model developed by Alibaba Cloud.
- ONNX conversion leverages PyTorch's `torch.onnx.export` API and ONNX Runtime for inference optimization.
