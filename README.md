# pytorch-toolkit-dev
building a multitask training platform based on Pytorch

## Introduction

rfdetr-dev branch for object detection and instance segmentation using [RF-DETR](https://github.com/roboflow/rf-detr).

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=FFFFFF)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1+cu12-EE4C2C?style=flat-square&logo=pytorch&logoColor=FFFFFF)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-1.14.0-005CED?style=flat-square&logo=ONNX&logoColor=FFFFFF)](https://onnx.ai)

### Supported Functions

- [x] Detect objects with RF-DETR (nano / small / medium / large / xlarge / 2xlarge).
- [x] Instance segmentation with RF-DETR-Seg.
- [x] Mixed-data training with class remapping and background-image mixing.
- [x] Convert model to ONNX / TensorRT.
- [x] Inference on videos and images.

### Quick Start

```bash
# Plain training
sh tools/train.sh my_dataset '' 100 4 640 detect medium

# Training with class remapping + background mixing
python tools/train.py \
    --data data/det-dataset/my_dataset \
    --epochs 100 --batch 4 --imgsz 640 --model medium \
    --class_mapping 5:0,6:1,7:2 \
    --mixed_alpha 0.06 \
    --background_data data/background.txt
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
