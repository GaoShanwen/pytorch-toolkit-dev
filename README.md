# pytorch-toolkit-dev
building a multitask training platform based on Pytorch

## Introduction

yolo-dev branch for object detection tasks.

[![](https://img.shields.io/badge/Python-3.8.18-3776AB?style=flat-square&logo=python&logoColor=FFFFFF)](https://www.python.org)
[![](https://img.shields.io/badge/PyTorch-1.10.1+cu102-EE4C2C?style=flat-square&logo=pytorch&logoColor=FFFFFF)](https://pytorch.org)
[![](https://img.shields.io/badge/Ultralytics-8.0.227-yellow.svg)](https://docs.ultralytics.com/zh/)
[![](https://img.shields.io/badge/ONNX-1.14.0-005CED?style=flat-square&logo=ONNX&logoColor=FFFFFF)](https://onnx.ai)

### Supported Functions

- [x] Detect object by YOLOv5/YOLOv6/YOLOv8.
- [ ] Convert model to ONNX/RKNN.
- [x] Normalize codes before commit.

### Major Motivation

1.👀 Use the State-of-the-Art AI-classfication toolkit.
 + Various backbones and pretrained models
 + Bag of training tricks
 + Large-scale training configs
 + High efficiency and extensibility
 + Powerful toolkits

2.🚀 Enhance codes' reusability.

3.🛠️ Minimize our project.

## Quick Start

### Install Environment

For detailed installation guides, please refer to [INSTALL.md](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/yolo-dev/docs/environment.md).

### Command Guides

+ Train & Validate with Training Curve

Refer to [README-Train/Val Models](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/tools/README.md) for details.

+ Normalize Codes Before Commit

```bash
<pytorch-toolkit-dev> ~$ # run this command after install black
<pytorch-toolkit-dev> ~$ black --line-length=120 ./
```

## License

This project is released under the [Apache 2.0 license](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/yolo-dev/LICENSE).

## Contribute

Feel free to create a pull request if you want to contribute (e.g. networks or tricks).
