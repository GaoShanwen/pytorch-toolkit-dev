# pytorch-toolkit-dev
building a multitask training platform based on Pytorch

## Introduction

yolo-dev branch for object detection tasks.

[![](https://img.shields.io/badge/Python-3.9.12-3776AB?style=flat-square&logo=python&logoColor=FFFFFF)](https://www.python.org)
[![](https://img.shields.io/badge/PyTorch-2.7.1+cu12-EE4C2C?style=flat-square&logo=pytorch&logoColor=FFFFFF)](https://pytorch.org)
[![](https://img.shields.io/badge/Ultralytics-8.1.47-yellow.svg)](https://docs.ultralytics.com/zh/)
[![](https://img.shields.io/badge/ONNX-1.14.0-005CED?style=flat-square&logo=ONNX&logoColor=FFFFFF)](https://onnx.ai)

<div align="center";style="display: none;">
    <img src="docs/demo4yolo.jpg" width="70%" alt="" />
</div>

### Supported Functions

- [x] Detect/Segment/Track object by YOLOv[8-11].
- [x] Convert model to ONNX/Tensorrt.
- [x] Support videos and images' inference.
- [x] Support ignore regions for detect task.
- [x] Support visualized badcase.

### Major Motivation

1.👀 Use the State-of-the-Art AI-detection toolkit.
 + Various backbones and pretrained models
 + Bag of training tricks
 + Large-scale training configs
 + High efficiency and extensibility
 + Powerful toolkits

2.🚀 Enhance codes' reusability.

3.🛠️ Minimize our project.

## Quick Start

### Install Environment

For detailed installation guides, please refer to [INSTALL.md](docs/INSTALL.md).

### Command Guides

+ Train & Validate with Training Curve

Refer to [README-Train/Val Models](tools/README.md) for details.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Contribute

Feel free to create a pull request if you want to contribute (e.g. networks or tricks).
