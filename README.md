# pytorch-toolkit-dev
building a multitask training platform based on Pytorch

## Introduction

poserefine-dev branch for pose estimation tasks.

[![](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=FFFFFF)](https://www.python.org)
[![](https://img.shields.io/badge/PyTorch-2.3.1+cu121-EE4C2C?style=flat-square&logo=pytorch&logoColor=FFFFFF)](https://pytorch.org)
[![](https://img.shields.io/badge/MMpose-1.3.0-005CED?style=flat-square&logo=MMpose&logoColor=FFFFFF)](https://mmpose.readthedocs.io/)
[![](https://img.shields.io/badge/ONNX-1.14.0-005CED?style=flat-square&logo=ONNX&logoColor=FFFFFF)](https://onnx.ai)

### Supported Functions

- [x] Pose estimation with RTMPose.
- [x] Symmetry matching for pose keypoints.
- [x] Support videos and images' inference.
- [x] Convert model to ONNX/TensorRT.

### Major Motivation

1.👀 Use State-of-the-Art pose estimation models.
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