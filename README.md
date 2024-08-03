# pytorch-toolkit-dev

building a multitask training platform based on Pytorch

## Introduction

[![](https://img.shields.io/badge/Python-3.8.18-3776AB?style=flat-square&logo=python&logoColor=FFFFFF)](https://www.python.org)
[![](https://img.shields.io/github/v/release/pytorch/pytorch?style=flat-square&logo=pytorch&logoColor=FFFFFF&label=PyTorch&color=EE4C2C)](https://pytorch.org)
[![](https://img.shields.io/github/v/release/huggingface/pytorch-image-models.svg?style=flat-square&logo=github&color=black&label=Timm)](https://github.com/huggingface/pytorch-image-models/tree/main)
[![](https://img.shields.io/github/v/release/onnx/onnx.svg?style=flat-square&logo=ONNX&logoColor=FFFFFF&label=ONNX&color=005CED)](https://onnx.ai)
[![](https://img.shields.io/github/v/release/PyMySQL/PyMySQL.svg?style=flat-square&logo=mysql&logoColor=FFFFFF&label=PyMySQL&color=FFBE00)](https://zetcode.com/python/pymysql/)

timm-dev branch for classfication or re-identification tasks.

<div align="center";style="display: none;">
    <img src="./docs/demo4reid.jpg" width="70%" alt="" />
</div>

### Supported Functions

- [x] Classfication (Single-label, Multi-label) and Re-identification tasks.
- [ ] Metric learning for Re-identification.
- [x] Visualize training curve (wandb / tensorboard).
- [x] Remove similarity or noise data, weighted k-nearest neighbor for reid.
- [ ] Visualize Precision-Recall / Receiver-Operating-Characteristic curve.
- [x] Mixed-Precision Training for faster speed.
- [x] Visualize models' heatmaps and U-MAPs.
- [x] Convert pth model to onnx / rknn / ncnn format.

### Major Motivation

1.👀 Use the State-of-the-Art image classfication toolkit.
 + Various backbones and pretrained models
 + Bag of training tricks
 + Large-scale training configs
 + High efficiency and extensibility
 + Powerful toolkits

2.🚀 Enhance codes' reusability.

3.🛠️ Minimize our project.

## Quick Start

### Install Environment

For detailed installation guides, please refer to [INSTALL.md](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/docs/INSTALL.md).

### Command Guides

+ Train & Validate with Training Curve

Refer to [README-Train/Val Models](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/tools/README.md) for details.

+ Convert Pth Model To ONNX/RKNN

Refer to [README-Convert Models](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/tools/deploy/README.md) for details.

## License

This project is released under the [Apache 2.0 license](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/LICENSE).

## Contribute

Feel free to create a pull request if you want to contribute (e.g. networks or tricks).
