# pytorch-toolkit-dev

building a multitask training platform based on Pytorch

## Introduction

timm-dev branch for classfication or re-identification tasks.

[![](https://img.shields.io/badge/Python-3.8.18-3776AB?style=flat-square&logo=python&logoColor=FFFFFF)](https://www.python.org)
[![](https://img.shields.io/badge/PyTorch-1.10.1+cu102-EE4C2C?style=flat-square&logo=pytorch&logoColor=FFFFFF)](https://pytorch.org)
[![](https://img.shields.io/badge/Timm-0.9.7-blue.svg?style=flat-square&logo=github&logoColor=FFFFFF)](https://github.com/huggingface/pytorch-image-models/tree/main)
[![](https://img.shields.io/badge/ONNX-1.14.0-005CED?style=flat-square&logo=ONNX&logoColor=FFFFFF)](https://onnx.ai)
[![](https://img.shields.io/badge/PyMySQL-1.1.0-FFBE00?style=flat-square&logo=mysql&logoColor=FFFFFF)](https://zetcode.com/python/pymysql/)

<div align="center">
    <img src="https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/docs/demo4reid.jpg" width="70%" alt="" />
</div>

### Supported Functions

- [x] Classfication or re-identification tasks.
- [x] Visualize training curve (wandb / tensorboard).
- [x] Remove similarity or noise data, weighted k-nearest neighbor for reid.
- [ ] Visualize precision-recall / receiver-operating-characteristic curve.
- [ ] Mixed-Precision Training for faster speed.
- [x] Visualize models' heatmaps.
- [x] Convert pth model to onnx/rknn.
- [x] Normalize codes before commit.

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

Refer to [README-Convert Models](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/tools/post/README.md) for details.

+ Normalize Codes Before Commit

```bash
<pytorch-toolkit-dev> ~$ # run this command after install black
<pytorch-toolkit-dev> ~$ black --line-length=120 ./
```

## License

This project is released under the [Apache 2.0 license](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/LICENSE).

## Contribute

Feel free to create a pull request if you want to contribute (e.g. networks or tricks).
