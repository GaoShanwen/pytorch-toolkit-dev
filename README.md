# pytorch-toolkit-dev

building a multitask training platform based on Pytorch

## Introduction

[![](https://img.shields.io/badge/Python-3.10-3776AB?style=plastic&logo=python&logoColor=FFFFFF)](https://www.python.org)
[![](https://img.shields.io/github/v/release/pytorch/pytorch?style=plastic&logo=pytorch&logoColor=FFFFFF&label=PyTorch&color=EE4C2C)](https://pytorch.org)
[![](https://img.shields.io/github/v/release/qubvel/segmentation_models.pytorch.svg?style=plastic&logo=segmentation&logoColor=FFFFFF&label=SMP&color=9B59B6)](https://github.com/qubvel/segmentation_models.pytorch)
[![](https://img.shields.io/github/v/release/albumentations-team/albumentations.svg?style=plastic&logo=albumentations&logoColor=FFFFFF&label=Albumentations&color=42A5F5)](https://github.com/albumentations-team/albumentations)
[![](https://img.shields.io/badge/W&B-Offline-FFBE00?style=plastic&logo=wandb&logoColor=FFFFFF&label=Wandb&color=FFBE00)](https://wandb.ai)

`seg-dev` branch for semantic segmentation tasks, built on top of
[segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch).

### Supported Functions

- [x] Semantic segmentation with 12 architectures (Unet, FPN, UPerNet, DeepLabV3+, Segformer, ...).
- [x] Multiple encoders / backbones from the smp model zoo.
- [x] Multiple losses: Dice, Jaccard, Focal, CrossEntropy, BCE.
- [x] YAML driven configuration with command line overrides.
- [x] Visualize training curve (wandb / tensorboard).
- [x] Mixed-Precision Training for faster speed.
- [x] Per-class IoU / mIoU evaluation.

### Major Motivation

1.👀 Use the State-of-the-Art segmentation toolkit.
 + Various backbones and pretrained models
 + Powerful decoders and loss functions
 + High efficiency and extensibility

2.🚀 Enhance codes' reusability.

3.🛠️ Minimize our project.

## Quick Start

### Install Environment

The project runs on Python 3.10 with `conda`:

```bash
conda activate py310
pip install -r docs/requirements.txt
```

For detailed installation guides, please refer to [INSTALL.md](docs/INSTALL.md).

### Prepare Dataset

The segmentation dataset follows the **yolo** convention: a txt list holds one
image path per line, and the mask path is derived by replacing the `images`
folder of the image path by `labels`.

```bash
<pytorch-toolkit-dev> ~$ ln -s /data/exp-data/seg-dataset dataset/
<pytorch-toolkit-dev> ~$ python tools/create_lists.py --root dataset/seg-dataset
```

```bash
dataset/seg-dataset
├── images      # 0001.jpg, 0002.jpg ...
├── labels      # 0001.png, 0002.png ...
├── train.txt   # one image path per line
├── val.txt
└── test.txt
```

### Command Guides

+ Train & Validate with Training Curve

Refer to [README-Train/Val Models](tools/README.md) for details.

```bash
<pytorch-toolkit-dev> ~$ # train with the template config
<pytorch-toolkit-dev> ~$ python tools/train.py --config dataset/seg-template.yaml
<pytorch-toolkit-dev> ~$ # the checkpoints are in output/ckpts/<dataset name>/<experiment>
<pytorch-toolkit-dev> ~$ # evaluate the trained model
<pytorch-toolkit-dev> ~$ python tools/evaluate.py --config dataset/seg-template.yaml \
        --checkpoint output/ckpts/<dataset name>/<exp>/best.pth
```

## Project Layout

```bash
<pytorch-toolkit-dev>
├── dataset
│   ├── seg-dataset       # your data (symlink) + train.txt / val.txt
│   └── seg-template.yaml # the config template
├── docs
├── local_lib
│   ├── data              # dataset, augmentation and dataloader factories
│   ├── models            # smp model / loss / optimizer factories, checkpoints
│   └── utils             # yaml config loader, wandb / tensorboard logger,
│                         # metrics, visualisation of the predictions
├── output
│   └── ckpts             # the produced checkpoints (<dataset name>/<experiment>)
├── tools
│   ├── create_lists.py   # build the yolo-style txt lists
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py      # write the predictions as pictures
│   └── README.md
└── README.md
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Contribute

Feel free to create a pull request if you want to contribute (e.g. networks or tricks).