from setuptools import find_packages, setup

import local_lib

requirements = [
    "numpy",
    "tqdm",
    "pyyaml",
    "opencv-python",
    "segmentation-models-pytorch",
    "albumentations",
    "torch",
    "torchvision",
    "wandb",
    "tensorboard",
    "setuptools",
]

setup(
    name="local_lib",
    version=local_lib.__version__,
    python_requires="==3.10",
    author="GaoWenJie",
    author_email="gaoshanwen@bupt.cn",
    url="https://github.com/GaoShanwen/pytorch-toolkit-dev/tree/seg-dev",
    description="Build a multi-task training platform based on Pytorch",
    license="Apache-2.0",
    packages=find_packages(exclude=("cfgs", "dataset", "docs", "local_lib")),
    install_requires=requirements,
)