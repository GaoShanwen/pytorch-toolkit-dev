from setuptools import find_packages, setup

import local_lib

requirements = [
    "numpy",
    "tqdm",
    "timm",
    "pandas",
    "argparse",
    "requests",
    "tensorboard",
    "setuptools",
]

setup(
    name="local_lib",
    version=local_lib.__version__,
    python_requires="==3.8",
    author="GaoWenJie",
    author_email="gaoshanwen@bupt.cn",
    url="https://github.com/GaoShanwen/pytorch-toolkit-dev/tree/timm-dev",
    description="Build a multi-task training platform based on Pytorch",
    license="Apache-2.0",
    packages=find_packages(exclude=("cfgs", "dataset", "docs", "local_lib")),
    install_requires=requirements,
)
