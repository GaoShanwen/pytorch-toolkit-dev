from setuptools import setup, find_packages
import local_lib

requirements = [
    "numpy",
    "tqdm",
    "timm",
    "pandas",
    "requests",
    "tensorboard",
    "setuptools",
]

setup(
    name="local_lib",
    version=local_lib.__version__,
    python_requires="==3.8",
    author="GaoWenJie",
    author_email="gaowenjie@rongxwy.com",
    url="https://github.com/GaoShanwen/pytorch-toolkit-dev/tree/timm-dev",
    description="Build a multi-task training platform based on Pytorch",
    license="Apache-2.0",
    packages=find_packages(exclude=("cfgs", "dataset", "docs", "results")),
    install_requires=requirements,
)
