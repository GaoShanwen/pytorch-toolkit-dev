######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2023.12.08
# filenaem: reader_factory.py
# function: create custom reader.
######################################################
import os

from .reader_image_in_txt import ReaderImageTxt


def create_reader(root, split="train", **kwargs):
    assert os.path.exists(root), f"{root} not exist !!!"
    reader = ReaderImageTxt(root, split=split, **kwargs)
    return reader
