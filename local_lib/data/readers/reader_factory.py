######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.12.08
# filenaem: reader_factory.py
# function: create custom reader.
######################################################
import os
from . import ReaderImageTxt


def create_reader(root, split="train", **kwargs):
    assert os.path.exists(root)
    reader = ReaderImageTxt(root, split=split, **kwargs)
    return reader
