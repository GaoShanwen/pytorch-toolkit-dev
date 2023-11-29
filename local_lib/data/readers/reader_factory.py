import os
from . import ReaderImageTxt

def create_reader(root, split='train', **kwargs):
    assert os.path.exists(root)
    reader = ReaderImageTxt(root, split=split, **kwargs)
    return reader
