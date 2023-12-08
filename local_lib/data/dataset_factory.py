######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.12.08
# filenaem: dataset_factory.py
# function: create custom dataset.
######################################################
from .dataset import TxtReaderImageDataset
from .readers import ReaderImagePaths
from timm.data import create_dataset


def create_custom_dataset(name, root, split="val", class_map=None, is_training=False, **kwargs):
    if name != "txt_data":
        return create_dataset(name, root, split, class_map, is_training, **kwargs)
    split = "train" if is_training else split.replace("validation", "val")
    assert split in ["train", "val", "infer"], f"split must be train/val or infer but you set {split}"
    reader = ReaderImagePaths(root, sort=False) if split == "infer" else None
    ds = TxtReaderImageDataset(root, reader=reader, split=split, class_map=class_map, **kwargs)
    return ds
