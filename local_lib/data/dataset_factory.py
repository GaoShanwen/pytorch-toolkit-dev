######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.12.08
# filenaem: dataset_factory.py
# function: create custom dataset.
######################################################
from timm.data import create_dataset

from .dataset import TxtReaderImageDataset, MultiLabelDataset
from .readers import ReaderImagePaths


def create_custom_dataset(name, root, split="val", class_map=None, is_training=False, **kwargs):
    if name not in ["txt_data", "multilabel"]:
        return create_dataset(name, root, split, class_map, is_training, **kwargs)
    split = "train" if is_training else split.replace("validation", "val")
    assert split in ["train", "val", "infer"], f"split must be train/val or infer, but you set {split}"
    if name == "multilabel":
        multilabel = kwargs.get("multilabel", None)
        assert isinstance(multilabel, dict), "please set multilabel for args!"
        return MultiLabelDataset(root, split, multilabel.get("attributes", None), **kwargs)
    
    class_to_idx = kwargs.get("class_to_idx", None)
    reader = ReaderImagePaths(root, sort=False, class_to_idx=class_to_idx) if split == "infer" else None
    return TxtReaderImageDataset(root, reader=reader, split=split, class_map=class_map, **kwargs)

