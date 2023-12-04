import os
from .dataset import TxtReaderImageDataset
from .readers import ReaderImagePaths
from timm.data import create_dataset


def create_owner_dataset(
    name,
    root,
    split="val",
    search_split=True,
    class_map=None,
    load_bytes=False,
    is_training=False,
    download=False,
    batch_size=None,
    seed=42,
    repeats=0,
    **kwargs,
):
    if name != "txt_data":
        return create_dataset(
            name,
            root,
            split,
            search_split,
            class_map,
            load_bytes,
            is_training,
            download,
            batch_size,
            seed,
            repeats,
            **kwargs,
        )
    # split = "train" if is_training else "val"
    assert split in [
        "train",
        "val",
        "infer",
    ], f"split must be train/val or infer but you set {split}"
    if split == "infer":
        reader = ReaderImagePaths(root, sort=False)
    ds = TxtReaderImageDataset(
        root, reader=reader, split=split, class_map=class_map, **kwargs
    )
    return ds
