import os
from .dataset import TxtReaderImageDataset

from timm.data import create_dataset


def create_owner_dataset(
        name,
        root,
        split='validation',
        search_split=True,
        class_map=None,
        load_bytes=False,
        is_training=False,
        download=False,
        batch_size=None,
        seed=42,
        repeats=0,
        **kwargs
):
    if name != "txt_data":
        return create_dataset(
            name, root, split, search_split, class_map, load_bytes, 
            is_training, download, batch_size, seed, repeats, **kwargs
        )
    split = "train" if is_training else "val"
    ds = TxtReaderImageDataset(root, split=split, class_map=class_map, **kwargs)
    return ds