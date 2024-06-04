######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: readers_txt.py
# function: create a reader for custom data.(load by txt)
######################################################
import os
from typing import Dict, List, Optional

from timm.data.readers.class_map import load_class_map
from timm.data.readers.img_extensions import get_img_extensions
from timm.data.readers.reader import Reader
from timm.utils.misc import natural_key


def check_img(filename):
    if not os.path.exists(filename):
        return False
    try:
        with open(filename, "rb") as f:
            f.seek(-2, 2)
            if not f.read() == b"\xff\xd9":
                return False
    except IOError:
        return False
    return True


def read_images_and_targets(anno_path: str, class_to_idx: Optional[Dict] = None, sort: bool = True, **kwargs):
    """Walk folder recursively to discover images and map them to classes by folder names.

    Args:
        anno_path: txtfile of annotation to search for in path
        class_to_idx: specify mapping for class (folder name) to class index if set
        sort: re-sort found images by name (for consistent ordering)

    Returns:
        A list of image and target tuples, class_to_idx mapping
    """
    # types = get_img_extensions(as_set=True) if not types else set(types)
    save_cats = None
    if kwargs.get("cats_path", None):
        cats_path = kwargs["cats_path"]
        with open(cats_path, "r") as f:
            save_cats = [line.strip("\n") for line in f.readlines()]

    if kwargs.get("num_classes", None):
        num_classes = kwargs["num_classes"]
        save_cats = save_cats[:num_classes] if save_cats is not None and num_classes < len(save_cats) else save_cats
    with open(anno_path, "r") as f:
        lines = [line.strip().replace(' ', '').split(",") for line in f.readlines()]
    filenames, labels = zip(
        *[(fpath, label) for fpath, label in lines if save_cats is None or label in save_cats]
    )
    # check images 
    for img_path in filenames:
        assert check_img(img_path), f"{img_path} in dataset is unreadable!"

    if class_to_idx is None:
        # building class index
        unique_labels = set(labels) if save_cats is None else save_cats
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


class ReaderImageTxt(Reader):
    def __init__(self, root, split, class_map="", **kwargs):
        super().__init__()

        self.root = root
        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        assert split in ["train", "val"], f"split must be train/val, but you set is {split}"
        anno_path = os.path.join(root, "train.txt") if split == "train" else os.path.join(root, "val.txt")
        self.samples, self.class_to_idx = read_images_and_targets(anno_path, class_to_idx=class_to_idx, **kwargs)
        if len(self.samples) == 0:
            raise RuntimeError(
                f"Found 0 images in subfolders of {root}. "
                f'Supported image extensions are {", ".join(get_img_extensions())}'
            )

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, "rb"), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename


class ReaderImagePaths(Reader):
    def __init__(self, images_and_targets: List, class_to_idx: Optional[Dict] = None, sort: bool = True):
        super().__init__()
        filepaths, _ = zip(*(images_and_targets))
        assert len(filepaths), (
            f"Found 0 images in subfolders of filenames. "
            f'Supported image extensions are {", ".join(get_img_extensions())}'
        )
        if class_to_idx is not None:
            images_and_targets = [(f, class_to_idx[l]) for f, l in images_and_targets]
        if sort:
            images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
        self.samples, self.class_to_idx = images_and_targets, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, "rb"), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename


if __name__ == "__main__":
    anno_path = "./dataset/optimize_task3/train.txt"
    cats_path="dataset/optimize_task3/2066_cats.txt"
    images_and_targets, class_to_idx = read_images_and_targets(
        anno_path, cats_path=cats_path, num_classes=2066
    )
    print(f"Loaded trainset: cats={len(class_to_idx)}, imgs={len(images_and_targets)}")

    # with open(anno_path, "r") as f:
    #     query_files = [line.strip("\n").split(", ")[0] for line in f.readlines()]
    # reader = ReaderImagePaths(query_files)
