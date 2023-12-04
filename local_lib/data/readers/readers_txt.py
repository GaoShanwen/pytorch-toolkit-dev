######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: readers_txt.py
# function: create a reader for owner data.(load by txt)
######################################################
import os
from typing import Dict, Optional, List

from timm.utils.misc import natural_key
from timm.data.readers.class_map import load_class_map
from timm.data.readers.img_extensions import get_img_extensions
from timm.data.readers.reader import Reader
from PIL import Image


def read_images_and_targets(
    anno_path: str,
    # types: Optional[Union[List, Tuple, Set]] = None,
    class_to_idx: Optional[Dict] = None,
    # leaf_name_only: bool = True,
    sort: bool = True,
    **kwargs,
):
    """Walk folder recursively to discover images and map them to classes by folder names.

    Args:
        folder: root of folder to recrusively search
        types: types (file extensions) to search for in path
        class_to_idx: specify mapping for class (folder name) to class index if set
        leaf_name_only: use only leaf-name of folder walk for class names
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

    if kwargs.get("pass_path", None):
        pass_path = kwargs["pass_path"]
        with open(pass_path, "r") as f:
            for line in f.readlines():
                save_cats.remove(line.strip("\n"))

    if kwargs.get("num_classes", None):
        num_classes = kwargs["num_classes"]
        save_cats = save_cats[:num_classes] if save_cats is not None and num_classes < len(save_cats) else save_cats
    choose_cats = save_cats
    if kwargs.get("num_choose", None):
        num_choose = kwargs["num_choose"]
        choose_cats = save_cats[num_choose[0] : num_choose[1]]
    with open(anno_path, "r") as f:
        lines = [line.strip().split(", ") for line in f.readlines() if line.startswith("/data/AI-scales/images")]
    filenames, labels = zip(
        *[(filename, label) for filename, label in lines if choose_cats is None or label in choose_cats]
    )

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
        assert split in [
            "train",
            "val",
        ], f"split must be train/val, but you set is {split}"
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
    def __init__(
        self,
        filenames: List,
        # types: Optional[Union[List, Tuple, Set]] = None,
        class_to_idx: Optional[Dict] = None,
        # leaf_name_only: bool = True,
        sort: bool = True,
    ):
        super().__init__()
        assert len(filenames), (
            f"Found 0 images in subfolders of filenames. "
            f'Supported image extensions are {", ".join(get_img_extensions())}'
        )
        images_and_targets = list(
            zip(
                *(
                    filenames,
                    [
                        None,
                    ]
                    * len(filenames),
                )
            )
        )
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
    anno_path = "./dataset/blacklist/train.txt"
    # images_and_targets, class_to_idx = read_images_and_targets(
    #     anno_path, cats_path="dataset/blacklist/save_cats.txt",
    #     num_classes=632)

    with open(anno_path, "r") as f:
        query_files = [line.strip("\n").split(", ")[0] for line in f.readlines()]
    reader = ReaderImagePaths(query_files)
    import pdb

    pdb.set_trace()
