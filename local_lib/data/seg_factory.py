######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: seg_factory.py
# function: build segmentation datasets / dataloaders from config.
######################################################
import os
import torch
from torch.utils.data import DataLoader

from .seg_dataset import SegDataset
from .seg_augmentation import build_augmentation


def get_dataset_name(cfg):
    """Return the name of the dataset of the config.

    The name is the name of the folder of the dataset, so the runs of the
    different datasets do not get mixed up (e.g. 'MegerData').  An explicit
    'data.name' in the config wins; when it is missing the name is taken from
    the root of the dataset (or from the list file).

    Returns an empty string when the dataset cannot be identified.
    """
    data_cfg = cfg.get("data", {}) or {}

    explicit = (data_cfg.get("name") or "").strip()
    if explicit:
        return explicit

    root = data_cfg.get("root") or data_cfg.get("dir")
    if root:
        # the name of the folder of the dataset
        root = os.path.expanduser(str(root)).rstrip("/")
        name = os.path.basename(root)
        if name:
            return name

    # fall back to the folder of the list file
    for key in ("train_list", "val_list"):
        list_file = data_cfg.get(key)
        if not list_file:
            continue
        list_file = os.path.expanduser(str(list_file)).rstrip("/")
        name = os.path.basename(os.path.dirname(list_file))
        if name:
            return name

    return ""


def _resolve_list_file(cfg, split, data_cfg):
    """Find the txt list of a split.

    The list is looked up in this order:
      1. ``data.<split>_list``      (explicit path)
      2. ``data.train_list`` / ``data.val_list`` (shared by both splits)
      3. ``<data.root>/<split>.txt`` (yolo convention)
    """
    explicit = data_cfg.get(f"{split}_list")
    if explicit:
        return os.path.expanduser(explicit)

    shared = data_cfg.get("train_list") or data_cfg.get("val_list")
    if shared:
        return os.path.expanduser(shared)

    root = data_cfg.get("root") or data_cfg.get("dir")
    if root:
        candidate = os.path.join(os.path.expanduser(root), f"{split}.txt")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"no txt list found for the split '{split}'. Set 'data.{split}_list' or "
        f"provide '<data.root>/{split}.txt'."
    )


def _resolve_split_dir(root, split, sub_dir):
    """Resolve the image/mask directories of a split (folder based layout)."""
    root = os.path.expanduser(str(root))
    if sub_dir:
        return os.path.join(root, split, sub_dir)

    # root/<split>/images
    candidate = os.path.join(root, split, "images")
    if os.path.isdir(candidate):
        return candidate

    # root/images  (single-split layout)
    candidate = os.path.join(root, "images")
    if os.path.isdir(candidate):
        return candidate

    # root itself already holds the images
    if os.path.isdir(root):
        return root

    raise FileNotFoundError(f"cannot resolve split '{split}' under {root}")


def create_seg_dataset(cfg, split="train", is_training=False):
    """Create a :class:`SegDataset` for the given split.

    The data list is resolved in this order:
      1. ``data.<split>_list``  -> a yolo-style txt list (one image path per
         line); the mask path is derived by replacing ``images`` by ``labels``.
      2. ``data.<split>_images_dir`` / ``data.<split>_masks_dir`` -> a folder
         based layout.
      3. ``data.root`` -> ``<root>/<split>/images`` and ``<root>/<split>/masks``.
    """
    data_cfg = cfg.get("data", {}) or {}
    classes = cfg.get("classes", []) or []
    mask_ext = data_cfg.get("mask_ext", ".png")
    ignore_index = cfg.get("ignore_index", 255)
    ignore_background = bool(data_cfg.get("ignore_background", False))
    img_size = cfg.get("img_size", 320)
    images_dir_name = data_cfg.get("images_dir_name", "images")
    labels_dir_name = data_cfg.get("labels_dir_name", "labels")
    allow_missing_mask = bool(data_cfg.get("allow_missing_mask", False))

    augmentation = build_augmentation(
        "train" if is_training else "val",
        img_size,
        **{
            "hflip": data_cfg.get("hflip", 0.5),
            "vflip": data_cfg.get("vflip", 0.0),
            "affine_p": data_cfg.get("affine_p", 0.5),
            "color_jitter": data_cfg.get("color_jitter", 0.2),
            "gauss_noise": data_cfg.get("gauss_noise", 0.2),
        },
    )

    images_dir = data_cfg.get(f"{split}_images_dir")
    masks_dir = data_cfg.get(f"{split}_masks_dir")
    list_file = data_cfg.get(f"{split}_list")

    if list_file is None and (images_dir is None or masks_dir is None):
        # fall back to <root>/<split>.txt
        try:
            list_file = _resolve_list_file(cfg, split, data_cfg)
        except FileNotFoundError:
            list_file = None

    if list_file:
        return SegDataset(
            list_file=list_file,
            classes=classes,
            mask_ext=mask_ext,
            images_dir_name=images_dir_name,
            labels_dir_name=labels_dir_name,
            ignore_index=ignore_index,
            ignore_background=ignore_background,
            allow_missing_mask=allow_missing_mask,
            augmentation=augmentation,
        )

    if images_dir is None or masks_dir is None:
        root = data_cfg.get("root") or data_cfg.get("dir")
        if root is None:
            raise ValueError(
                "data.root (or data.<split>_list / data.<split>_images_dir) must be set in the config"
            )
        images_dir = _resolve_split_dir(root, split, data_cfg.get("images_subdir", "images"))
        masks_dir = _resolve_split_dir(root, split, data_cfg.get("masks_subdir", "masks"))

    return SegDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        classes=classes,
        mask_ext=mask_ext,
        ignore_index=ignore_index,
        ignore_background=ignore_background,
        allow_missing_mask=allow_missing_mask,
        augmentation=augmentation,
    )


def create_seg_loader(dataset, batch_size=8, num_workers=4, shuffle=None, seed=42, drop_last=False):
    """Wrap a dataset into a ``DataLoader``."""
    if shuffle is None:
        shuffle = True
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def create_seg_loaders(cfg):
    """Create the ``(train_loader, val_loader)`` pair described by the config."""
    data_cfg = cfg.get("data", {}) or {}
    batch_size = cfg.get("batch_size", 8)
    num_workers = cfg.get("workers", 4)
    seed = cfg.get("seed", 42)

    train_split = data_cfg.get("train_split", "train")
    val_split = data_cfg.get("val_split", "val")

    train_set = create_seg_dataset(cfg, split=train_split, is_training=True)
    val_set = create_seg_dataset(cfg, split=val_split, is_training=False)

    train_loader = create_seg_loader(
        train_set, batch_size, num_workers, shuffle=True, seed=seed, drop_last=False
    )
    val_loader = create_seg_loader(
        val_set, batch_size, num_workers, shuffle=False, seed=seed, drop_last=False
    )
    return train_loader, val_loader
