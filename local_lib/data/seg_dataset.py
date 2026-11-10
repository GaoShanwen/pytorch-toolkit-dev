######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: seg_dataset.py
# function: segmentation dataset for smp (multi-class / binary).
#          the data list follows the yolo convention: a txt file with one
#          image path per line, the mask path is derived by replacing the
#          'images' folder by 'labels'.
######################################################
import os
import logging
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as BaseDataset


_logger = logging.getLogger("seg_dataset")

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def img2label_path(img_path, images_dir="images", labels_dir="labels", mask_ext=".png"):
    """Derive the mask path of an image, yolo style.

    The ``images`` part of the path is replaced by ``labels`` and the
    extension is replaced by ``mask_ext``.  e.g.
    ``/data/seg/images/a.jpg`` -> ``/data/seg/labels/a.png``.
    """
    path = str(img_path)
    if images_dir and images_dir in path:
        path = path.replace(images_dir, labels_dir)
    else:
        # no 'images' folder in the path: fall back to the sibling folder
        path = os.path.join(
            os.path.dirname(os.path.dirname(path)), labels_dir, os.path.basename(path)
        )
    return os.path.splitext(path)[0] + mask_ext


def read_image_list(list_file, mask_ext=".png", images_dir="images", labels_dir="labels",
                    mask_ext_override=None, allow_missing_mask=False):
    """Read a yolo-style txt list and build the ``(image, mask)`` pairs.

    The txt file holds one image path per line; empty lines are skipped.
    The mask path of every image is derived with :func:`img2label_path`.

    Args:
        list_file: path of the txt list (e.g. ``train.txt``).
        mask_ext: extension of the mask files.
        images_dir: folder name replaced by ``labels_dir`` in the path.
        labels_dir: folder name of the masks.
        mask_ext_override: unused placeholder kept for API symmetry.
        allow_missing_mask: when True, an image without a mask file is kept
            and its mask path is set to ``None`` (the mask is then treated as
            an all-zero label).

    Returns:
        list of ``(image_path, mask_path)``; ``mask_path`` is ``None`` when
        the mask is missing and ``allow_missing_mask`` is set.
    """
    if not os.path.exists(list_file):
        raise FileNotFoundError(f"image list not found: {list_file}")

    samples = []
    missing = 0
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            img_path = os.path.expanduser(line)
            if not os.path.isabs(img_path):
                # relative paths are resolved against the list file folder
                img_path = os.path.join(os.path.dirname(os.path.abspath(list_file)), img_path)
            mask_path = img2label_path(img_path, images_dir, labels_dir, mask_ext)
            if not os.path.exists(mask_path):
                missing += 1
                if not allow_missing_mask:
                    continue
                mask_path = None
            samples.append((img_path, mask_path))

    if not samples:
        raise RuntimeError(f"the image list is empty: {list_file}")
    if missing:
        _logger.warning(
            f"{missing} images of {list_file} have no mask file "
            f"({'kept as empty labels' if allow_missing_mask else 'skipped'})."
        )
    return samples


def list_images(images_dir, mask_ext=".png", masks_dir=None, allow_missing_mask=False):
    """Collect image / mask pairs by scanning a folder.

    Kept for the folder based layout; the txt list is the preferred way.
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    mask_root = masks_dir if masks_dir else images_dir
    samples = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(IMG_EXTENSIONS):
            continue
        stem = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_root, f"{stem}{mask_ext}")
        if not os.path.exists(mask_path):
            if not allow_missing_mask:
                continue
            mask_path = None
        samples.append((os.path.join(images_dir, fname), mask_path))
    if not samples:
        raise RuntimeError(
            f"no paired image/mask found: images={images_dir} masks={mask_root} (mask ext: {mask_ext})"
        )
    return samples


class SegDataset(BaseDataset):
    """Paired image / mask dataset for semantic segmentation.

    Two ways of describing the data are supported:

    * **txt list** (recommended, yolo style): pass ``list_file``, a txt file
      holding one image path per line.  The mask path is derived by
      replacing the ``images`` folder by ``labels``.
    * **folders**: pass ``images_dir`` / ``masks_dir`` to scan a folder.

    Args:
        list_file: path of the yolo-style txt list.
        images_dir: directory holding the source images (folder mode).
        masks_dir: directory holding the ground-truth masks (folder mode).
        classes: list of class names, index 0 is the background by default.
        mask_ext: extension of the mask files.
        images_dir_name: folder name replaced by ``labels_dir_name`` in the
            image path when deriving the mask path.
        labels_dir_name: folder name of the masks.
        ignore_index: label id that is excluded from loss and metrics.
        ignore_background: if True, pixel value 0 of the raw mask is remapped
            to ``ignore_index`` (useful for datasets without a background class).
        allow_missing_mask: if True, an image without a mask file is kept and
            its label is treated as an all-zero mask (i.e. "no hair").
        augmentation: an ``albumentations.Compose`` instance applied to both
            the image and the mask.
        input_size: ``(H, W)`` used to resize the sample when given.
    """

    def __init__(
        self,
        list_file=None,
        images_dir=None,
        masks_dir=None,
        classes=None,
        mask_ext=".png",
        images_dir_name="images",
        labels_dir_name="labels",
        ignore_index=255,
        ignore_background=False,
        allow_missing_mask=False,
        augmentation=None,
        input_size=None,
    ):
        self.classes = list(classes) if classes else []
        self.num_classes = len(self.classes)
        self.mask_ext = mask_ext
        self.images_dir_name = images_dir_name
        self.labels_dir_name = labels_dir_name
        self.ignore_index = ignore_index
        self.ignore_background = ignore_background
        self.allow_missing_mask = allow_missing_mask
        self.augmentation = augmentation
        self.input_size = input_size

        if list_file:
            self.samples = read_image_list(
                list_file,
                mask_ext=mask_ext,
                images_dir=images_dir_name,
                labels_dir=labels_dir_name,
                allow_missing_mask=allow_missing_mask,
            )
        elif images_dir:
            self.samples = list_images(
                images_dir, mask_ext, masks_dir, allow_missing_mask=allow_missing_mask
            )
        else:
            raise ValueError("either 'list_file' or 'images_dir' must be given.")

        self.images_fps = [s[0] for s in self.samples]
        self.masks_fps = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        image_path, mask_path = self.samples[i]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path is None:
            # no annotation for this image -> an all-zero (background) label
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # single channel!
            if mask is None:
                raise RuntimeError(f"failed to read mask: {mask_path}")

        if self.ignore_background and self.num_classes > 0:
            # raw pixel 0 is "unlabelled" -> exclude it from learning
            mask = np.where(mask == 0, self.ignore_index, mask)

        # the class ids of the mask must be lower than the number of classes
        if self.num_classes > 0:
            max_id = int(mask.max()) if mask.size else 0
            if max_id >= self.num_classes:
                raise ValueError(
                    f"the mask '{mask_path}' holds the class id {max_id} but only "
                    f"{self.num_classes} classes are configured ({self.classes}). "
                    "The 'classes' of the config must list every class of the masks, "
                    "the background included (e.g. ['background', 'hair'])."
                )

        if self.augmentation is not None:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.input_size is not None:
            image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()  # class indices
        return image, mask
