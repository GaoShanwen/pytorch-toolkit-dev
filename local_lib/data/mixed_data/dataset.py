import logging
import random
from copy import deepcopy

import numpy as np
import torch

from rfdetr.utilities.logger import get_logger

logger = get_logger(__name__)


def remap_target_labels(target: dict, mapping: dict[int, int]) -> dict:
    """Return a deep-copied target with class IDs remapped per ``mapping``.

    Behaviour:
    - If mapping maps an old id to new id, only the label value is changed; other
      labels are preserved.
    - Boxes/area/iscrowd/masks are preserved and kept in the original order.
    - If the mapping causes duplicate label ids (two boxes mapped to the same
      class), they are retained as separate boxes with the same label.
    """
    if not mapping:
        return target
    target = deepcopy(target)
    labels = target.get("labels")
    if labels is None or labels.numel() == 0:
        return target

    labels_np = labels.detach().cpu().numpy().astype(np.int64, copy=True)
    remapped = labels_np.copy()
    for old, new in mapping.items():
        remapped[remapped == int(old)] = int(new)

    target["labels"] = torch.from_numpy(remapped).to(device=labels.device, dtype=labels.dtype)

    for key in ("boxes", "area", "iscrowd", "masks"):
        v = target.get(key)
        if v is None:
            continue
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == labels_np.shape[0]:
            target[key] = v
    return target


class MixedRFDETRDataset(torch.utils.data.Dataset):
    """Wraps an RF-DETR YoloDetection backend with class remapping and
    per-epoch background-image mixing."""

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        class_mapping: dict[int, int] | None = None,
        background_im_files: list | None = None,
        alpha: float | None = None,
    ):
        super().__init__()
        self.base = base_dataset
        self.class_mapping: dict[int, int] = class_mapping or {}
        self.background_im_files: list = list(background_im_files or [])
        self.alpha = alpha

        self._main_ids = list(range(len(base_dataset)))
        self._bg_ids: list = []
        self._n_main = len(self._main_ids)
        self._n_bg = 0
        self.class_counts: dict[int, int] = {}

        if self.alpha and self.alpha > 0 and self.background_im_files:
            self.resample_background()
        self.class_counts = self._count_instance_labels()
        self._log_dataset_summary()

    def _count_instance_labels(self) -> dict[int, int]:
        """Count instances per class on the wrapped main dataset after remapping."""
        counts: dict[int, int] = {}
        try:
            total = len(self.base)
        except Exception:
            total = 0

        for idx in range(total):
            try:
                sample = self.base[idx]
            except Exception:
                continue
            if not isinstance(sample, (tuple, list)) or len(sample) < 2:
                continue
            target = sample[1]
            if target is None:
                continue
            if isinstance(target, dict):
                labels = target.get("labels")
            else:
                labels = getattr(target, "labels", None)
            if labels is None:
                continue
            if self.class_mapping:
                target = remap_target_labels(target, self.class_mapping)
                labels = target.get("labels") if isinstance(target, dict) else getattr(target, "labels", None)
            try:
                labels_arr = labels.detach().cpu().tolist() if torch.is_tensor(labels) else list(labels)
            except Exception:
                labels_arr = np.asarray(labels).reshape(-1).tolist()
            for label in labels_arr:
                if label is None:
                    continue
                key = int(label)
                counts[key] = counts.get(key, 0) + 1

        return dict(sorted(counts.items()))

    def _log_dataset_summary(self) -> None:
        """Emit a compact summary proving the mixed dataset changed the class distribution."""
        main_total = sum(self.class_counts.values())
        logger.info(
            "MixedRFDETR final dataset summary: main_samples=%d, background_slots=%d, total_instances=%d, per_class_instances=%s",
            self._n_main,
            self._n_bg,
            main_total,
            self.class_counts,
        )
        if self.class_mapping:
            logger.info(
                "MixedRFDETR class mapping active: %s; remapped final class counts=%s",
                self.class_mapping,
                self.class_counts,
            )

    def resample_background(self) -> None:
        if not self.alpha or self.alpha <= 0 or not self.background_im_files:
            self._bg_ids = []
            self._n_bg = 0
            return
        n_bg = max(1, int(self._n_main * self.alpha)) if self._n_main > 0 else 0
        if n_bg == 0:
            self._bg_ids = []
            self._n_bg = 0
            return
        pool = self.background_im_files
        self._bg_ids = [random.randrange(len(pool)) for _ in range(n_bg)]
        self._n_bg = n_bg
        logger.info("MixedRFDETR: %d main + %d bg (alpha=%.2f)", self._n_main, self._n_bg, self.alpha)

    def __len__(self) -> int:
        return self._n_main + self._n_bg

    def __getitem__(self, idx: int):
        if idx < self._n_main:
            image, target = self.base[idx]
            if self.class_mapping and target is not None:
                target = remap_target_labels(target, self.class_mapping)
            return image, target

        try:
            sample = self.base[0][0]
            if hasattr(sample, "shape") and len(sample.shape) == 3:
                c, h, w = int(sample.shape[0]), int(sample.shape[1]), int(sample.shape[2])
            else:
                c, h, w = 3, 512, 512
        except Exception:
            c, h, w = 3, 512, 512

        image = torch.zeros((c, h, w), dtype=torch.float32)
        bg_idx = self._bg_ids[idx - self._n_main]
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([-1]),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
            "size": torch.tensor([h, w], dtype=torch.int64),
            "is_background": True,
            "bg_path": self.background_im_files[bg_idx],
        }
        return image, target