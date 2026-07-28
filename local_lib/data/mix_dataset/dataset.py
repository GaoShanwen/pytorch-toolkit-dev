import numpy as np
import os
import random
from pathlib import Path
from ultralytics.data.dataset import YOLODataset, DATASET_CACHE_VERSION
from ultralytics.utils import LOGGER, TQDM, LOCAL_RANK
from ultralytics.data.utils import HELP_URL, get_hash, img2label_paths, load_dataset_cache_file


class MixedDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        data = kwargs.get("data", {})
        self.background_data = data.get("background_data", None)
        self.alpha = data.get("mixed_alpha", None)
        if not kwargs.get("augment", False) and self.alpha is not None:
            self.alpha = 0.0
            self.background_data = None
        self.class_mapping = data.get("class_mapping", None)

        assert self.class_mapping is not None or os.path.exists(self.background_data), \
                f"background_data should exist or class_mapping must be provided!"
        self.background_im_files = []
        self.original_im_files = None
        
        if self.background_data:
            self._load_background_images()
        super().__init__(*args, **kwargs)
        
        if self.class_mapping is not None:
            self.labels = self._apply_class_mapping(self.labels)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        files = super().get_img_files(img_path)
        if self.background_data:
            num_background = int(len(files) * self.alpha)
            sampled_bgs = random.sample(self.background_im_files, min(num_background, len(self.background_im_files)))
            files.extend(sampled_bgs)
            LOGGER.info(f"Resampled dataset: {len(files)-num_background} main + {num_background} background images (alpha={self.alpha})")
        return files

    def _load_background_images(self):
        if isinstance(self.background_data, str):
            with open(self.background_data, 'r') as f:
                self.background_im_files = [line.strip() for line in f if line.strip()]
            LOGGER.info(f"Loaded {len(self.background_im_files)} background images from {self.background_data}")

    def _apply_class_mapping(self, labels):
        if self.class_mapping is None or not isinstance(self.class_mapping, dict):
            return labels
        
        for label in labels:
            cls = label["cls"]
            if len(cls) == 0:
                continue
            for old_cls, new_cls in self.class_mapping.items():
                cls[cls == old_cls] = new_cls
            label["cls"] = cls
        return labels

        
