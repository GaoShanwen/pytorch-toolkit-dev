from copy import copy

from ultralytics.models.yolo.detect import DetectionValidator

from ...data import build_mixed_dataset


class MixedDataValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build YOLO Dataset for validation with class mapping support."""
        return build_mixed_dataset(self.args, img_path, batch, self.data, mode=mode, rect=True, stride=self.stride)
