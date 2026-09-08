import yaml

from ultralytics.models.yolo.pose import PoseValidator


class CustomPoseValidator(PoseValidator):
    """Validator for custom pose estimation."""

    def __init__(self, *args, **kwargs):
        self.symmetry_categories = kwargs.pop("symmetry_categories", None)
        self.symmetry_pairs = kwargs.pop("symmetry_pairs", None)
        if self.symmetry_pairs is None:
            cfg_path = kwargs.get("args", {}).get("data", "dataset.yaml")
            with open(cfg_path, "r") as f:
                cfgs = yaml.safe_load(f)
            self.symmetry_pairs = cfgs["symmetry_pairs"]
            self.symmetry_categories = cfgs["symmetry_categories"]
        super().__init__(*args, **kwargs)