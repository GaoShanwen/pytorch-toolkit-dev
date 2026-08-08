import yaml

from .symmetry_match.val import SymmetryMatchPoseValidator
from .mixed_data.val import MixedDataValidator
from .categorized_vis.val import CategorizedVisValidator


class CustomPoseValidator(SymmetryMatchPoseValidator, MixedDataValidator, CategorizedVisValidator):
    """Combined validator supporting symmetry matching, mixed data, and categorized visualization.

    MRO: CustomPoseValidator -> SymmetryMatchPoseValidator -> PoseValidator -> MixedDataValidator -> CategorizedVisValidator -> DetectionValidator
    - init_metrics: from CategorizedVisValidator (categorized FP/FN visualization)
    - build_dataset: from MixedDataValidator (mixed dataset support)
    - update_metrics: from SymmetryMatchPoseValidator (symmetry matching)
    """

    def __init__(self, *args, **kwargs):
        self.symmetry_categories = kwargs.pop("symmetry_categories", None)
        self.symmetry_pairs = kwargs.pop("symmetry_pairs", None)
        if self.symmetry_pairs is None:
            cfg_path = kwargs.get("args", {}).get("data", "dataset.yaml")
            with open(cfg_path, "r") as f:
                cfgs = yaml.safe_load(f)
            self.symmetry_pairs = cfgs["symmetry_pairs"]
            self.symmetry_categories = cfgs["symmetry_categories"]
        
        MixedDataValidator.__init__(self, *args, **kwargs)

    