from ultralytics.models.yolo.pose import PoseValidator

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
        MixedDataValidator.__init__(self, *args, **kwargs)

    