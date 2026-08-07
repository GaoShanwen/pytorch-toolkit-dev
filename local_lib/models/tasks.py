from ultralytics.nn.tasks import PoseModel

from .symmetry_match.tasks import SymmetryMatchPoseModel
from .quad_eiou.tasks import QuadPoseModel
from .loss import CustomPoseLoss, E2ELoss, v8PoseLoss


class CustomPoseModel(SymmetryMatchPoseModel, QuadPoseModel):
    """Combined PoseModel supporting both symmetry matching and quad EIoU features.

    MRO: CustomPoseModel -> SymmetryMatchPoseModel -> QuadPoseModel -> PoseModel
    - init_criterion: from SymmetryMatchPoseModel (first in MRO)
    - quad_categories/quad_indices: from QuadPoseModel
    - symmetry_categories/symmetry_pairs: from SymmetryMatchPoseModel
    """

    def __init__(self, *args, **kwargs):
        self.symmetry_categories = kwargs.pop("symmetry_categories", None)
        self.symmetry_pairs = kwargs.pop("symmetry_pairs", None)
        assert self.symmetry_categories is not None and self.symmetry_pairs is not None, "Both symmetry_categories and symmetry_pairs must be provided."
        self.quad_categories = kwargs.pop("quad_categories", None)
        self.quad_indices = kwargs.pop("quad_indices", None)
        if self.quad_categories is None and self.quad_indices is None:
            print("====== without eiou for CustomPoseModel ======")
        PoseModel.__init__(self, *args, **kwargs)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        if self.quad_categories is None and self.quad_indices is None:
            return SymmetryMatchPoseModel.init_criterion(self)
        return E2ELoss(self, CustomPoseLoss) if getattr(self, "end2end", False) else v8PoseLoss(self)