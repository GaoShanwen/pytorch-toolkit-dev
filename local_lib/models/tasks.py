from ultralytics.nn.tasks import PoseModel

from .symmetry_match.tasks import SymmetryMatchPoseModel


class CustomPoseModel(SymmetryMatchPoseModel):
    """Combined PoseModel supporting symmetry matching for pose estimation.

    MRO: CustomPoseModel -> SymmetryMatchPoseModel -> PoseModel
    """

    def __init__(self, *args, **kwargs):
        self.symmetry_categories = kwargs.pop("symmetry_categories", None)
        self.symmetry_pairs = kwargs.pop("symmetry_pairs", None)
        assert self.symmetry_categories is not None and self.symmetry_pairs is not None, "Both symmetry_categories and symmetry_pairs must be provided."
        PoseModel.__init__(self, *args, **kwargs)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return SymmetryMatchPoseModel.init_criterion(self)