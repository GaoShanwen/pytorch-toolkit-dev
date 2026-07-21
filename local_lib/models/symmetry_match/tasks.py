from ultralytics.nn.tasks import PoseModel
from .loss import E2ELoss, SymmetryMatchPoseLoss, v8PoseLoss


class SymmetryMatchPoseModel(PoseModel):
    def __init__(self, *args, **kwargs):
        self.symmetry_categories = kwargs.pop("symmetry_categories", None)
        self.symmetry_pairs = kwargs.pop("symmetry_pairs", None)
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return E2ELoss(self, SymmetryMatchPoseLoss) if getattr(self, "end2end", False) else v8PoseLoss(self)
        
