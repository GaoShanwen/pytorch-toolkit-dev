from ultralytics.nn.tasks import PoseModel

from .loss import E2ELoss, QuadPoseLoss, v8PoseLoss


class QuadPoseModel(PoseModel):
    def __init__(self, *args, **kwargs):
        self.quad_categories = kwargs.pop("quad_categories", None)
        self.quad_indices = kwargs.pop("quad_indices", None)
        assert self.quad_categories is not None, "quad_categories must be provided"
        assert self.quad_indices is not None, "quad_indices must be provided"
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        """Initialize the loss criterion for the QuadEIOU PoseModel."""
        return E2ELoss(self, QuadPoseLoss) if getattr(self, "end2end", False) else v8PoseLoss(self)