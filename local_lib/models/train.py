from copy import copy

from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .tasks import CustomPoseModel


class CustomPoseTrainer(PoseTrainer):
    """Trainer for custom pose estimation."""

    def __init__(self, *args, **kwargs):
        overrides = kwargs.get("overrides", {})
        self.symmetry_categories = overrides.pop("symmetry_categories", None)
        self.symmetry_pairs = overrides.pop("symmetry_pairs", None)
        assert self.symmetry_categories is not None, "symmetry_categories must be provided"
        assert self.symmetry_pairs is not None, "symmetry_pairs must be provided"
        PoseTrainer.__init__(self, *args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CustomPoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
            symmetry_categories=self.symmetry_categories,
            symmetry_pairs=self.symmetry_pairs,
        )
        if weights:
            model.load(weights)
        return model