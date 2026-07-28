from pathlib import Path
from typing import Any
from copy import copy
import torch.distributed as dist

from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .val import SymmetryMatchPoseValidator
from .tasks import SymmetryMatchPoseModel


class SymmetryMatchPoseTrainer(PoseTrainer):
    def __init__(self, *args, **kwargs):
        self.symmetry_categories = kwargs.get("overrides", {}).pop("symmetry_categories", None)
        self.symmetry_pairs = kwargs.get("overrides", {}).pop("symmetry_pairs", None)

        super().__init__(*args, **kwargs)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> SymmetryMatchPoseModel:
        """Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (PoseModel): Initialized pose estimation model.
        """
        model = SymmetryMatchPoseModel(
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

    def get_validator(self):
        """Return an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        model = unwrap_model(self.model)
        if hasattr(model, "student_model"):
            model = model.student_model  # copy_attr does not copy nn.Module attributes like .model
        if getattr(model.model[-1], "flow_model", None) is not None:
            self.loss_names += ("rle_loss",)
        return SymmetryMatchPoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks,
            symmetry_categories=self.symmetry_categories, symmetry_pairs=self.symmetry_pairs,
        )
