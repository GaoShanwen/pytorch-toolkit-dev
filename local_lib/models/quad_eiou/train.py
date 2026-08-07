from pathlib import Path
from typing import Any
from copy import copy
import torch.distributed as dist

from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .tasks import QuadPoseModel


class QuadPoseTrainer(PoseTrainer):
    def __init__(self, *args, **kwargs):
        self.quad_categories = kwargs.get("overrides", {}).pop("quad_categories", None)
        self.quad_indices = kwargs.get("overrides", {}).pop("quad_indices", None)

        super().__init__(*args, **kwargs)

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> QuadPoseModel:
        """Get pose estimation model with QuadEIOU configuration and weights.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (QuadEIOUPoseModel): Initialized pose estimation model.
        """
        model = QuadPoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
            quad_categories=self.quad_categories,
            quad_indices=self.quad_indices,
        )
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of the QuadEIOUPoseValidator for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        model = unwrap_model(self.model)
        if hasattr(model, "student_model"):
            model = model.student_model
        if getattr(model.model[-1], "flow_model", None) is not None:
            self.loss_names += ("rle_loss",)
        self.loss_names += ("eiou_loss",)
        return PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks,
        )