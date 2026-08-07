from copy import copy

from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .symmetry_match.train import SymmetryMatchPoseTrainer
from .quad_eiou.train import QuadPoseTrainer
from .mixed_data.train import MixedDataTrainer
from .tasks import CustomPoseModel
from .val import CustomPoseValidator


class CustomPoseTrainer(SymmetryMatchPoseTrainer, QuadPoseTrainer, MixedDataTrainer):
    """Combined trainer supporting mixed_data, symmetry_match, and quad_eiou features.

    MRO: CustomPoseTrainer -> SymmetryMatchPoseTrainer -> QuadPoseTrainer -> MixedDataTrainer
    - __init__: cooperative multiple inheritance extracts each feature's parameters from overrides
    - build_dataset: from MixedDataTrainer (mixed dataset)
    - get_model: overridden to create ComboPoseModel
    - get_validator: overridden to return CustomPoseValidator
    - _model_train: overridden to fix None check in mixed_data rotation
    """
    def __init__(self, *args, **kwargs):
        overrides = kwargs.get("overrides", {})
        self.quad_categories = overrides.pop("quad_categories", None)
        self.quad_indices = overrides.pop("quad_indices", None)
        if self.quad_categories is None and self.quad_indices is None:
            print("====== without eiou for CustomPoseTrainer ======")

        self.symmetry_categories = overrides.pop("symmetry_categories", None)
        self.symmetry_pairs = overrides.pop("symmetry_pairs", None)
        assert self.symmetry_categories is not None, "symmetry_categories must be provided"
        assert self.symmetry_pairs is not None, "symmetry_pairs must be provided"

        self.class_mapping = overrides.pop("class_mapping", None)
        self.mixed_alpha = overrides.pop("mixed_alpha", None)
        assert self.mixed_alpha is not None, "mixed_alpha must be provided"
        assert self.class_mapping is not None, "class_mapping must be provided"

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
            quad_categories=self.quad_categories,
            quad_indices=self.quad_indices,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        model = unwrap_model(self.model)
        if hasattr(model, "student_model"):
            model = model.student_model
        if getattr(model.model[-1], "flow_model", None) is not None:
            self.loss_names += ("rle_loss",)
        if self.quad_categories is not None and self.quad_indices is not None:
            self.loss_names += ("eiou_loss",)
        return CustomPoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks,
            symmetry_categories=self.symmetry_categories, symmetry_pairs=self.symmetry_pairs,
        )
