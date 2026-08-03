from copy import copy
import random
import yaml

from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .symmetry_match.tasks import SymmetryMatchPoseModel
from .val import CustomPoseValidator
from ..data import build_mixed_dataset


class CustomPoseTrainer(PoseTrainer):
    def __init__(self, *args, **kwargs):
        # 提取 symmetry_match 参数
        self.symmetry_categories = kwargs.get("overrides", {}).pop("symmetry_categories", None)
        self.symmetry_pairs = kwargs.get("overrides", {}).pop("symmetry_pairs", None)
        
        # 提取 mixed_data 参数
        overrides = kwargs.get("overrides", {})
        self.class_mapping = overrides.pop("class_mapping", None)
        self.mixed_alpha = overrides.pop("mixed_alpha", None)
        self.background_data = overrides.pop("background_data", None)
        
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_mixed_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_model(self, cfg=None, weights=None, verbose=True):
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
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        model = unwrap_model(self.model)
        if hasattr(model, "student_model"):
            model = model.student_model
        if getattr(model.model[-1], "flow_model", None) is not None:
            self.loss_names += ("rle_loss",)
        return CustomPoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks,
            symmetry_categories=self.symmetry_categories, symmetry_pairs=self.symmetry_pairs,
        )
    
    def _model_train(self):
        super()._model_train()
        if self.mixed_alpha is not None and self.mixed_alpha >= 0:
            random.seed(self.epoch)
            self._recreate_train_loader()
    
    def _recreate_train_loader(self):
        if isinstance(self.args.data, str):
            with open(self.args.data, 'r', encoding='utf-8') as f:
                data_dict = yaml.safe_load(f)
            dataset_path = data_dict["path"] + "/" + data_dict["train"]
        else:
            dataset_path = self.args.data["train"]
        
        self.train_loader = self.get_dataloader(
            dataset_path,
            batch_size=self.args.batch,
            rank=RANK,
            mode="train"
        )
    