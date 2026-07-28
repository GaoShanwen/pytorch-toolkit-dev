from copy import copy
import random
import yaml

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from ...data import MixedDataset, build_mixed_dataset
from .val import MixedDataValidator


class MixedDataTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        overrides = kwargs.get("overrides", {})
        self.class_mapping = overrides.pop("class_mapping", None)
        self.mixed_alpha = overrides.pop("mixed_alpha", None)
        self.background_data = overrides.pop("background_data", None)

        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_mixed_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_validator(self):
        """Return an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        model = unwrap_model(self.model)
        if hasattr(model, "student_model"):
            model = model.student_model  # copy_attr does not copy nn.Module attributes like .model
        if getattr(model.model[-1], "flow_model", None) is not None:
            self.loss_names += ("rle_loss",)
        return MixedDataValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _model_train(self):
        super()._model_train()
        if self.mixed_alpha >= 0:
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