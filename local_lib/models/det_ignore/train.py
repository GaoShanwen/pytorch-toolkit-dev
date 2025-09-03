######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2025.08.29
# filename: train.py
# function: create detect trainer for ignore region.
######################################################
from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import de_parallel

from ...data import YOLOProDataset, build_yolopro_dataset
from .tasks import WithIgnoreModel
from .val import WithIgnoreValidator

class WithIgnoreTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolopro_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super(WithIgnoreTrainer, self).get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)
        loader.collate_fn = YOLOProDataset.collate_fn
        return loader

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = WithIgnoreModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return WithIgnoreValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )