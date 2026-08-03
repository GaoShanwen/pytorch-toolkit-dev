######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2025.08.28
# filename: yolopro.py
# function: use custom yolo.
######################################################
from ultralytics import YOLO
from ultralytics.models import yolo

from .train import CustomPoseTrainer
from .val import CustomPoseValidator
from .symmetry_match.tasks import SymmetryMatchPoseModel
from .symmetry_match.val import SymmetryMatchPoseValidator
from .symmetry_match.train import SymmetryMatchPoseTrainer
from .mixed_data.train import MixedDataTrainer
from .mixed_data.val import MixedDataValidator
from ultralytics.nn.tasks import PoseModel


class YOLOPro(YOLO):
    def __init__(self, model="yolo26n.pt", task=None):
        self.use_mixed_data = False
        self.use_symmetry_match = False
        super().__init__(model=model, task=task)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        if self.use_symmetry_match and self.use_mixed_data:
            base_map = {
                "pose": {
                    "model": SymmetryMatchPoseModel,
                    "trainer": CustomPoseTrainer,
                    "validator": CustomPoseValidator,
                    "predictor": yolo.pose.PosePredictor,
                }
            }
        elif self.use_symmetry_match:
            base_map = {
                "pose": {
                    "model": SymmetryMatchPoseModel,
                    "trainer": SymmetryMatchPoseTrainer,
                    "validator": SymmetryMatchPoseValidator,
                    "predictor": yolo.pose.PosePredictor,
                }
            }
        else:
            base_map = {
                "pose": {
                    "model": PoseModel,
                    "trainer": MixedDataTrainer,
                    "validator": MixedDataValidator,
                    "predictor": yolo.pose.PosePredictor,
                }
            }
            
        return base_map

    def train(self, **kwargs):
        self.use_mixed_data = kwargs.pop("mixed_data", False)
        self.use_symmetry_match = kwargs.pop("symmetry_match", False)
        assert self.use_mixed_data or self.use_symmetry_match, "At least one of mixed_data or symmetry_match must be setted!"
        super().train(**kwargs)

    def tune(self, **kwargs):
        self.use_mixed_data = kwargs.pop("mixed_data", False)
        self.use_symmetry_match = kwargs.pop("symmetry_match", False)
        assert self.use_mixed_data or self.use_symmetry_match, "At least one of mixed_data or symmetry_match must be setted!"
        super().tune(**kwargs)