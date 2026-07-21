######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2025.08.28
# filename: yolopro.py
# function: use custom yolo.
######################################################
from ultralytics import YOLO
from ultralytics.models import yolo

from .symmetry_match.tasks import SymmetryMatchPoseModel
from .symmetry_match.val import SymmetryMatchPoseValidator
from .symmetry_match.train import SymmetryMatchPoseTrainer


class YOLOPro(YOLO):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "pose": {
                "model": SymmetryMatchPoseModel,
                "trainer": SymmetryMatchPoseTrainer,
                "validator": SymmetryMatchPoseValidator,
                "predictor": yolo.pose.PosePredictor,
            }
        }
        
