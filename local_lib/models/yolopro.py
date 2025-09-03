######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2025.08.28
# filename: yolopro.py
# function: use custom yolo.
######################################################
from ultralytics import YOLO
from .det_ignore import WithIgnoreModel, WithIgnoreTrainer, WithIgnoreValidator, DetectionPredictor


class YOLOPro(YOLO):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": WithIgnoreModel,
                "trainer": WithIgnoreTrainer,
                "validator": WithIgnoreValidator,
                "predictor": DetectionPredictor,
            }
        }

