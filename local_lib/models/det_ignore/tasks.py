######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2025.08.29
# filename: tasks.py
# function: create detect model for ignore region.
######################################################
from ultralytics.nn.tasks import DetectionModel

from .loss import WithIgnoreLoss

class WithIgnoreModel(DetectionModel):
    """YOLOv8 detection model."""
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return WithIgnoreLoss(self)
