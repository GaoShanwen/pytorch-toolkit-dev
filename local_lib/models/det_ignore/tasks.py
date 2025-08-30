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
    # def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
    #     """Initialize YOLOv8 OBB model with given config and parameters."""
    #     super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
    
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return WithIgnoreLoss(self)