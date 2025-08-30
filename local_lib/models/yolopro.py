######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2025.08.28
# filename: yolopro.py
# function: use custom yolo.
######################################################
# from ultralytics.engine.model import Model
from ultralytics import YOLO
from .det_ignore import WithIgnoreModel, WithIgnoreTrainer, WithIgnoreValidator, DetectionPredictor

class YOLOPro(YOLO):
    """YOLO (You Only Look Once) object detection model."""
    def __init__(self, model = "yolov8n.pt", task = None, verbose = False):
        super().__init__(model, task, verbose)

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

    # def val(
    #     self,
    #     validator=None,
    #     **kwargs,
    # ):
    #     """
    #     Validates the model using a specified dataset and validation configuration.

    #     This method facilitates the model validation process, allowing for a range of customization through various
    #     settings and configurations. It supports validation with a custom validator or the default validation approach.
    #     The method combines default configurations, method-specific defaults, and user-provided arguments to configure
    #     the validation process. After validation, it updates the model's metrics with the results obtained from the
    #     validator.

    #     The method supports various arguments that allow customization of the validation process. For a comprehensive
    #     list of all configurable options, users should refer to the 'configuration' section in the documentation.

    #     Args:
    #         validator (BaseValidator, optional): An instance of a custom validator class for validating the model. If
    #             None, the method uses a default validator. Defaults to None.
    #         **kwargs (any): Arbitrary keyword arguments representing the validation configuration. These arguments are
    #             used to customize various aspects of the validation process.

    #     Returns:
    #         (dict): Validation metrics obtained from the validation process.

    #     Raises:
    #         AssertionError: If the model is not a PyTorch model.
    #     """
    #     custom = {"rect": True}  # method defaults
    #     args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

    #     validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
    #     validator(model=self.model)
    #     self.metrics = validator.metrics
    #     return validator.metrics
