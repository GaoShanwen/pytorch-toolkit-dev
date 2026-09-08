import torch

from ultralytics.utils.loss import v8PoseLoss
from ultralytics.utils.ops import xyxy2xywh


class CustomPoseLoss(v8PoseLoss):
    """Pose loss for custom pose estimation."""

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        symmetry_categories: list[int] = None,
        symmetry_pairs: list[tuple[int, int]] = None,
    ):
        self.symmetry_categories = symmetry_categories or getattr(model, "symmetry_categories", None)
        self.symmetry_pairs = symmetry_pairs or getattr(model, "symmetry_pairs", None)
        super().__init__(model, tal_topk, tal_topk2)