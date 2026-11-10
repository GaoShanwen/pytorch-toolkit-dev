######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: seg_metrics.py
# function: confusion-matrix based segmentation metrics.
######################################################
import torch
import segmentation_models_pytorch as smp


class SegMetricTracker:
    """Accumulate the confusion matrix over an epoch and derive IoU / F1.

    Args:
        num_classes: number of classes (including background).
        ignore_index: label id excluded from the statistics.
    """

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes, dtype=torch.float64)
        self.fp = torch.zeros(self.num_classes, dtype=torch.float64)
        self.fn = torch.zeros(self.num_classes, dtype=torch.float64)
        self.tn = torch.zeros(self.num_classes, dtype=torch.float64)
        self.loss_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, pred, target, loss=None):
        """Update the statistics with one batch.

        Args:
            pred: predicted class indices, shape ``(N, H, W)``.
            target: ground-truth class indices, shape ``(N, H, W)``.
            loss: optional scalar loss value of this batch.
        """
        pred = pred.detach()
        target = target.detach()
        b_tp, b_fp, b_fn, b_tn = smp.metrics.get_stats(
            pred,
            target,
            mode="multiclass",
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        self.tp += b_tp.sum(dim=0).double().cpu()
        self.fp += b_fp.sum(dim=0).double().cpu()
        self.fn += b_fn.sum(dim=0).double().cpu()
        self.tn += b_tn.sum(dim=0).double().cpu()
        if loss is not None:
            self.loss_sum += float(loss)
            self.count += 1

    def _iou(self):
        """Per-class IoU, shape ``(num_classes,)``."""
        return smp.metrics.iou_score(
            self.tp, self.fp, self.fn, self.tn, reduction="none", zero_division=0
        )

    def compute(self, classes=None):
        """Return the metrics of the accumulated statistics.

        The returned dict contains ``loss``, ``iou`` (pixel-accuracy weighted
        IoU), ``miou`` (mean over the classes) and ``iou_per_class``.
        """
        classes = classes or [str(i) for i in range(self.num_classes)]
        per_class = self._iou()
        # mean over the classes that actually appear in the ground truth
        present = (self.tp + self.fn) > 0
        miou = (
            per_class[present].mean().item()
            if present.any()
            else per_class.mean().item()
        )
        # micro IoU: IoU computed over all the pixels at once
        micro = smp.metrics.iou_score(
            self.tp.sum().reshape(1),
            self.fp.sum().reshape(1),
            self.fn.sum().reshape(1),
            self.tn.sum().reshape(1),
            reduction="micro",
            zero_division=0,
        ).item()

        metrics = {
            "loss": self.loss_sum / self.count if self.count else 0.0,
            "iou": micro,
            "miou": miou,
            "iou_per_class": {
                cls: round(float(per_class[i]), 6) for i, cls in enumerate(classes)
            },
        }
        return metrics
