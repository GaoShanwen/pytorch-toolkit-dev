######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2025.08.29
# filename: loss.py
# function: create detect loss for ignore region.
######################################################
import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import make_anchors

from ...data import box_ioa
from ...utils.tal import WithIgnoredAssigner

class WithIgnoreLoss(v8DetectionLoss):
    def __init__(self, model, ioav=0.25):
        self.ioav = ioav
        super(WithIgnoreLoss, self).__init__(model)
        self.assigner = WithIgnoredAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"], batch["iscrowds"].view(-1, 1)), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes, gt_iscrowds = targets.split((1, 4, 1), 2)  # cls, xyxy, iscrowd
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        
        target_labels, target_bboxes, target_iscrowds, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_iscrowds,
            mask_gt,
        )

        # Filter iscrowd predictions
        fg_mask_sum = fg_mask.sum()
        if fg_mask_sum:
            target_bboxes /= stride_tensor
            target_iscrowds = target_iscrowds[fg_mask].bool()
            if target_iscrowds.sum():
                _ioas = box_ioa(pred_bboxes[fg_mask].T, target_bboxes[fg_mask], istrain=True)
                save_idx = ((_ioas <= self.ioav) | ~target_iscrowds | \
                            target_labels[fg_mask] != torch.argmax(pred_scores[fg_mask], dim=1))
                device = fg_mask.device
                keeps = torch.where(
                    save_idx, 
                    torch.ones(fg_mask_sum, dtype=torch.float32, device=device),
                    torch.zeros(fg_mask_sum, dtype=torch.float32, device=device)
                )

                target_scores[fg_mask].mul_(keeps[:, None])
                with torch.no_grad():
                    pred_scores[fg_mask].mul_(keeps[:, None])
                fg_mask[fg_mask.clone()] = keeps.bool()
                del keeps, save_idx, _ioas
                torch.cuda.empty_cache()

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            # target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
