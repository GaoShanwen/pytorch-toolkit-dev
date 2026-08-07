import torch

from ultralytics.utils.loss import E2ELoss, PoseLoss26, v8PoseLoss
from ultralytics.utils.ops import xyxy2xywh

from .symmetry_match.loss import SymmetryMatchPoseLoss
from .quad_eiou.loss import QuadPoseLoss, QuadEIOULoss


class CustomPoseLoss(SymmetryMatchPoseLoss, QuadPoseLoss):
    """Combined loss supporting both symmetry matching and quad EIoU features.

    MRO: CustomPoseLoss -> SymmetryMatchPoseLoss -> QuadPoseLoss -> PoseLoss26
    - calculate_keypoints_loss: overridden to combine symmetry matching + EIoU exclusion
    - _compute_eiou: from QuadPoseLoss (EIoU computation)
    - get_assigned_targets_and_loss: from SymmetryMatchPoseLoss (identical in both)
    - loss: overridden to combine both symmetry matching and EIoU
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        symmetry_categories: list[int] = None,
        symmetry_pairs: list[tuple[int, int]] = None,
        quad_categories: list[int] = None,
        quad_indices: list[int] = None,
    ):
        self.symmetry_categories = symmetry_categories or getattr(model, "symmetry_categories", None)
        self.symmetry_pairs = symmetry_pairs or getattr(model, "symmetry_pairs", None)
        self.quad_categories = quad_categories or getattr(model, "quad_categories", None)
        self.quad_indices = quad_indices or getattr(model, "quad_indices", None)
        PoseLoss26.__init__(self, model, tal_topk, tal_topk2)
        self.quad_loss_fn = QuadEIOULoss(self.quad_categories, self.quad_indices, self.device)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
        target_cls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate keypoints loss combining symmetry matching and EIoU exclusion.

        Symmetry matching: for symmetric keypoint pairs, swap to minimize loss.
        EIoU exclusion: targets of quad_categories with all quad_indices visible
        are excluded from kpts_loss (but still participate in rle_loss and kpts_obj_loss).
        """
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)

            # --- Symmetry matching: swap symmetric keypoints to minimize loss ---
            modified_gt_kpt = gt_kpt.clone()

            cls_mask = torch.zeros_like(target_cls, dtype=torch.bool)
            for cat in self.symmetry_categories:
                cls_mask |= (target_cls == cat)
            cls_mask = cls_mask[masks]

            if cls_mask.any():
                for i, j in self.symmetry_pairs:
                    dist_orig = (pred_kpt[cls_mask, i, :2] - gt_kpt[cls_mask, i, :2]).norm(dim=-1) + \
                                (pred_kpt[cls_mask, j, :2] - gt_kpt[cls_mask, j, :2]).norm(dim=-1)
                    dist_swap = (pred_kpt[cls_mask, i, :2] - gt_kpt[cls_mask, j, :2]).norm(dim=-1) + \
                                (pred_kpt[cls_mask, j, :2] - gt_kpt[cls_mask, i, :2]).norm(dim=-1)

                    need_swap = dist_swap < dist_orig
                    modified_gt_kpt[cls_mask, i, :] = torch.where(
                        need_swap.unsqueeze(-1), gt_kpt[cls_mask, j, :], gt_kpt[cls_mask, i, :]
                    )
                    modified_gt_kpt[cls_mask, j, :] = torch.where(
                        need_swap.unsqueeze(-1), gt_kpt[cls_mask, i, :], gt_kpt[cls_mask, j, :]
                    )

            # --- EIoU exclusion: exclude quad_categories targets with all quad_indices visible ---
            eiou_cls_mask = torch.zeros_like(target_cls, dtype=torch.bool)
            for cat in self.quad_categories:
                eiou_cls_mask |= (target_cls == cat)
            eiou_cls_mask = eiou_cls_mask[masks]  # [N_fg]

            gt_quad_vis = gt_kpt[:, self.quad_indices, 2]  # [N_fg, len(quad_indices)]
            quad_vis_mask = (gt_quad_vis == 2).all(dim=-1)  # [N_fg]

            eiou_exclude_mask = eiou_cls_mask & quad_vis_mask  # [N_fg]

            # kpts_loss uses filtered mask (exclude eiou samples)
            kpt_loss_mask = kpt_mask & ~eiou_exclude_mask.unsqueeze(-1).expand_as(kpt_mask)

            if kpt_loss_mask.any():
                kpts_loss = self.keypoint_loss(pred_kpt, modified_gt_kpt, kpt_loss_mask, area)

            # rle_loss and kpts_obj_loss use original mask (include eiou samples)
            if kpt_mask.any():
                if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                    rle_loss = self.calculate_rle_loss(pred_kpt, modified_gt_kpt, kpt_mask)
                    rle_loss = rle_loss.clamp(min=0)
                if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                    kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss, rle_loss

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        num_losses = 6 if self.rle_loss else 5
        num_losses += 1  # eiou_loss
        loss = torch.zeros(num_losses, device=self.device)
        # box, kpt_location, kpt_visibility, cls, dfl[, rle], eiou
        (fg_mask, target_gt_idx, target_bboxes, target_cls, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]
        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
                target_cls,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

            eiou_loss = self._compute_eiou(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                pred_kpts,
                target_cls,
            )
            eiou_idx = 6 if self.rle_loss else 5
            loss[eiou_idx] = eiou_loss

        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle
        eiou_idx = 6 if self.rle_loss else 5
        loss[eiou_idx] *= getattr(self.hyp, "eiou", 1.0)

        return loss * batch_size, loss.detach()