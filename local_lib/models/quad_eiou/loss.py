import os
import torch
import torch.nn as nn
from datetime import datetime
from typing import Any

from ultralytics.utils.loss import E2ELoss, PoseLoss26, v8PoseLoss
from ultralytics.utils.metrics import probiou
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.tal import make_anchors

from .utils import quad_aligned_bbox


class QuadEIOULoss(nn.Module):
    """Dedicated EIOU loss computation for oriented bounding boxes derived from quadrilateral keypoints.

    This class encapsulates all EIOU-specific logic including:
    - Class filtering based on selected categories
    - Keypoint visibility checking
    - Oriented bounding box computation from quad keypoints
    - EIOU loss calculation (center distance + width/height difference)
    """

    def __init__(self, eiou_categories: list[int], eiou_quad_indices: list[int], device: torch.device):
        self.eiou_categories = eiou_categories
        self.eiou_quad_indices = eiou_quad_indices
        self.device = device
        self._err_count = 0  # track how many times error was logged
        self._err_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "err.log")

    @staticmethod
    def _log_error(log_path: str, count: int, **kwargs) -> None:
        """Write error details to err.log when EIOU produces inf/nan."""
        with open(log_path, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] EIOU error #{count}\n")
            for name, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    v = value.detach().cpu()
                    f.write(f"  {name}: shape={list(v.shape)}, dtype={v.dtype}\n")
                    f.write(f"    min={v.min().item():.6f}, max={v.max().item():.6f}\n")
                    f.write(f"    nan={v.isnan().any().item()}, inf={v.isinf().any().item()}\n")
                    f.write(f"    values={v.flatten().tolist()}\n")
                elif isinstance(value, list):
                    f.write(f"  {name}: {value}\n")
                else:
                    f.write(f"  {name}: {value}\n")
            f.write(f"{'=' * 80}\n\n")

    def __call__(
        self, gt_kpt: torch.Tensor, pred_kpt: torch.Tensor, target_cls: torch.Tensor
    ) -> torch.Tensor:
        """Compute EIOU loss for samples matching selected categories with fully visible quad keypoints.

        Args:
            gt_kpt: Ground truth keypoints after masking and stride normalization, shape [N, n_kpts, 3].
            pred_kpt: Predicted keypoints after masking, shape [N, n_kpts, kpts_dim].
            target_cls: Target class indices for anchors, shape [N_fg].

        Returns:
            eiou_loss: The EIOU loss scalar (0.0 if no valid samples).
        """
        eiou_loss = torch.tensor(0.0, device=self.device)

        # --- 1. Class filter ---
        cls_mask = torch.zeros_like(target_cls, dtype=torch.bool)
        for cat in self.eiou_categories:
            cls_mask |= (target_cls == cat)

        if not cls_mask.any():
            return eiou_loss

        qi = self.eiou_quad_indices

        # --- 2. Visibility check: all 4 quad keypoints must be fully visible (2) ---
        gt_vis = gt_kpt[cls_mask][:, qi, 2]  # [N', 4]
        vis_mask = (gt_vis == 2).all(dim=-1)  # [N']

        if not vis_mask.any():
            return eiou_loss

        # --- 3. Build quads (逆时针: 0=左上, 1=左下, 2=右下, 3=右上) ---
        gt_quads = gt_kpt[cls_mask][vis_mask][:, qi, :2]  # [M, 4, 2]
        pred_quads = pred_kpt[cls_mask][vis_mask][:, qi, :2]  # [M, 4, 2]

        # --- 4. Compute GT direction vector (前两个点中心 → 后两个点中心) ---
        gt_top_center = (gt_quads[:, 0] + gt_quads[:, 3]) / 2
        gt_bottom_center = (gt_quads[:, 1] + gt_quads[:, 2]) / 2
        gt_direction = gt_top_center - gt_bottom_center  # [M, 2]

        # --- 5. Compute oriented bounding boxes using GT direction for both ---
        gt_W, gt_H, _, gt_centers = quad_aligned_bbox(gt_quads, direction=gt_direction)
        pred_W, pred_H, _, pred_centers = quad_aligned_bbox(pred_quads, direction=gt_direction)

        # --- 6. EIOU: (1 - IoU) + center_loss + wh_loss ---
        eps = 1e-5  # to avoid division by zero
        # Convert OBBs to xywhr format for probiou: [cx, cy, w, h, angle]
        angle = torch.atan2(gt_direction[..., 1], gt_direction[..., 0])
        gt_obbs = torch.cat([gt_centers, gt_W.unsqueeze(1), gt_H.unsqueeze(1), angle.unsqueeze(1)], dim=1)
        pred_obbs = torch.cat([pred_centers, pred_W.unsqueeze(1), pred_H.unsqueeze(1), angle.unsqueeze(1)], dim=1)
        iou = probiou(gt_obbs, pred_obbs)  # [M], differentiable

        center_dist = torch.norm(gt_centers - pred_centers, dim=-1)
        diagonal = torch.sqrt(gt_W ** 2 + gt_H ** 2 + eps)
        center_loss = center_dist / diagonal

        wh_loss = (
            (gt_W - pred_W).abs() / gt_W.clamp_min(eps)
            + (gt_H - pred_H).abs() / gt_H.clamp_min(eps)
        )

        eiou_loss = ((1.0 - iou) + center_loss + wh_loss).mean()
        return eiou_loss


class QuadPoseLoss(PoseLoss26):
    """Criterion class for computing training losses with EIOU loss for oriented bounding boxes.

    This class serves as a scheduler/switch:
    - Stores EIOU configuration parameters
    - Delegates actual EIOU computation to QuadEIOULoss
    - Orchestrates box/cls/dfl/rle/keypoint/eiou losses in the loss() method
    """

    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None,
        quad_categories: list[int] = None, quad_indices: list[int] = None,
    ):
        """Initialize QuadPoseLoss with model parameters and EIOU-specific settings."""
        super().__init__(model, tal_topk, tal_topk2)
        self.quad_categories = quad_categories or getattr(model, 'quad_categories', None)
        self.quad_indices = quad_indices or getattr(model, 'quad_indices', None)
        assert self.quad_categories is not None, "quad_categories must be set!"
        assert self.quad_indices is not None, "quad_indices must be set!"

        # Dedicated EIOU loss computation engine
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
        """Calculate the keypoints loss, excluding EIoU category targets.

        Targets belonging to quad_categories use EIoU loss instead of standard keypoints loss,
        so their kpt_mask is set to False to exclude them from kpts_loss / kpts_obj_loss / rle_loss.

        Args:
            masks: Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx: Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints: Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx: Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor: Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes: Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts: Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
            target_cls: Target class indices, shape (BS, N_anchors).

        Returns:
            kpts_loss, kpts_obj_loss, rle_loss
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

            # Exclude EIoU category targets from keypoints loss only when all quad keypoints are visible
            eiou_cls_mask = torch.zeros_like(target_cls, dtype=torch.bool)
            for cat in self.quad_categories:
                eiou_cls_mask |= (target_cls == cat)
            eiou_cls_mask = eiou_cls_mask[masks]  # [N_fg]

            # Check visibility of quad keypoints: all must be visible (== 2)
            gt_quad_vis = gt_kpt[:, self.quad_indices, 2]  # [N_fg, len(quad_indices)]
            quad_vis_mask = (gt_quad_vis == 2).all(dim=-1)  # [N_fg]

            # Only exclude when both conditions are met: category matches AND all quad points visible
            eiou_exclude_mask = eiou_cls_mask & quad_vis_mask  # [N_fg]

            # kpts_loss uses filtered mask (exclude eiou samples), rle_loss and kpts_obj_loss use original mask
            kpt_loss_mask = kpt_mask & ~eiou_exclude_mask.unsqueeze(-1).expand_as(kpt_mask)

            if kpt_loss_mask.any():
                kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_loss_mask, area)

            if kpt_mask.any():
                if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                    rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                    rle_loss = rle_loss.clamp(min=0)
                if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                    kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss, rle_loss

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss including EIOU loss for oriented bounding boxes."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        num_losses = 6 if self.rle_loss else 5
        num_losses += 1  # for eiou_loss
        loss = torch.zeros(num_losses, device=self.device)
        # box, kpt_location, kpt_visibility, cls, dfl[, rle], eiou
        (fg_mask, target_gt_idx, target_bboxes, target_cls, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Keypoint loss
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

            # EIOU loss --- delegate to dedicated computation engine
            eiou_loss = self._compute_eiou(
                fg_mask, target_gt_idx, keypoints, batch["batch_idx"].view(-1, 1),
                stride_tensor, pred_kpts, target_cls,
            )
            eiou_idx = 6 if self.rle_loss else 5
            loss[eiou_idx] = eiou_loss

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain
        eiou_idx = 6 if self.rle_loss else 5
        loss[eiou_idx] *= getattr(self.hyp, 'eiou', 1.0)  # eiou gain

        return loss * batch_size, loss.detach()  # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle], eiou)

    def _compute_eiou(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        pred_kpts: torch.Tensor,
        target_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Extract keypoints, apply stride, and delegate to QuadEIOULoss for actual computation.

        This is a thin wrapper that handles the keypoint selection and stride normalization
        common to all pose loss computations, then passes the relevant tensors to the
        dedicated QuadEIOULoss engine.
        """
        if not masks.any():
            return torch.tensor(0.0, device=self.device)

        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        gt_kpt = selected_keypoints[masks]  # [N, n_kpts, 3]
        pred_kpt = pred_kpts[masks]  # [N, n_kpts, kpts_dim]

        return self.quad_loss_fn(gt_kpt, pred_kpt, target_cls[masks])

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        target_cls, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, target_cls, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)


def test_quad_eiou_loss():
    """Test QuadEIOULoss with synthetic data and visualize quads + oriented bounding boxes.

    Generates multiple test cases with different quad shapes, runs QuadEIOULoss,
    draws the GT/pred quad points, oriented bounding boxes, and saves the result as an image.

    Usage:
        python -c "from local_lib.models.quad_eiou.loss import test_quad_eiou_loss; test_quad_eiou_loss()"
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    device = torch.device("cpu")
    n_kpts = 7
    quad_indices = [3, 4, 5, 6]
    eiou_categories = [0]

    # --- Single GT quad (counter-clockwise: 左上, 左下, 右下, 右上) ---
    gt_quad = np.array([[400, 100], [300, 0], [0, 300], [100, 400]], dtype=np.float32)
    center = gt_quad.mean(axis=0)
    cx, cy = center

    # --- Compute oriented bbox of GT for W/H scaling ---
    from .utils import quad_aligned_bbox
    gt_quad_t = torch.from_numpy(gt_quad).unsqueeze(0).to(device)
    gt_W, gt_H, gt_bbox_corners, _ = quad_aligned_bbox(gt_quad_t)
    gt_bbox_corners_np = gt_bbox_corners[0].numpy()

    def scale_obbox(bbox_corners, sw, sh):
        """Scale oriented bbox around its center by sw (width) and sh (height)."""
        bc = bbox_corners.mean(axis=0)
        return (bbox_corners - bc) * np.array([sw, sh]) + bc

    # --- Build 4 pred quads as transformations of GT ---
    # 1. Rotate 90° around center
    pred_rot90 = np.empty_like(gt_quad)
    pred_rot90[:, 0] = cx - (gt_quad[:, 1] - cy)
    pred_rot90[:, 1] = cy + (gt_quad[:, 0] - cx)

    # 2. Point symmetry (180° around center)
    pred_sym = 2 * center - gt_quad

    # 3. Width × 2, height / 2 (using oriented bbox corners as quad)
    pred_w2_hhalf = scale_obbox(gt_bbox_corners_np, 2.0, 0.5)

    # 4. Width / 2, height × 2 (using oriented bbox corners as quad)
    pred_whalf_h2 = scale_obbox(gt_bbox_corners_np, 0.5, 2.0)

    pred_quads_list = [pred_rot90, pred_sym, pred_w2_hhalf, pred_whalf_h2]
    pred_labels = ["Rotate 90°", "Point symmetry", "W×2 H/2", "W/2 H×2"]

    N = len(pred_quads_list)
    gt_quads_list = [gt_quad] * N

    # Wrap into full keypoint tensors [N, n_kpts, 3]
    gt_kpt = np.zeros((N, n_kpts, 3), dtype=np.float32)
    pred_kpt_np = np.zeros((N, n_kpts, 3), dtype=np.float32)
    for i in range(N):
        gt_kpt[i, quad_indices, :2] = gt_quads_list[i]
        gt_kpt[i, quad_indices, 2] = 2  # fully visible
        pred_kpt_np[i, quad_indices, :2] = pred_quads_list[i]
        pred_kpt_np[i, quad_indices, 2] = 2

    gt_kpt_t = torch.from_numpy(gt_kpt).to(device)
    pred_kpt_t = torch.from_numpy(pred_kpt_np).to(device)
    target_cls = torch.zeros(N, dtype=torch.long).to(device)  # all class 0

    # --- Run QuadEIOULoss (mean over all samples) ---
    loss_fn = QuadEIOULoss(eiou_categories, quad_indices, device)
    eiou_val = loss_fn(gt_kpt_t, pred_kpt_t, target_cls)
    print(f"EIOU loss (mean): {eiou_val.item():.6f}")

    # --- Compute individual EIOU losses per sample ---
    from .utils import quad_aligned_bbox

    eiou_individual = []
    eps = 1e-8
    for i in range(N):
        gt_q = gt_kpt_t[i:i+1, quad_indices, :2]
        pred_q = pred_kpt_t[i:i+1, quad_indices, :2]
        gt_top_c = (gt_q[:, 0] + gt_q[:, 3]) / 2
        gt_bot_c = (gt_q[:, 1] + gt_q[:, 2]) / 2
        gt_dir = gt_top_c - gt_bot_c
        gt_W_i, gt_H_i, _, gt_ct_i = quad_aligned_bbox(gt_q, direction=gt_dir)
        pred_W_i, pred_H_i, _, pred_ct_i = quad_aligned_bbox(pred_q, direction=gt_dir)
        angle_i = torch.atan2(gt_dir[..., 1], gt_dir[..., 0])
        gt_obb_i = torch.cat([gt_ct_i, gt_W_i.unsqueeze(1), gt_H_i.unsqueeze(1), angle_i.unsqueeze(1)], dim=1)
        pred_obb_i = torch.cat([pred_ct_i, pred_W_i.unsqueeze(1), pred_H_i.unsqueeze(1), angle_i.unsqueeze(1)], dim=1)
        iou_i = probiou(gt_obb_i, pred_obb_i).item()
        center_dist = torch.norm(gt_ct_i - pred_ct_i, dim=-1)
        diagonal = torch.sqrt(gt_W_i ** 2 + gt_H_i ** 2 + eps)
        c_loss = (center_dist / diagonal).item()
        wh_loss = ((gt_W_i - pred_W_i).abs() / (gt_W_i + eps) +
                   (gt_H_i - pred_H_i).abs() / (gt_H_i + eps)).item()
        eiou_i = (1.0 - iou_i) + c_loss + wh_loss
        eiou_individual.append(eiou_i)
        print(f"  [{pred_labels[i]}] IoU={iou_i:.4f}  center={c_loss:.4f}  wh={wh_loss:.4f}  EIOU={eiou_i:.6f}")

    # --- Compute oriented bboxes for visualization ---
    from .utils import quad_aligned_bbox

    gt_quads_t = gt_kpt_t[:, quad_indices, :2]
    pred_quads_t = pred_kpt_t[:, quad_indices, :2]
    gt_top_center = (gt_quads_t[:, 0] + gt_quads_t[:, 3]) / 2
    gt_bottom_center = (gt_quads_t[:, 1] + gt_quads_t[:, 2]) / 2
    gt_dir_vis = gt_top_center - gt_bottom_center
    gt_W, gt_H, gt_bbox_corners, gt_centers = quad_aligned_bbox(gt_quads_t, direction=gt_dir_vis)
    pred_W, pred_H, pred_bbox_corners, pred_centers = quad_aligned_bbox(pred_quads_t, direction=gt_dir_vis)

    gt_W_np = gt_W.numpy()
    gt_H_np = gt_H.numpy()
    pred_W_np = pred_W.numpy()
    pred_H_np = pred_H.numpy()
    gt_centers_np = gt_centers.numpy()
    pred_centers_np = pred_centers.numpy()
    gt_bbox_corners_np = gt_bbox_corners.numpy()
    pred_bbox_corners_np = pred_bbox_corners.numpy()

    # --- Visualization ---
    cols = 2
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    colors_gt = "#2196F3"
    colors_pred = ["#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for i in range(N):
        r, c = i // cols, i % cols
        ax = axes[r, c]

        # Draw GT quad
        gt_q = gt_quads_list[i]
        gt_q_closed = np.vstack([gt_q, gt_q[0]])
        ax.plot(gt_q_closed[:, 0], gt_q_closed[:, 1], "o-", color=colors_gt,
                linewidth=2, markersize=8, label="GT quad")
        for j, (x, y) in enumerate(gt_q):
            ax.text(x, y, f"GT{j}", fontsize=7, color=colors_gt, ha="right", va="bottom")

        # Draw GT oriented bbox
        gt_bb = gt_bbox_corners_np[i]
        gt_bb_closed = np.vstack([gt_bb, gt_bb[0]])
        ax.plot(gt_bb_closed[:, 0], gt_bb_closed[:, 1], "--", color=colors_gt,
                linewidth=1.5, alpha=0.6, label="GT obbox")
        ax.scatter(*gt_centers_np[i], c=colors_gt, marker="x", s=100, linewidths=2, zorder=5)

        # Draw Pred quad
        pred_q = pred_quads_list[i]
        pred_q_closed = np.vstack([pred_q, pred_q[0]])
        ax.plot(pred_q_closed[:, 0], pred_q_closed[:, 1], "s-", color=colors_pred[i],
                linewidth=2, markersize=8, label="Pred quad")
        for j, (x, y) in enumerate(pred_q):
            ax.text(x, y, f"P{j}", fontsize=7, color=colors_pred[i], ha="left", va="top")

        # Draw Pred oriented bbox
        pred_bb = pred_bbox_corners_np[i]
        pred_bb_closed = np.vstack([pred_bb, pred_bb[0]])
        ax.plot(pred_bb_closed[:, 0], pred_bb_closed[:, 1], "--", color=colors_pred[i],
                linewidth=1.5, alpha=0.6, label="Pred obbox")
        ax.scatter(*pred_centers_np[i], c=colors_pred[i], marker="+", s=100, linewidths=2, zorder=5)

        # Connect GT center to Pred center
        ax.plot([gt_centers_np[i, 0], pred_centers_np[i, 0]],
                [gt_centers_np[i, 1], pred_centers_np[i, 1]],
                "k:", linewidth=1, alpha=0.5)

        # Info text
        ax.set_title(
            f"{pred_labels[i]} | GT: {gt_W_np[i]:.1f} x {gt_H_np[i]:.1f} | "
            f"Pred: {pred_W_np[i]:.1f} x {pred_H_np[i]:.1f} | "
            f"EIOU={eiou_individual[i]:.4f}",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"QuadEIOULoss Test — EIOU = {eiou_val.item():.6f}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "test_quad_eiou.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved to: {save_path}")

    print(f"Eiou={eiou_val.item()}")


if __name__ == "__main__":
    test_quad_eiou_loss()