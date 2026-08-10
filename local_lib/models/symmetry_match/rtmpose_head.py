from typing import Tuple, Optional

import torch
from mmpose.models.heads.coord_cls_heads import RTMCCHead
from mmpose.registry import MODELS
from mmpose.utils.typing import OptSampleList, OptConfigType, Tensor
from mmengine.structures import InstanceData


@MODELS.register_module()
class SymmetryMatchRTMCCHead(RTMCCHead):
    """RTMCCHead with category-aware symmetry matching.

    Extracts category IDs from batch_data_samples and sets a symmetry mask
    on the loss module before computing the loss. Only samples belonging to
    symmetry_categories will have symmetry matching applied.

    Args:
        symmetry_categories (list[int]): Category IDs eligible for symmetry
            matching. Default: [] (disabled). Set to e.g. [0, 1, 2] to
            enable for specific categories.
        **kwargs: Arguments passed to RTMCCHead.
    """

    def __init__(self, symmetry_categories=None, symmetry_pairs=None, **kwargs):
        super().__init__(**kwargs)
        self.symmetry_categories = symmetry_categories
        self.symmetry_pairs = symmetry_pairs
        assert self.symmetry_categories is not None, "symmetry_categories must be provided"
        assert self.symmetry_pairs is not None, "symmetry_pairs must be provided"

    def _compute_per_keypoint_loss(self, pred, target):
        N, K, L = pred.shape
        pred_flat = pred.reshape(-1, L)
        target_flat = target.reshape(-1, L)
        t_loss = self.loss_module.criterion(pred_flat, target_flat)
        return t_loss.reshape(N, K)

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses with category-aware symmetry matching.

        When symmetry is disabled (no symmetry_categories or loss_module
        doesn't support symmetry mask), delegates to parent RTMCCHead.loss.
        """
        pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ], dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ], dim=0)
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ], dim=0)

        category_ids = self._extract_category_ids(batch_data_samples)
        if category_ids is not None:
            category_ids = category_ids.to(gt_x.device)

        N, K = pred_x.shape[0], pred_x.shape[1]
        lm = self.loss_module

        t_loss_x = self._compute_per_keypoint_loss(pred_x, gt_x)
        t_loss_y = self._compute_per_keypoint_loss(pred_y, gt_y)

        # calculate accuracy with symmetry-aware GT matching
        gt_x_acc, gt_y_acc = gt_x.clone(), gt_y.clone()
        if category_ids is not None:
            sym_mask = torch.zeros(len(category_ids), dtype=torch.bool, device=gt_x.device)
            for cat in self.symmetry_categories:
                sym_mask |= (category_ids == cat)
            if sym_mask.any():
                for i, j in self.symmetry_pairs:
                    gt_x_swap = gt_x_acc.clone()
                    gt_x_swap[:, i], gt_x_swap[:, j] = gt_x_acc[:, j].clone(), gt_x_acc[:, i].clone()
                    t_loss_swap_x = self._compute_per_keypoint_loss(pred_x, gt_x_swap)

                    gt_y_swap = gt_y_acc.clone()
                    gt_y_swap[:, i], gt_y_swap[:, j] = gt_y_acc[:, j].clone(), gt_y_acc[:, i].clone()
                    t_loss_swap_y = self._compute_per_keypoint_loss(pred_y, gt_y_swap)

                    pair_orig = t_loss_x[:, i] + t_loss_x[:, j] + t_loss_y[:, i] + t_loss_y[:, j]
                    pair_swap = t_loss_swap_x[:, i] + t_loss_swap_x[:, j] + t_loss_swap_y[:, i] + t_loss_swap_y[:, j]
                    take_swap = (pair_swap < pair_orig) & sym_mask
                    if take_swap.any():
                        gt_x_acc[take_swap, i], gt_x_acc[take_swap, j] = gt_x_acc[take_swap, j].clone(), gt_x_acc[take_swap, i].clone()
                        gt_y_acc[take_swap, i], gt_y_acc[take_swap, j] = gt_y_acc[take_swap, j].clone(), gt_y_acc[take_swap, i].clone()

            t_loss_x = self._compute_per_keypoint_loss(pred_x, gt_x_acc)
            t_loss_y = self._compute_per_keypoint_loss(pred_y, gt_y_acc)

        pred_simcc = (pred_x, pred_y)
        if lm.use_target_weight:
            weight = keypoint_weights.reshape(-1)
        else:
            weight = 1.

        loss = 0
        for t_loss in [t_loss_x, t_loss_y]:
            t_loss = t_loss.reshape(-1).mul(weight)
            if lm.mask is not None:
                t_loss = t_loss.reshape(N, K)
                t_loss[:, lm.mask] = t_loss[:, lm.mask] * lm.mask_weight
            loss = loss + t_loss.sum()
        loss = loss / K

        losses = dict()
        losses.update(loss_kpt=loss)

        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy((gt_x_acc, gt_y_acc)),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @staticmethod
    def _extract_category_ids(batch_data_samples) -> Optional[torch.Tensor]:
        """Extract category IDs from batch data samples.

        category_id is stored in metainfo by PackPoseInputs (not in gt_instances).
        Note: base_coco_style_dataset stores category_id as np.array(int),
        which is a 0-d ndarray. We must convert it to a Python scalar.

        Args:
            batch_data_samples: List of PoseDataSample objects.

        Returns:
            Tensor of shape (N,) with category IDs, or None if unavailable.
        """
        import numpy as np
        ids = []
        for d in batch_data_samples:
            cid = d.get('category_id', None)
            if cid is None:
                return None
            if isinstance(cid, torch.Tensor) or isinstance(cid, np.ndarray):
                cid = cid.item()
            ids.append(cid)
        return torch.tensor(ids, dtype=torch.long)


# Import here to avoid circular imports
from mmpose.models.heads.coord_cls_heads.rtmcc_head import (simcc_pck_accuracy,
                                                              to_numpy)