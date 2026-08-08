import torch
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils.metrics import box_iou


class SymmetryMatchPoseValidator(PoseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None, **kwargs):
        super().__init__(dataloader, save_dir, args=args, _callbacks=_callbacks)
        self.symmetry_categories = kwargs.get("symmetry_categories", None)
        self.symmetry_pairs = kwargs.get("symmetry_pairs", None)
        assert self.symmetry_categories is not None and self.symmetry_pairs is not None, \
            "Symmetry categories and pairs must be set!"

    def update_metrics(self, preds, batch):
        """Apply symmetry matching to predictions before metric computation."""
        # Apply symmetry matching to predictions if configured
        preds = self._apply_symmetry_matching(preds, batch)
        
        super().update_metrics(preds, batch)

    def _apply_symmetry_matching(self, preds, batch):
        """Apply symmetry matching to predictions, preserving original structure."""
        gt_kpts = batch.get("keypoints", None)
        if gt_kpts is None:
            return preds
        
        # 记录是否有任何修改
        for si, pred in enumerate(preds):
            if len(pred) == 0:
                continue
            
            # pred 是 dict 类型，包含 'boxes', 'keypoints', 'cls', 'conf' 等键
            kpts = pred.get('keypoints', None)
            boxes = pred.get('bboxes', None)
            cls = pred.get('cls', None)
            
            if kpts is None or boxes is None or cls is None:
                continue
            
            # 获取原始形状
            original_shape = kpts.shape
            # 判断关键点形状
            if kpts.dim() == 3:
                n_kpts = kpts.shape[1]
                kpts_reshaped = kpts.clone()
            elif kpts.dim() == 2:
                n_kpts = kpts.shape[1] // 3
                kpts_reshaped = kpts.reshape(-1, n_kpts, 3)
            else:
                continue
            
            # 获取当前图像的GT数据
            batch_idx = batch["batch_idx"] == si
            gt_kpts_sample = gt_kpts[batch_idx] if gt_kpts is not None else None
            gt_boxes = batch["bboxes"][batch_idx]
            
            if gt_kpts_sample is None or len(gt_kpts_sample) == 0 or len(gt_boxes) == 0:
                continue
            
            # ========== 关键修复：统一坐标尺度 ==========
            # 获取图像尺寸信息
            imgsz = batch.get('img').shape[2:]  # [height, width]
            
            # pred_boxes 是像素坐标，需要转换到与 gt_boxes 相同的尺度
            # gt_boxes 通常是归一化坐标 [0, 1]，需要转换为像素坐标
            gt_boxes_pixel = gt_boxes.clone()
            if len(imgsz) == 2:
                gt_boxes_pixel[..., [0, 2]] *= imgsz[1]  # x 坐标乘以宽度
                gt_boxes_pixel[..., [1, 3]] *= imgsz[0]  # y 坐标乘以高度
                # 先把宽高拷贝出来，防止后续被覆盖
                w = gt_boxes_pixel[..., 2].clone()
                h = gt_boxes_pixel[..., 3].clone()

                gt_boxes_pixel[..., 0] -= w / 2
                gt_boxes_pixel[..., 1] -= h / 2
                gt_boxes_pixel[..., 2] = gt_boxes_pixel[..., 0] + w
                gt_boxes_pixel[..., 3] = gt_boxes_pixel[..., 1] + h

            # 现在 pred_boxes 和 gt_boxes_pixel 都是像素坐标，可以计算 IoU
            iou_matrix = box_iou(boxes, gt_boxes_pixel)
            matches = torch.argmax(iou_matrix, dim=1)
            
            # 向量化对称匹配
            for i, j in self.symmetry_pairs:
                if i >= n_kpts or j >= n_kpts:
                    continue
                
                is_symmetry_cls = torch.isin(cls, torch.tensor(self.symmetry_categories, device=cls.device))
                if not is_symmetry_cls.any():
                    continue
                
                matched_gt = gt_kpts_sample[matches]
                
                # GT关键点也需要转换到像素坐标
                matched_gt_pixel = matched_gt.clone()
                if len(imgsz) == 2:
                    matched_gt_pixel[..., [0, 1]] *= torch.tensor(imgsz[::-1], device=matched_gt.device)
                
                # 计算距离（现在都是像素坐标）
                dist_orig = (kpts_reshaped[is_symmetry_cls, i, :2] - matched_gt_pixel[is_symmetry_cls, i, :2]).norm(dim=-1) + \
                           (kpts_reshaped[is_symmetry_cls, j, :2] - matched_gt_pixel[is_symmetry_cls, j, :2]).norm(dim=-1)
                
                dist_swap = (kpts_reshaped[is_symmetry_cls, i, :2] - matched_gt_pixel[is_symmetry_cls, j, :2]).norm(dim=-1) + \
                            (kpts_reshaped[is_symmetry_cls, j, :2] - matched_gt_pixel[is_symmetry_cls, i, :2]).norm(dim=-1)
                
                need_swap = dist_swap < dist_orig
                
                if need_swap.any():
                    original_i = kpts_reshaped[is_symmetry_cls, i, :3].clone()
                    original_j = kpts_reshaped[is_symmetry_cls, j, :3].clone()
                    
                    kpts_reshaped[is_symmetry_cls, i, :3] = torch.where(need_swap.unsqueeze(-1), original_j, original_i)
                    kpts_reshaped[is_symmetry_cls, j, :3] = torch.where(need_swap.unsqueeze(-1), original_i, original_j)
            
            # 写回处理后的关键点
            if original_shape == kpts_reshaped.shape:
                pred['keypoints'] = kpts_reshaped
            else:
                pred['keypoints'] = kpts_reshaped.reshape(-1, n_kpts * 3)
                
        return preds
