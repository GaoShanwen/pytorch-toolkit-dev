# local_lib/models/combo/val.py
from copy import copy
import torch
import yaml

from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils.metrics import box_iou

from ..data import build_mixed_dataset


class CustomPoseValidator(PoseValidator):
    def __init__(self, *args, symmetry_categories=None, symmetry_pairs=None, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(kwargs.get("args").get("data"), str):
            with open(kwargs.get("args").get("data"), 'r') as f:
                data = yaml.safe_load(f)
                _symmetry_categories = data.get("symmetry_categories", None)
                _symmetry_pairs = data.get("symmetry_pairs", None)
        else:
            _symmetry_categories = data.get("symmetry_categories", None)
            _symmetry_pairs = data.get("symmetry_pairs", None)
        self.symmetry_categories = symmetry_categories or _symmetry_categories
        self.symmetry_pairs = symmetry_pairs or _symmetry_pairs
        assert self.symmetry_categories is not None and self.symmetry_pairs is not None, \
            "symmetry_categories and symmetry_pairs must be provided"

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        orig_plot_matches = self.confusion_matrix.plot_matches
        self.confusion_matrix.plot_matches = lambda img, im_file, save_dir, \
            show_labels=True, show_conf=True: \
            self._categorized_plot_matches(
                orig_plot_matches, img, im_file, save_dir, show_labels, show_conf)

    def _categorized_plot_matches(self, orig_fn, img, im_file, save_dir,
                                  show_labels=True, show_conf=True):
        fp = sum(len(v) for v in self.confusion_matrix.matches["FP"].values())
        fn = sum(len(v) for v in self.confusion_matrix.matches["FN"].values())
        if fp == 0 and fn == 0:
            return
        folder_name = "err" if fp != 0 and fn != 0 else \
            ("fp" if fn == 0 and fp != 0 else "fn")
        orig_fn(img, im_file, save_dir / folder_name, show_labels, show_conf)

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_mixed_dataset(self.args, img_path, batch, self.data, mode=mode, rect=True, stride=self.stride)
    
    def update_metrics(self, preds, batch):
        preds = self._apply_symmetry_matching(preds, batch)
        super().update_metrics(preds, batch)

    def _apply_symmetry_matching(self, preds, batch):
        gt_kpts = batch.get("keypoints", None)
        if gt_kpts is None:
            return preds
        
        for si, pred in enumerate(preds):
            if len(pred) == 0:
                continue
            
            kpts = pred.get('keypoints', None)
            boxes = pred.get('bboxes', None)
            cls = pred.get('cls', None)
            
            if kpts is None or boxes is None or cls is None:
                continue
            
            original_shape = kpts.shape
            if kpts.dim() == 3:
                n_kpts = kpts.shape[1]
                kpts_reshaped = kpts.clone()
            elif kpts.dim() == 2:
                n_kpts = kpts.shape[1] // 3
                kpts_reshaped = kpts.reshape(-1, n_kpts, 3)
            else:
                continue
            
            batch_idx = batch["batch_idx"] == si
            gt_kpts_sample = gt_kpts[batch_idx] if gt_kpts is not None else None
            gt_boxes = batch["bboxes"][batch_idx]
            
            if gt_kpts_sample is None or len(gt_kpts_sample) == 0 or len(gt_boxes) == 0:
                continue
            
            imgsz = batch.get('img').shape[2:]
            
            gt_boxes_pixel = gt_boxes.clone()
            if len(imgsz) == 2:
                gt_boxes_pixel[..., [0, 2]] *= imgsz[1]
                gt_boxes_pixel[..., [1, 3]] *= imgsz[0]
                w = gt_boxes_pixel[..., 2].clone()
                h = gt_boxes_pixel[..., 3].clone()
                gt_boxes_pixel[..., 0] -= w / 2
                gt_boxes_pixel[..., 1] -= h / 2
                gt_boxes_pixel[..., 2] = gt_boxes_pixel[..., 0] + w
                gt_boxes_pixel[..., 3] = gt_boxes_pixel[..., 1] + h
            
            iou_matrix = box_iou(boxes, gt_boxes_pixel)
            matches = torch.argmax(iou_matrix, dim=1)
            
            if self.symmetry_pairs is not None:
                for i, j in self.symmetry_pairs:
                    if i >= n_kpts or j >= n_kpts:
                        continue
                    
                    if self.symmetry_categories is not None:
                        is_symmetry_cls = torch.isin(cls, torch.tensor(self.symmetry_categories, device=cls.device))
                    else:
                        is_symmetry_cls = torch.ones(len(cls), dtype=torch.bool, device=cls.device)
                    
                    if not is_symmetry_cls.any():
                        continue
                    
                    matched_gt = gt_kpts_sample[matches]
                    matched_gt_pixel = matched_gt.clone()
                    if len(imgsz) == 2:
                        matched_gt_pixel[..., [0, 1]] *= torch.tensor(imgsz[::-1], device=matched_gt.device)
                    
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
            
            if original_shape == kpts_reshaped.shape:
                pred['keypoints'] = kpts_reshaped
            else:
                pred['keypoints'] = kpts_reshaped.reshape(-1, n_kpts * 3)
                
        return preds