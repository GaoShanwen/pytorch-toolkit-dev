######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2025.08.28
# filename: val.py
# function: create detect validator for ignore region.
######################################################
import torch
from pathlib import Path

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops

from ...data import build_yolopro_dataset, box_ioa

class WithIgnoreValidator(DetectionValidator):
    def __init__(self, ioav=0.25, **kwargs):
        self.ioav = ioav
        super(WithIgnoreValidator, self).__init__(**kwargs)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLOPro Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolopro_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes", "iscrowds"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        iscrowds = batch["iscrowds"][idx].squeeze(-1)
        return {"cls": cls, "bbox": bbox, "iscrowds": iscrowds, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            
            iscrowds = pbatch.pop("iscrowds").bool() # iscrowd
            bbox_masks, cls, bbox = bbox[iscrowds], cls[~iscrowds], bbox[~iscrowds]
            if len(bbox_masks) != 0:
                pred_keeps = (box_ioa(pred[:, :4].T, bbox_masks) <= self.ioav).all(dim=0)
                pred = pred[pred_keeps]
            
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            nl = len(cls)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
                
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

