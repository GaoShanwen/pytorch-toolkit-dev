import cv2
import random
import numpy as np
from copy import deepcopy
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.data.augment import (
    Compose, Mosaic, CopyPaste, RandomPerspective, LetterBox, LOGGER, MixUp, Albumentations, RandomHSV, RandomFlip
)

class WithIgnoreMosaic(Mosaic):
    def _cat_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        iscrowds = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
            iscrowds.append(labels["iscrowds"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "iscrowds": np.concatenate(iscrowds, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        final_labels["iscrowds"] = final_labels["iscrowds"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels

class WithIgnoreCopyPaste(CopyPaste):
    def __call__(self, labels):
        """
        Applies the Copy-Paste augmentation to the given image and instances.

        Args:
            labels (dict): A dictionary containing:
                           - 'img': The image to augment.
                           - 'cls': Class labels associated with the instances.
                           - 'instances': Object containing bounding boxes, and optionally, keypoints and segments.

        Returns:
            (dict): Dict with augmented image and updated instances under the 'img', 'cls', and 'instances' keys.

        Notes:
            1. Instances are expected to have 'segments' as one of their attributes for this augmentation to work.
            2. This method modifies the input dictionary 'labels' in place.
        """
        im = labels["img"]
        cls = labels["cls"]
        iscrowds = labels["iscrowds"]

        h, w = im.shape[:2]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)
        if self.p and len(instances.segments):
            n = len(instances)
            _, w, _ = im.shape  # height, width, channels
            im_new = np.zeros(im.shape, np.uint8)

            # Calculate ioa first then select indexes randomly
            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)

            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)  # intersection over area, (N, M)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                iscrowds = np.concatenate((iscrowds, iscrowds[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                cv2.drawContours(im_new, instances.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

            result = cv2.flip(im, 1)  # augment segments (flip left-right)
            i = cv2.flip(im_new, 1).astype(bool)
            im[i] = result[i]

        labels["img"] = im
        labels["cls"] = cls
        labels["iscrowds"] = iscrowds
        labels["instances"] = instances
        return labels

class WithIgnorePerspect(RandomPerspective):
    def __call__(self, labels):
        """
        Affine images and targets.

        Args:
            labels (dict): a dict of `bboxes`, `segments`, `keypoints`.
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad

        img = labels["img"]
        cls = labels["cls"]
        iscrowds = labels["iscrowds"]
        instances = labels.pop("instances")
        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["iscrowds"] = iscrowds[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

class WithIgnoreMixUp(MixUp):
    """Class for applying MixUp augmentation to the dataset."""

    def _mix_transform(self, labels):
        """Applies MixUp augmentation as per https://arxiv.org/pdf/1710.09412.pdf."""
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        labels["iscrowds"] = np.concatenate([labels["iscrowds"], labels2["iscrowds"]], 0)
        return labels

class WithIgnoreAlbument(Albumentations):
    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels["img"]
        cls = labels["cls"]
        iscrowds = labels["iscrowds"]
        if len(cls):
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*im.shape[:2][::-1])
            bboxes = labels["instances"].bboxes
            # TODO: add supports of segments and keypoints
            if self.transform and random.random() < self.p:
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls, iscorwds=iscrowds)  # transformed
                if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    labels["iscrowds"] = np.array(new["iscrowds"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
            labels["instances"].update(bboxes=bboxes)
        return labels

def v8_ignore_trans(dataset, imgsz, hyp, stretch=False):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose(
        [
            WithIgnoreMosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
            WithIgnoreCopyPaste(p=hyp.copy_paste),
            WithIgnorePerspect(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
            ),
        ]
    )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            WithIgnoreMixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            WithIgnoreAlbument(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms

