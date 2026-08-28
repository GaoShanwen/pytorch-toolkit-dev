"""
Visualization utilities for RF-DETR validation results.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import supervision as sv
from supervision import box_iou_batch

COLOR_FN = sv.Color(203, 192, 255)
COLOR_FP = sv.Color(0, 0, 139)
COLOR_GT = sv.Color(255, 255, 0)
COLOR_PRED = sv.Color(255, 0, 0)


def _compute_matching(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_threshold: float = 0.5,
    class_mapping: dict[int, int] = None,
):
    """Compute TP, FP, FN for each class.

    Args:
        class_mapping: Optional dict mapping original class IDs to remapped IDs.
            If provided, adjusts label comparison accordingly.
    """
    results = {
        "tp_boxes": [],
        "tp_labels": [],
        "tp_scores": [],
        "fp_boxes": [],
        "fp_labels": [],
        "fp_scores": [],
        "fn_boxes": [],
        "fn_labels": [],
    }

    if class_mapping is None:
        class_mapping = {}

    reverse_mapping = {v: k for k, v in class_mapping.items()}

    if len(gt_boxes) == 0:
        for i in range(len(pred_boxes)):
            results["fp_boxes"].append(pred_boxes[i].numpy())
            results["fp_labels"].append(pred_labels[i].item())
            results["fp_scores"].append(pred_scores[i].item())
        return results

    if len(pred_boxes) == 0:
        for i in range(len(gt_boxes)):
            results["fn_boxes"].append(gt_boxes[i].numpy())
            results["fn_labels"].append(gt_labels[i].item())
        return results

    gt_labels_np = gt_labels.numpy()
    pred_labels_np = pred_labels.numpy()
    pred_scores_np = pred_scores.numpy()
    pred_boxes_np = pred_boxes.numpy()
    gt_boxes_np = gt_boxes.numpy()

    iou_matrix = box_iou_batch(
        boxes_true=gt_boxes_np,
        boxes_detection=pred_boxes_np,
    )

    matched_gt = set()
    matched_pred = set()

    def labels_match(pred_label, gt_label):
        if pred_label == gt_label:
            return True
        if gt_label in reverse_mapping and reverse_mapping[gt_label] == pred_label:
            return True
        if pred_label in class_mapping and class_mapping[pred_label] == gt_label:
            return True
        return False

    for pred_idx in np.argsort(-pred_scores_np):
        if pred_idx in matched_pred:
            continue
        pred_label = pred_labels_np[pred_idx]
        for gt_idx in np.argsort(-iou_matrix[:, pred_idx]):
            if gt_idx in matched_gt:
                continue
            gt_label = gt_labels_np[gt_idx]
            if not labels_match(pred_label, gt_label):
                continue
            iou = iou_matrix[gt_idx, pred_idx]
            if iou >= iou_threshold:
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
                results["tp_boxes"].append(pred_boxes_np[pred_idx])
                results["tp_labels"].append(pred_label)
                results["tp_scores"].append(pred_scores_np[pred_idx])
                break

    for pred_idx in range(len(pred_boxes)):
        if pred_idx not in matched_pred:
            results["fp_boxes"].append(pred_boxes_np[pred_idx])
            results["fp_labels"].append(pred_labels_np[pred_idx])
            results["fp_scores"].append(pred_scores_np[pred_idx])

    for gt_idx in range(len(gt_boxes)):
        if gt_idx not in matched_gt:
            results["fn_boxes"].append(gt_boxes_np[gt_idx])
            results["fn_labels"].append(gt_labels_np[gt_idx])

    return results


def save_error_analysis_visualizations(
    model: "RFDETR",
    image_paths: list[str],
    gt_dataset: "torch.utils.data.Dataset",
    output_dir: Path,
    threshold: float,
    iou_threshold: float = 0.5,
    class_mapping: dict[int, int] = None,
) -> dict[str, int]:
    """Save error analysis visualizations with gt, pred, fp, fn in 2x2 grid.

    Args:
        model: RFDETR model instance.
        image_paths: List of image paths to visualize.
        gt_dataset: Dataset that provides ground truth for each image.
        output_dir: Base output directory.
        threshold: Detection confidence threshold.
        iou_threshold: IoU threshold for matching predictions to gt.
        class_mapping: Optional class mapping dict for label matching.

    Returns:
        dict with counts: {total, only_fp, only_fn, both_fp_fn, no_errors}
    """
    class_names = getattr(model.model, "class_names", None) or []

    error_dir = output_dir / "error"
    fp_only_dir = output_dir / "fp"
    fn_only_dir = output_dir / "fn"

    error_dir.mkdir(parents=True, exist_ok=True)
    fp_only_dir.mkdir(parents=True, exist_ok=True)
    fn_only_dir.mkdir(parents=True, exist_ok=True)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)

    stats = {"total": len(image_paths), "only_fp": 0, "only_fn": 0, "both_fp_fn": 0, "no_errors": 0}

    id_to_idx = {}
    filename_to_idx = {}
    for idx in range(len(gt_dataset)):
        _, target = gt_dataset[idx]
        img_id = target.get("image_id")
        if img_id is not None:
            img_id_val = img_id.item() if torch.is_tensor(img_id) else img_id
            id_to_idx[img_id_val] = idx
            if hasattr(gt_dataset, 'coco'):
                file_name = gt_dataset.coco.loadImgs(img_id_val)[0].get('file_name')
                if file_name:
                    filename_to_idx[file_name] = idx
                    filename_to_idx[Path(file_name).stem] = idx

    for img_idx, image_path in enumerate(image_paths):
        img_path = Path(image_path)
        img_stem = img_path.stem

        gt_idx = filename_to_idx.get(img_path.name) or filename_to_idx.get(img_stem)

        if gt_idx is None:
            for fn, idx in filename_to_idx.items():
                if img_stem in fn:
                    gt_idx = idx
                    break

        if gt_idx is None:
            print(f"[WARN] No GT found for {img_path.name}")
            continue

        detections = model.predict(image_path, threshold=threshold)
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]

        _, target = gt_dataset[gt_idx]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        if class_mapping:
            gt_labels_np = gt_labels.numpy()
            mapped_labels = np.array([class_mapping.get(int(l), int(l)) for l in gt_labels_np])
            gt_labels = torch.from_numpy(mapped_labels)

        gt_boxes_np = gt_boxes.numpy() if torch.is_tensor(gt_boxes) else gt_boxes

        orig_size = target.get("orig_size")
        if orig_size is not None:
            orig_h, orig_w = orig_size[0].item() if torch.is_tensor(orig_size) else orig_size[0], \
                             orig_size[1].item() if torch.is_tensor(orig_size) else orig_size[1]
        else:
            orig_h, orig_w = h, w

        if len(gt_boxes_np) > 0:
            cx, cy, bw, bh = gt_boxes_np[:, 0], gt_boxes_np[:, 1], gt_boxes_np[:, 2], gt_boxes_np[:, 3]
            x_min = (cx - bw / 2) * orig_w
            y_min = (cy - bh / 2) * orig_h
            x_max = (cx + bw / 2) * orig_w
            y_max = (cy + bh / 2) * orig_h
            gt_boxes_xyxy = np.stack([x_min, y_min, x_max, y_max], axis=1)
        else:
            gt_boxes_xyxy = np.empty((0, 4))

        scale_x = w / orig_w
        scale_y = h / orig_h
        gt_boxes_xyxy[:, [0, 2]] *= scale_x
        gt_boxes_xyxy[:, [1, 3]] *= scale_y

        gt_boxes_for_matching = torch.from_numpy(gt_boxes_xyxy)

        pred_boxes = torch.from_numpy(detections.xyxy)
        pred_labels = torch.from_numpy(detections.class_id) if detections.class_id is not None else torch.tensor([], dtype=torch.int64)
        pred_scores = torch.from_numpy(detections.confidence) if detections.confidence is not None else torch.tensor([], dtype=torch.float32)

        matching = _compute_matching(
            gt_boxes_for_matching, gt_labels,
            pred_boxes, pred_labels, pred_scores,
            iou_threshold=iou_threshold,
            class_mapping=class_mapping,
        )

        n_fp = len(matching["fp_boxes"])
        n_fn = len(matching["fn_boxes"])

        if n_fp == 0 and n_fn == 0:
            stats["no_errors"] += 1
            continue

        base_name = img_path.stem
        ext = img_path.suffix

        if n_fp > 0 and n_fn > 0:
            stats["both_fp_fn"] += 1
            save_dir = error_dir
        elif n_fp > 0:
            stats["only_fp"] += 1
            save_dir = fp_only_dir
        else:
            stats["only_fn"] += 1
            save_dir = fn_only_dir

        gt_det = sv.Detections(
            xyxy=np.array(matching["fn_boxes"]) if matching["fn_boxes"] else np.empty((0, 4)),
            class_id=np.array(matching["fn_labels"]) if matching["fn_labels"] else np.array([], dtype=np.int64),
            confidence=np.ones(len(matching["fn_boxes"])) if matching["fn_boxes"] else np.array([]),
        )
        fn_annot = sv.BoxAnnotator(color=COLOR_FN, thickness=2)
        fn_label_annot = sv.LabelAnnotator(text_scale=0.8, text_padding=4, color=COLOR_FN)
        fn_labels = [f"{class_names[l]}" if 0 <= l < len(class_names) else str(l) for l in matching["fn_labels"]]

        fn_det = sv.Detections(
            xyxy=np.array(matching["fn_boxes"]) if matching["fn_boxes"] else np.empty((0, 4)),
            class_id=np.array(matching["fn_labels"]) if matching["fn_labels"] else np.array([], dtype=np.int64),
            confidence=np.ones(len(matching["fn_boxes"])) if matching["fn_boxes"] else np.array([]),
        )

        # pred_tp_det = sv.Detections(
        #     xyxy=np.array(matching["tp_boxes"]) if matching["tp_boxes"] else np.empty((0, 4)),
        #     class_id=np.array(matching["tp_labels"]) if matching["tp_labels"] else np.array([], dtype=np.int64),
        #     confidence=np.array(matching["tp_scores"]) if matching["tp_scores"] else np.array([]),
        # )
        pred_fp_det = sv.Detections(
            xyxy=np.array(matching["fp_boxes"]) if matching["fp_boxes"] else np.empty((0, 4)),
            class_id=np.array(matching["fp_labels"]) if matching["fp_labels"] else np.array([], dtype=np.int64),
            confidence=np.array(matching["fp_scores"]) if matching["fp_scores"] else np.array([]),
        )
        fp_labels = [f"{class_names[l]}:{s:.2f}" if 0 <= l < len(class_names) else f"{l}:{s:.2f}"
                     for l, s in zip(matching["fp_labels"], matching["fp_scores"])]
        fp_annot = sv.BoxAnnotator(color=COLOR_FP, thickness=2)
        fp_label_annot = sv.LabelAnnotator(text_scale=0.8, text_padding=4, color=COLOR_FP)

        gt_boxes_np = gt_boxes.numpy() if torch.is_tensor(gt_boxes) else gt_boxes
        gt_labels_np = gt_labels.numpy() if torch.is_tensor(gt_labels) else gt_labels

        gt_det_all = sv.Detections(
            xyxy=gt_boxes_xyxy,
            class_id=gt_labels_np,
            confidence=np.ones(len(gt_boxes_xyxy)),
        )
        all_gt_labels = [f"{class_names[l]}" if 0 <= l < len(class_names) else str(l) for l in gt_labels_np]
        gt_annot = sv.BoxAnnotator(color=COLOR_GT, thickness=2)
        gt_label_annot = sv.LabelAnnotator(text_scale=0.8, text_padding=4, color=COLOR_GT)

        pred_det_all = sv.Detections(
            xyxy=detections.xyxy,
            class_id=detections.class_id if detections.class_id is not None else np.array([], dtype=np.int64),
            confidence=detections.confidence if detections.confidence is not None else np.array([]),
        )
        all_pred_labels = [f"{class_names[l]}:{s:.2f}" if 0 <= l < len(class_names) else f"{l}:{s:.2f}"
                         for l, s in zip(detections.class_id, detections.confidence)] if detections.class_id is not None else []
        pred_annot = sv.BoxAnnotator(color=COLOR_PRED, thickness=2)
        pred_label_annot = sv.LabelAnnotator(text_scale=0.8, text_padding=4, color=COLOR_PRED)

        gt_img = image.copy()
        if len(gt_det_all) > 0:
            gt_img = gt_annot.annotate(scene=gt_img, detections=gt_det_all)
            gt_img = gt_label_annot.annotate(scene=gt_img, detections=gt_det_all, labels=all_gt_labels)

        pred_img = image.copy()
        if len(pred_det_all) > 0:
            pred_img = pred_annot.annotate(scene=pred_img, detections=pred_det_all)
            pred_img = pred_label_annot.annotate(scene=pred_img, detections=pred_det_all, labels=all_pred_labels)

        fp_img = image.copy()
        if len(pred_fp_det) > 0:
            fp_img = fp_annot.annotate(scene=fp_img, detections=pred_fp_det)
            fp_img = fp_label_annot.annotate(scene=fp_img, detections=pred_fp_det, labels=fp_labels)

        fn_img = image.copy()
        if len(fn_det) > 0:
            fn_img = fn_annot.annotate(scene=fn_img, detections=fn_det)
            fn_img = fn_label_annot.annotate(scene=fn_img, detections=fn_det, labels=fn_labels)

        grid_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        grid_img[0:h, 0:w] = gt_img
        grid_img[0:h, w:w*2] = pred_img
        grid_img[h:h*2, 0:w] = fp_img
        grid_img[h:h*2, w:w*2] = fn_img

        grid_pil = Image.fromarray(grid_img)
        draw = ImageDraw.Draw(grid_pil)
        try:
            fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            fnt = ImageFont.load_default()

        draw.text((10, 10), "GT", fill=(0, 255, 0), font=fnt)
        draw.text((w + 10, 10), "PRED", fill=(255, 255, 255), font=fnt)
        draw.text((10, h + 10), "FP", fill=(255, 0, 0), font=fnt)
        draw.text((w + 10, h + 10), "FN", fill=(255, 0, 0), font=fnt)

        grid_pil.save(save_dir / f"{base_name}{ext}")

    print(f"Error analysis: total={stats['total']}, no_errors={stats['no_errors']}, "
          f"only_fp={stats['only_fp']}, only_fn={stats['only_fn']}, both={stats['both_fp_fn']}")

    return stats


def save_visualizations(
    model: "RFDETR",
    image_paths: list[str],
    output_dir: Path,
    threshold: float,
) -> None:
    vis_dir = output_dir / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)
    class_names = getattr(model.model, "class_names", None) or []

    for image_path in image_paths:
        detections = model.predict(image_path, threshold=threshold)
        image = np.array(Image.open(image_path).convert("RGB"))

        labels = []
        if detections.class_id is not None:
            for class_id in detections.class_id:
                if 0 <= class_id < len(class_names):
                    labels.append(class_names[class_id])
                else:
                    labels.append(str(class_id))

        annotated = box_annotator.annotate(scene=image, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        Image.fromarray(annotated).save(vis_dir / Path(image_path).name)

    print(f"Saved {len(image_paths)} visualization(s) to {vis_dir}")