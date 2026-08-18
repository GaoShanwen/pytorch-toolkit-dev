"""
Compare PTH vs ONNX outputs with correctly-matched preprocessing.
Run: python tools/compare_pth_onnx.py --img <path> [--pth <pth>] [--onnx <onnx>]
"""
import argparse
import io
import logging
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

from rfdetr import RFDETR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img", required=True)
    p.add_argument("--pth", default="ckpts/detect/BakingRecognizeCOCO/202608141803/checkpoint_best_total.pth")
    p.add_argument("--onnx", default="exports/rfdetr-nano.onnx")
    p.add_argument("--conf", type=float, default=0.3)
    return p.parse_args()


def suppress_logs():
    old = logging.root.handlers[:]
    logging.root.handlers = [logging.NullHandler()]
    logging.root.setLevel(logging.CRITICAL)
    return old


def restore_logs(old):
    logging.root.handlers = old


# ── Preprocessing (matches rfdetr RFDETR.predict) ─────────────────────────────
def preprocess_pth_style(image_rgb: np.ndarray, target_h: int, target_w: int):
    """Direct bilinear resize + ImageNet normalize. Matches RFDETR.predict."""
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    # Use np.float32 constant to avoid upcasting to float64
    _inv255 = np.float32(1.0 / 255.0)
    chw = resized.transpose(2, 0, 1).astype(np.float32) * _inv255
    # ImageNet normalization (matches RFDETR.predict: F.normalize with mean/std)
    _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    chw = (chw - _mean) / _std
    return chw[np.newaxis]  # (1, 3, H, W)


def run_onnx_corrected(onnx_path: str, image_rgb: np.ndarray, target_h: int, target_w: int):
    """Run ONNX with PTH-matching preprocessing."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    inp = preprocess_pth_style(image_rgb, target_h, target_w)
    dets, labels = sess.run(None, {inp_name: inp})
    dets, labels = dets[0], labels[0]   # (Q,4), (Q, C+1)
    labels = labels[:, :-1]              # strip no-object column
    scores_all = 1.0 / (1.0 + np.exp(-labels))
    scores = scores_all.max(axis=1)
    cls_ids = scores_all.argmax(axis=1).astype(int)
    return dets, scores, cls_ids


def run_pth(model: RFDETR, image_path: str, threshold: float):
    """PTH inference via model.predict(), returns xyxy in pixel coords."""
    dets_list = model.predict(image_path, threshold=threshold)
    if isinstance(dets_list, (list, tuple)):
        boxes_px = np.array([d[0] for d in dets_list])
        confs = np.array([d[2] for d in dets_list])
        cls_ids = np.array([d[3] for d in dets_list])
    else:
        boxes_px = dets_list.xyxy
        confs = dets_list.confidence if dets_list.confidence is not None else np.array([])
        cls_ids = dets_list.class_id if dets_list.class_id is not None else np.array([], dtype=int)
    return boxes_px, confs, cls_ids


def main():
    args = parse_args()

    # Load PTH model
    old = suppress_logs()
    old_out = sys.stdout
    sys.stdout = io.StringIO()
    model = RFDETR.from_checkpoint(args.pth, trust_checkpoint=True)
    sys.stdout = old_out
    restore_logs(old)

    target_h = model.model.resolution
    target_w = target_h
    print(f"PTH resolution: {target_h}x{target_w}")

    # Check ONNX shape
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_shape = sess.get_inputs()[0].shape
    print(f"ONNX input shape: {onnx_shape}")
    if onnx_shape[2] != target_h or onnx_shape[3] != target_w:
        print(f"WARNING: ONNX shape {onnx_shape[2:]} != PTH resolution {target_h}x{target_w}")
        target_h, target_w = int(onnx_shape[2]), int(onnx_shape[3])

    # Load image
    img_bgr = cv2.imread(args.img)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_bgr.shape[:2]
    print(f"Image: {orig_h}x{orig_w}")

    # ONNX inference
    dets, scores_onnx, cls_onnx = run_onnx_corrected(args.onnx, img_rgb, target_h, target_w)
    mask = scores_onnx > args.conf
    dets_onnx = dets[mask]
    scores_onnx = scores_onnx[mask]
    cls_onnx = cls_onnx[mask]
    print(f"\nONNX detections (conf>{args.conf}): {len(scores_onnx)}")

    # PTH inference
    boxes_px, scores_pth, cls_pth = run_pth(model, args.img, args.conf)
    print(f"PTH  detections (conf>{args.conf}): {len(scores_pth)}")

    # Convert ONNX dets (norm cxcywh → pixel xyxy) using same logic as PTH
    cx, cy, bw, bh = dets_onnx[:, 0], dets_onnx[:, 1], dets_onnx[:, 2], dets_onnx[:, 3]
    xyxy_onnx = np.stack([
        (cx - bw * 0.5) * orig_w,
        (cy - bh * 0.5) * orig_h,
        (cx + bw * 0.5) * orig_w,
        (cy + bh * 0.5) * orig_h,
    ], axis=1)

    print(f"\n{'='*60}")
    print(f"{'DETAIL COMPARISON':^60}")
    print(f"{'='*60}")
    print(f"{'#':<4} {'ONNX (pixel xyxy)':<30} {'PTH (pixel xyxy)':<30}")
    print(f"{'='*60}")

    # IoU-based matching
    def compute_iou(a, b):
        """Compute IoU between one box and an array of boxes. Both in xyxy format."""
        a = np.asarray(a, dtype=np.float64).flatten()
        b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
        xi1 = np.maximum(a[0], b[:, 0])
        yi1 = np.maximum(a[1], b[:, 1])
        xi2 = np.minimum(a[2], b[:, 2])
        yi2 = np.minimum(a[3], b[:, 3])
        inter = np.maximum(0.0, xi2 - xi1) * np.maximum(0.0, yi2 - yi1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a + area_b - inter
        ious = inter / (union + 1e-9)
        return float(ious) if ious.ndim == 0 else ious

    matched = []
    pth_used = np.zeros(len(boxes_px), dtype=bool)
    onnx_used = np.zeros(len(xyxy_onnx), dtype=bool)

    for i in range(len(xyxy_onnx)):
        best_iou, best_j = 0, -1
        for j in range(len(boxes_px)):
            if pth_used[j]:
                continue
            iou = compute_iou(xyxy_onnx[i], boxes_px[j:j+1])[0]
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= 0.5:
            pth_used[best_j] = True
            onnx_used[i] = True
            matched.append((i, best_j, best_iou))
            oc, pc = scores_onnx[i], scores_pth[best_j]
            print(f"  [{i}] ONNX cls={cls_onnx[i]} c={oc:.4f}  "
                  f"PTH cls={cls_pth[best_j]} c={pc:.4f}  "
                  f"IoU={best_iou:.3f}  Δc={abs(oc-pc):.4f}")

    print(f"\nMatched: {len(matched)} / ONNX:{len(scores_onnx)}  PTH:{len(scores_pth)}")
    if matched:
        avg_dc = sum(abs(scores_onnx[i]-scores_pth[j]) for i, j, _ in matched) / len(matched)
        print(f"Avg |Δconf| for matched: {avg_dc:.4f}")

    onnx_unmatched = np.where(~onnx_used)[0]
    pth_unmatched = np.where(~pth_used)[0]
    if len(onnx_unmatched):
        print(f"\nONNX unique ({len(onnx_unmatched)}):")
        for i in onnx_unmatched:
            print(f"  cls={cls_onnx[i]} c={scores_onnx[i]:.4f}  box={xyxy_onnx[i].round(1)}")
    if len(pth_unmatched):
        print(f"\nPTH unique ({len(pth_unmatched)}):")
        for j in pth_unmatched:
            print(f"  cls={cls_pth[j]} c={scores_pth[j]:.4f}  box={boxes_px[j].round(1)}")


if __name__ == "__main__":
    main()
