#!/usr/bin/env python3
"""
Pipeline inference: detect -> keypoint for BakingRefine.
Pure ONNX Runtime implementation, no third-party ML framework dependencies.

All 15 detection classes are visualized. Only targets meeting pose estimation
conditions (class_id > 4, confidence > 0.6, aspect ratio 0.3-3) undergo
keypoint inference and visualization.
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --- Keypoint visualization config ---
KEYPOINT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 0)]
SKELETON_COLOR = (51, 153, 255)

# 15 classes, pick distinct colors
CLASS_COLORS = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255), (255, 0, 128), (0, 255, 128),
    (128, 255, 128), (255, 128, 128), (128, 128, 255),
]


# ---------------------------------------------------------------------------
# Detection (YOLOv8) helpers
# ---------------------------------------------------------------------------

def det_preprocess(img, input_width, input_height):
    """
    Resize and pad image, keeping aspect ratio.
    Places resized image on a 114-filled canvas (same approach as test_onnx.py).
    Returns: input_tensor (1, 3, H, W) float32 normalized to [0,1], scale, pad_w, pad_h.
    """
    orig_h, orig_w = img.shape[:2]

    scale = min(input_width / orig_w, input_height / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    pad_w = (input_width - new_w) // 2
    pad_h = (input_height - new_h) // 2

    input_img = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    input_img[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    input_tensor = input_img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor, scale, pad_w, pad_h


def nms(boxes, scores, iou_threshold):
    """Pure numpy NMS."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def det_postprocess(output, scale, pad_w, pad_h, orig_shape, target_w=0, target_h=0, conf_thres=0.25):
    """
    Decode detection ONNX output (model already has NMS built-in).

    Supports two formats:
    1. Old format: output is [1, N, 6] or [N, 6] where columns are [x1, y1, x2, y2, conf, cls_id]
    2. RF-DETR format: output is tuple (dets, labels) where:
       - dets: [1, 300, 4] in normalized [cx, cy, w, h] format
       - labels: [1, 300, num_classes] raw logits

    Returns: [(x1, y1, x2, y2, conf, cls_id), ...] in original image coords.
    """
    if len(output) == 2:
        dets, labels = output
        dets = dets.reshape(-1, 4)
        labels = labels.reshape(-1, labels.shape[-1])

        confs = sigmoid(labels.max(axis=-1))
        cls_ids = labels.argmax(axis=-1).astype(int)

        mask = confs > conf_thres
        dets = dets[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        if len(dets) == 0:
            return []

        cx = dets[:, 0]
        cy = dets[:, 1]
        w = dets[:, 2]
        h = dets[:, 3]

        x1_norm = cx - w * 0.5
        y1_norm = cy - h * 0.5
        x2_norm = cx + w * 0.5
        y2_norm = cy + h * 0.5

        x1 = x1_norm * orig_shape[1]
        y1 = y1_norm * orig_shape[0]
        x2 = x2_norm * orig_shape[1]
        y2 = y2_norm * orig_shape[0]

        x1 = x1.clip(0, orig_shape[1] - 1)
        y1 = y1.clip(0, orig_shape[0] - 1)
        x2 = x2.clip(0, orig_shape[1] - 1)
        y2 = y2.clip(0, orig_shape[0] - 1)

        results = []
        for i in range(len(x1)):
            results.append((
                float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(confs[i]), int(cls_ids[i])
            ))
        return results
    else:
        output = output[0].reshape(-1, 6)  # [N, 6]

        if len(output) == 0:
            return []

        x1 = output[:, 0]
        y1 = output[:, 1]
        x2 = output[:, 2]
        y2 = output[:, 3]
        confs = output[:, 4]
        cls_ids = output[:, 5].astype(int)

        mask = confs > conf_thres
        x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        if len(x1) == 0:
            return []

        # Map back to original image coords (same as test_onnx.py):
        # orig_coord = (coord - pad) / scale
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        x1 = x1.clip(0, orig_shape[1]-1)
        y1 = y1.clip(0, orig_shape[0]-1)
        x2 = x2.clip(0, orig_shape[1]-1)
        y2 = y2.clip(0, orig_shape[0]-1)

        results = []
        for i in range(len(x1)):
            results.append((float(x1[i]), float(y1[i]),
                            float(x2[i]), float(y2[i]),
                            float(confs[i]), int(cls_ids[i])))
        return results

# ---------------------------------------------------------------------------
# Keypoint helpers (no mmpose dependency)
# ---------------------------------------------------------------------------

def _fix_aspect_ratio(bbox_scale, aspect_ratio):
    w, h = bbox_scale
    if w > h * aspect_ratio:
        return np.array([w, w / aspect_ratio])
    return np.array([h * aspect_ratio, h])


def _get_3rd_point(a, b):
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def _get_warp_matrix(center, scale, rot, output_size):
    w, h = output_size
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + np.array([
        scale[0] * -0.5 * np.sin(np.deg2rad(rot)),
        scale[0] * -0.5 * np.cos(np.deg2rad(rot)),
    ])
    dst[0, :] = [w * 0.5, h * 0.5]
    dst[1, :] = [w * 0.5, h * 0.5 - w * 0.5]
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(src, dst)


def kpt_preprocess(img, center, scale, input_size):
    w, h = input_size[1], input_size[0]
    crop_w, crop_h = scale[0], scale[1]
    x1 = int(center[0] - crop_w / 2)
    y1 = int(center[1] - crop_h / 2)
    x2 = int(center[0] + crop_w / 2)
    y2 = int(center[1] + crop_h / 2)
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(img.shape[1], x2)
    y2_clip = min(img.shape[0], y2)
    crop = img[y1_clip:y2_clip, x1_clip:x2_clip]
    actual_h, actual_w = crop.shape[:2]
    longer = max(actual_w, actual_h)
    shorter = min(actual_w, actual_h)
    scale_ratio = longer / shorter
    if actual_w >= actual_h:
        new_w, new_h = longer, int(longer / scale_ratio)
    else:
        new_w, new_h = int(longer / scale_ratio), longer
    crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_warped = np.zeros((longer, longer, 3), dtype=np.uint8)
    pad_x = (longer - new_w) // 2
    pad_y = (longer - new_h) // 2
    img_warped[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = crop_resized
    img_warped = cv2.resize(img_warped, (w, h), interpolation=cv2.INTER_LINEAR)
    scale_fixed = np.array([longer, longer], dtype=np.float32)
    warp_mat = np.eye(2, 3, dtype=np.float32)
    img_chw = img_warped.transpose(2, 0, 1)
    return np.expand_dims(img_chw, axis=0), warp_mat, scale_fixed, img_warped


def decode_simcc(simcc_x, simcc_y, simcc_split_ratio=2.0):
    if simcc_x.ndim == 3:
        simcc_x = simcc_x  # already has batch dimension
    elif simcc_x.ndim == 2:
        simcc_x = simcc_x[np.newaxis, ...]  # add batch dimension
    if simcc_y.ndim == 3:
        simcc_y = simcc_y
    elif simcc_y.ndim == 2:
        simcc_y = simcc_y[np.newaxis, ...]
    x_locs = simcc_x.argmax(axis=-1)
    y_locs = simcc_y.argmax(axis=-1)
    x_coords = x_locs / simcc_split_ratio
    y_coords = y_locs / simcc_split_ratio
    x_conf = simcc_x.max(axis=-1)
    y_conf = simcc_y.max(axis=-1)
    scores = np.stack([x_conf, y_conf], axis=-1).mean(axis=-1)
    return x_coords[0], y_coords[0], scores[0]


def keypoints_to_original(kpts, input_size, center, scale):
    return kpts / np.array(input_size[::-1]) * scale + center - 0.5 * scale


def draw_keypoints(img, kpts, scores, thr=0.3, cls_id=0):
    skeleton_color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
    for a, b in SKELETON:
        # if scores[a] < thr or scores[b] < thr:
        #     continue
        x1, y1 = int(round(kpts[a, 0])), int(round(kpts[a, 1]))
        x2, y2 = int(round(kpts[b, 0])), int(round(kpts[b, 1]))
        cv2.line(img, (x1, y1), (x2, y2), skeleton_color, 2)
    for k in range(len(kpts)):
        # if scores[k] < thr:
        #     continue
        x, y = int(round(kpts[k, 0])), int(round(kpts[k, 1]))
        cv2.circle(img, (x, y), 3, KEYPOINT_COLORS[k], -1)
        cv2.putText(img, f'{k}:{scores[k]:.2f}', (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, KEYPOINT_COLORS[k], 1)


def bbox_xyxy2cs(bbox_xyxy, padding=1.0):
    x1, y1, x2, y2 = bbox_xyxy
    w, h = x2 - x1, y2 - y1
    center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)
    scale = np.array([w, h], dtype=np.float32) * padding
    return center, scale


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def expand_bbox(x1, y1, x2, y2, expand_ratio, img_w, img_h):
    w, h = x2 - x1, y2 - y1
    dw, dh = w * expand_ratio, h * expand_ratio
    return (max(0, int(x1 - dw)), max(0, int(y1 - dh)),
            min(img_w, int(x2 + dw)), min(img_h, int(y2 + dh)))


def draw_detection(img, x1, y1, x2, y2, cls_id, conf):
    color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f'cls{cls_id}:{conf:.2f}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Pipeline: detection + keypoint inference')
    parser.add_argument('--det-onnx', type=str,
                        default='runs/detect/ckpts/BakingRecognize/202608072206/weights/best.onnx')
    parser.add_argument('--kpt-onnx', type=str,
                        default='ckpts/rtmpose/BakingRefine/202608101114/best_coco_AP_epoch_100.onnx')
    parser.add_argument('--input-dir', type=str, default='/home/wenjie/Downloads/test')
    parser.add_argument('--output-dir', type=str, default='runs/test')
    parser.add_argument('--det-imgsz', type=int, nargs=2, default=[384, 640],
                        help='Detection model input size (W H)')
    parser.add_argument('--kpt-input-size', type=int, nargs=2, default=[192, 192])
    parser.add_argument('--simcc-split-ratio', type=float, default=2.0)
    parser.add_argument('--det-conf', type=float, default=0.3,
                        help='Confidence threshold for pose filtering')
    parser.add_argument('--det-nms-conf', type=float, default=0.25,
                        help='Confidence threshold for detection NMS')
    parser.add_argument('--det-nms-iou', type=float, default=0.45,
                        help='IoU threshold for detection NMS')
    parser.add_argument('--det-cls-min', type=int, default=4,
                        help='Minimum class id for pose estimation')
    parser.add_argument('--ar-min', type=float, default=0.1)
    parser.add_argument('--ar-max', type=float, default=10.0)
    parser.add_argument('--expand-ratio', type=float, default=0.25)
    parser.add_argument('--kpt-conf', type=float, default=0.3)
    parser.add_argument('--flip', action='store_true', help='horizontal flip augmentation')
    args = parser.parse_args()

    print(f'Loading detection model: {args.det_onnx}')
    det_session = ort.InferenceSession(
        args.det_onnx,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    det_input_name = det_session.get_inputs()[0].name
    det_input_h, det_input_w = args.det_imgsz[0], args.det_imgsz[1]

    print(f'Loading keypoint model: {args.kpt_onnx}')
    kpt_session = ort.InferenceSession(
        args.kpt_onnx,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    kpt_input_size = tuple(args.kpt_input_size)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    img_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in img_exts])

    if not img_paths:
        print(f'No images found in {input_dir}')
        return

    print(f'Found {len(img_paths)} images\n')

    for img_path in img_paths:
        print(f'Processing: {img_path.name}')

        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            print(f'  Skipping, cannot read image')
            continue

        process_img(img_orig, img_path, det_session, det_input_name,
                    det_input_w, det_input_h, kpt_session, kpt_input_size,
                    args, output_dir, False)
        if args.flip:
            img_flip = cv2.flip(img_orig, 1)
            process_img(img_flip, img_path, det_session, det_input_name,
                        det_input_w, det_input_h, kpt_session, kpt_input_size,
                        args, output_dir, True)
    print(f'\nDone. {len(img_paths)} images -> {output_dir}/')


def process_img(img, img_path, det_session, det_input_name,
                det_input_w, det_input_h, kpt_session, kpt_input_size,
                args, output_dir, is_flip):
    img_h, img_w = img.shape[:2]
    suffix = '_flip' if is_flip else ''

    det_input, scale, pad_w, pad_h = det_preprocess(
        img, det_input_w, det_input_h)
    det_output = det_session.run(None, {det_input_name: det_input})

    detections = det_postprocess(
        det_output, scale, pad_w, pad_h, (img_h, img_w),
        det_input_w, det_input_h,
        conf_thres=args.det_nms_conf)

    print(f'  Detections: {len(detections)}')

    vis = img.copy()

    pose_targets = []
    for x1, y1, x2, y2, conf, cls_id in detections:
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        draw_detection(vis, ix1, iy1, ix2, iy2, cls_id, conf)
        # if cls_id not in [5,6,7]:
        #     continue
        w, h = x2 - x1, y2 - y1
        ar = w / h if h > 0 else float('inf')
        if (cls_id > args.det_cls_min and conf >= args.det_conf
                and args.ar_min <= ar <= args.ar_max):
            pose_targets.append((x1, y1, x2, y2, cls_id, conf))

    print(f'  Pose targets: {len(pose_targets)}')

    for i, (x1, y1, x2, y2, cls_id, conf) in enumerate(pose_targets):
        ex1, ey1, ex2, ey2 = expand_bbox(
            x1, y1, x2, y2, args.expand_ratio, img_w, img_h)

        # iex1, iey1, iex2, iey2 = int(round(ex1)), int(round(ey1)), int(round(ex2)), int(round(ey2))
        # draw_detection(vis, iex1, iey1, iex2, iey2, cls_id, conf)

        bbox_xyxy = np.array([ex1, ey1, ex2, ey2], dtype=np.float32)
        center, scale = bbox_xyxy2cs(bbox_xyxy, padding=1.0)

        input_tensor, _, scale_fixed, img_warped = kpt_preprocess(
            img, center, scale, kpt_input_size)

        # warp_save_path = output_dir / f'{img_path.stem}_warp{i}{img_path.suffix}'
        # cv2.imwrite(str(warp_save_path), img_warped)

        ort_out = kpt_session.run(
            None, {'input': input_tensor.astype(np.float32)})
        simcc_x, simcc_y = ort_out[0], ort_out[1]
        x_coords, y_coords, scores = decode_simcc(
            simcc_x[0], simcc_y[0], args.simcc_split_ratio)
        kpts_model = np.stack([x_coords, y_coords], axis=-1)

        kpts_orig = keypoints_to_original(
            kpts_model, kpt_input_size, center, scale_fixed)

        draw_keypoints(vis, kpts_orig, scores, args.kpt_conf, cls_id)

    save_path = output_dir / f'{img_path.stem}{suffix}{img_path.suffix}'
    cv2.imwrite(str(save_path), vis)
    print(f'  Saved: {save_path}')


if __name__ == '__main__':
    main()