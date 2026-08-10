"""
Evaluate ONNX model with test images from annotation file.

Usage:
    python tools/convert/eval_onnx.py \
        --onnx ckpts/rtmpose/BakingRefine/202608091326/best_coco_AP_epoch_90.onnx \
        --ann data/pose-dataset/BakingRefine/annotations/val260807.json \
        --img-dir data/pose-dataset/BakingRefine/images \
        --out-dir work_dirs/onnx \
        --num 8
"""

import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from mmpose.structures.bbox import bbox_xyxy2cs, get_warp_matrix


KEYPOINT_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 0)]
SKELETON_COLOR = (51, 153, 255)


def _fix_aspect_ratio(bbox_scale, aspect_ratio):
    w, h = bbox_scale
    if w > h * aspect_ratio:
        bbox_scale = np.array([w, w / aspect_ratio])
    else:
        bbox_scale = np.array([h * aspect_ratio, h])
    return bbox_scale


def preprocess(img, center, scale, input_size):
    w, h = input_size[1], input_size[0]
    scale_fixed = _fix_aspect_ratio(scale, aspect_ratio=w / h)
    warp_mat = get_warp_matrix(center, scale_fixed, 0.0, output_size=(w, h))
    img_warped = cv2.warpAffine(img, warp_mat, (w, h), flags=cv2.INTER_LINEAR)
    img_chw = img_warped.transpose(2, 0, 1)
    return np.expand_dims(img_chw, axis=0), warp_mat, scale_fixed


def decode_simcc(simcc_x, simcc_y, simcc_split_ratio=2.0):
    if simcc_x.ndim == 2:
        simcc_x = simcc_x[np.newaxis, ...]
    if simcc_y.ndim == 2:
        simcc_y = simcc_y[np.newaxis, ...]

    x_locs = simcc_x.argmax(axis=-1)
    y_locs = simcc_y.argmax(axis=-1)
    x_coords = x_locs / simcc_split_ratio
    y_coords = y_locs / simcc_split_ratio

    x_conf = simcc_x.max(axis=-1)
    y_conf = simcc_y.max(axis=-1)
    scores = np.stack([x_conf, y_conf], axis=-1).mean(axis=-1)

    return x_coords, y_coords, scores


def keypoints_to_original(kpts, input_size, center, scale):
    kpts = kpts / np.array(input_size[::-1]) * scale + center - 0.5 * scale
    return kpts


def draw_keypoints(img, kpts, scores, thr=0.3):
    for k in range(len(kpts)):
        # if scores[k] < thr:
        #     continue
        x, y = int(round(kpts[k, 0])), int(round(kpts[k, 1]))
        color = KEYPOINT_COLORS[k]
        cv2.circle(img, (x, y), 3, color, -1)
        cv2.putText(img, f'{k}:{scores[k]:.2f}', (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    for a, b in SKELETON:
        # if scores[a] < thr or scores[b] < thr:
        #     continue
        x1, y1 = int(round(kpts[a, 0])), int(round(kpts[a, 1]))
        x2, y2 = int(round(kpts[b, 0])), int(round(kpts[b, 1]))
        cv2.line(img, (x1, y1), (x2, y2), SKELETON_COLOR, 2)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ONNX RTMPose model')
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--ann', type=str, required=True, help='Path to COCO annotation JSON')
    parser.add_argument('--img-dir', type=str, default='data/pose-dataset/BakingRefine/images', help='Image directory')
    parser.add_argument('--out-dir', type=str, default='work_dirs/onnx', help='Output directory')
    parser.add_argument('--num', type=int, default=8, help='Number of images to test')
    parser.add_argument('--input-size', type=int, nargs=2, default=[192, 192],
                        help='Model input size (H W)')
    parser.add_argument('--simcc-split-ratio', type=float, default=2.0)
    parser.add_argument('--conf-thr', type=float, default=0.0,
                        help='Confidence threshold for visualization')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.ann) as f:
        coco = json.load(f)

    images = coco['images'][:args.num]
    img_id_to_ann = {ann['image_id']: ann for ann in coco['annotations']}

    os.makedirs(args.out_dir, exist_ok=True)
    session = ort.InferenceSession(args.onnx)

    input_size = tuple(args.input_size)

    for img_info in images:
        img_path = os.path.join(args.img_dir, img_info['file_name'])
        img_orig = cv2.imread(img_path)
        if img_orig is None:
            print(f'Skipping missing image: {img_path}')
            continue

        ann = img_id_to_ann.get(img_info['id'])
        if ann is None:
            print(f'Skipping image without annotation: {img_info["file_name"]}')
            continue

        bbox_xywh = ann['bbox']
        x, y, w, h = bbox_xywh
        bbox_xyxy = np.array([x, y, x + w, y + h])
        center, scale = bbox_xyxy2cs(bbox_xyxy, padding=1.25)

        input_tensor, warp_mat, scale_fixed = preprocess(
            img_orig, center, scale, input_size)

        ort_out = session.run(None, {'input': input_tensor.astype(np.float32)})
        simcc_x, simcc_y = ort_out[0], ort_out[1]

        x_coords, y_coords, scores = decode_simcc(
            simcc_x[0], simcc_y[0], args.simcc_split_ratio)

        kpts_model = np.stack([x_coords[0], y_coords[0]], axis=-1)
        kpts_orig = keypoints_to_original(kpts_model, input_size, center, scale_fixed)

        vis = img_orig.copy()
        draw_keypoints(vis, kpts_orig, scores[0], args.conf_thr)

        save_path = os.path.join(args.out_dir, img_info['file_name'])
        cv2.imwrite(save_path, vis)
        print(f'Visualized: {save_path}')

    print(f'\nDone. {len(images)} images saved to {args.out_dir}/')


if __name__ == '__main__':
    main()