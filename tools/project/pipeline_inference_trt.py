#!/usr/bin/env python3
"""
Pipeline inference: detect -> keypoint for BakingRefine.
TensorRT implementation for high-performance inference on NVIDIA GPUs.

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
import tensorrt as trt

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
# TensorRT Runtime Helper
# ---------------------------------------------------------------------------

_cuda_initialized = False
_cuda_context = None


def _init_cuda():
    global _cuda_initialized, _cuda_context
    if _cuda_initialized:
        return True
    try:
        import pycuda.driver as cuda
        cuda.init()
        _cuda_context = cuda.Device(0).make_context()
        _cuda_initialized = True
        return True
    except ImportError:
        return False
    except Exception:
        return False


def _push_cuda_context():
    global _cuda_context
    if _cuda_context:
        _cuda_context.push()


def _pop_cuda_context():
    global _cuda_context
    if _cuda_context:
        _cuda_context.pop()


class TRTEngine:
    def __init__(self, engine_path, batch_size=1):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.batch_size = batch_size

        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()

        self.num_io_tensors = self.engine.num_io_tensors
        self.input_names = []
        self.output_names = []
        for i in range(self.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self.input_shape = self.engine.get_tensor_shape(self.input_names[0])
        self.max_batch_size = self._get_max_batch_size()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []

        for name in self.input_names:
            input_shape = self.engine.get_tensor_shape(name)
            input_shape = tuple(1 if d < 0 else d for d in input_shape)
            self.host_inputs.append(np.zeros(input_shape, dtype=np.float32))
            self.cuda_inputs.append(np.zeros(input_shape, dtype=np.float32))

        for name in self.output_names:
            output_shape = self.engine.get_tensor_shape(name)
            output_shape = tuple(1 if d < 0 else d for d in output_shape)
            self.host_outputs.append(np.zeros(output_shape, dtype=np.float32))
            self.cuda_outputs.append(np.zeros(output_shape, dtype=np.float32))

        self.use_cuda = _init_cuda()

    def allocate_buffers(self):
        if not self.use_cuda:
            return
        _push_cuda_context()
        import pycuda.driver as cuda
        for i in range(len(self.host_inputs)):
            self.cuda_inputs[i] = cuda.mem_alloc(self.host_inputs[i].nbytes)
        for i in range(len(self.host_outputs)):
            self.cuda_outputs[i] = cuda.mem_alloc(self.host_outputs[i].nbytes)
        _pop_cuda_context()

    def _get_max_batch_size(self):
        try:
            profile_shape = self.engine.get_tensor_profile_shape(
                self.input_names[0], 0
            )
            return profile_shape[2][0]
        except Exception:
            return self.input_shape[0] if self.input_shape[0] > 0 else 1

    def run(self, input_tensor):
        if self.use_cuda:
            return self._run_cuda(input_tensor)
        else:
            return self._run_cpu(input_tensor)

    def _allocate_if_needed(self, input_tensor):
        if not self.use_cuda:
            return
        import pycuda.driver as cuda
        current_shape = self.host_inputs[0].shape
        new_shape = input_tensor.shape

        need_realloc = current_shape != new_shape

        if -1 in self.input_shape or need_realloc:
            nbytes = int(np.prod(new_shape)) * 4
            self.host_inputs[0] = np.zeros(new_shape, dtype=np.float32)
            self.cuda_inputs[0] = cuda.mem_alloc(nbytes)

            self.context.set_input_shape(self.input_names[0], new_shape)

            for i, name in enumerate(self.output_names):
                output_shape = self.context.get_tensor_shape(name)
                output_shape = tuple(max(1, d) if d > 0 else 1 for d in output_shape)
                nbytes = int(np.prod(output_shape)) * 4
                self.host_outputs[i] = np.zeros(output_shape, dtype=np.float32)
                self.cuda_outputs[i] = cuda.mem_alloc(nbytes)

    def _run_cuda(self, input_tensor):
        import pycuda.driver as cuda
        _push_cuda_context()
        self._allocate_if_needed(input_tensor)

        self.host_inputs[0] = np.ascontiguousarray(input_tensor)
        cuda.memcpy_htod(self.cuda_inputs[0], self.host_inputs[0])

        bindings = [int(self.cuda_inputs[0])]
        for cuda_output in self.cuda_outputs:
            bindings.append(int(cuda_output))

        self.context.execute_v2(bindings=bindings)
        cuda.Context.synchronize()

        for i in range(len(self.host_outputs)):
            cuda.memcpy_dtoh(self.host_outputs[i], self.cuda_outputs[i])

        _pop_cuda_context()

        return [output.copy() for output in self.host_outputs]

    def _run_cpu(self, input_tensor):
        input_shape = input_tensor.shape
        self.context.set_input_shape(self.input_names[0], input_shape)

        self.host_inputs[0] = input_tensor
        outputs = []
        for i in range(len(self.host_outputs)):
            self.context.execute_v2(
                bindings=[int(self.host_inputs[0]), int(self.host_outputs[i])]
            )
            outputs.append(self.host_outputs[i].copy())
        return outputs


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


def det_postprocess(output, scale, pad_w, pad_h, orig_shape, conf_thres=0.25):
    """
    Decode detection TensorRT output (model already has NMS built-in).

    output: [1, N, 6] or [N, 6] where columns are [x1, y1, x2, y2, conf, cls_id]
    Returns: [(x1, y1, x2, y2, conf, cls_id), ...] in original image coords.
    """
    output = output.reshape(-1, 6)  # [N, 6]

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

    x1 = (x1 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    x2 = (x2 - pad_w) / scale
    y2 = (y2 - pad_h) / scale

    x1 = x1.clip(0, orig_shape[1])
    y1 = y1.clip(0, orig_shape[0])
    x2 = x2.clip(0, orig_shape[1])
    y2 = y2.clip(0, orig_shape[0])

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


def batch_kpt_preprocess(img, roi_list, input_size):
    """
    Batch preprocess ROIs for keypoint model.

    Args:
        img: Original image (H, W, 3) BGR
        roi_list: List of (x1, y1, x2, y2) bounding boxes
        input_size: (W, H) tuple

    Returns:
        batch_tensor: (N, 3, H, W) float32 batch input
        meta_list: List of (center, scale_fixed) for each ROI
    """
    if not roi_list:
        return None, []

    w, h = input_size
    batch_tensors = []
    meta_list = []

    for x1, y1, x2, y2 in roi_list:
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        scale = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        crop_w, crop_h = scale[0], scale[1]
        cx1 = int(center[0] - crop_w / 2)
        cy1 = int(center[1] - crop_h / 2)
        cx2 = int(center[0] + crop_w / 2)
        cy2 = int(center[1] + crop_h / 2)
        x1_clip = max(0, cx1)
        y1_clip = max(0, cy1)
        x2_clip = min(img.shape[1], cx2)
        y2_clip = min(img.shape[0], cy2)
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
        img_chw = img_warped.transpose(2, 0, 1)
        batch_tensors.append(img_chw)
        meta_list.append((center.copy(), np.array([longer, longer], dtype=np.float32)))

    batch_tensor = np.stack(batch_tensors, axis=0)
    return batch_tensor, meta_list


def batch_kpt_postprocess(simcc_x, simcc_y, meta_list, input_size, simcc_split_ratio):
    """
    Batch decode keypoints from SIMCC output.

    Args:
        simcc_x: (N, K, S) cumulative distribution for x
        simcc_y: (N, K, S) cumulative distribution for y
        meta_list: List of (center, scale_fixed) for each ROI
        input_size: (W, H) tuple
        simcc_split_ratio: SIMCC split ratio

    Returns:
        results: List of (keypoints, scores) tuples
    """
    N = simcc_x.shape[0]
    K = simcc_x.shape[1]
    results = []

    for i in range(N):
        center, scale_fixed = meta_list[i]
        x_coords = simcc_x[i].argmax(axis=-1) / simcc_split_ratio
        y_coords = simcc_y[i].argmax(axis=-1) / simcc_split_ratio
        x_conf = simcc_x[i].max(axis=-1)
        y_conf = simcc_y[i].max(axis=-1)
        scores = (x_conf + y_conf) * 0.5

        kpts = np.stack([x_coords, y_coords], axis=-1)
        kpts_orig = kpts / np.array(input_size[::-1]) * scale_fixed + center - 0.5 * scale_fixed
        results.append((kpts_orig, scores))

    return results


def run_batched_inference(engine, batch_input, max_batch_size):
    """
    Run batched inference, splitting large batches into smaller chunks.

    Args:
        engine: TRTEngine instance
        batch_input: (N, C, H, W) input tensor
        max_batch_size: Maximum batch size supported by the engine

    Returns:
        Combined outputs from all batches
    """
    n_samples = batch_input.shape[0]
    if n_samples <= max_batch_size:
        return engine.run(batch_input.astype(np.float32))

    print(f'  Splitting batch of {n_samples} into chunks of {max_batch_size}')
    all_outputs = []
    for start_idx in range(0, n_samples, max_batch_size):
        end_idx = min(start_idx + max_batch_size, n_samples)
        batch_chunk = batch_input[start_idx:end_idx]
        chunk_outputs = engine.run(batch_chunk.astype(np.float32))
        all_outputs.append(chunk_outputs)

    combined_outputs = []
    num_outputs = len(all_outputs[0])
    for out_idx in range(num_outputs):
        if isinstance(all_outputs[0][out_idx], np.ndarray):
            combined_outputs.append(np.concatenate(
                [out[out_idx] for out in all_outputs], axis=0
            ))
        else:
            combined_outputs.append(all_outputs[0][out_idx])

    return combined_outputs


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
    parser = argparse.ArgumentParser(description='Pipeline: detection + keypoint inference with TensorRT')
    parser.add_argument('--det-engine', type=str,
                        default='runs/detect/ckpts/BakingRecognize/202608130125/weights/best.engine')
    parser.add_argument('--kpt-engine', type=str,
                        default='ckpts/rtmpose/BakingRefine/202608112313/best_coco_AP_epoch_40.engine')
    parser.add_argument('--input-dir', type=str, default='/home/wenjie/Downloads/test')
    parser.add_argument('--output-dir', type=str, default='runs/test_trt')
    parser.add_argument('--det-imgsz', type=int, nargs=2, default=[640, 384],
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

    print(f'Loading detection model: {args.det_engine}')
    det_engine = TRTEngine(args.det_engine)
    det_engine.allocate_buffers()
    det_input_w, det_input_h = args.det_imgsz[0], args.det_imgsz[1]

    print(f'Loading keypoint model: {args.kpt_engine}')
    kpt_engine = TRTEngine(args.kpt_engine)
    kpt_engine.allocate_buffers()
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

        process_img(img_orig, img_path, det_engine, det_input_w, det_input_h,
                    kpt_engine, kpt_input_size, args, output_dir, False)
        if args.flip:
            img_flip = cv2.flip(img_orig, 1)
            process_img(img_flip, img_path, det_engine, det_input_w, det_input_h,
                        kpt_engine, kpt_input_size, args, output_dir, True)

    if _cuda_initialized and _cuda_context:
        _pop_cuda_context()

    print(f'\nDone. {len(img_paths)} images -> {output_dir}/')


def process_img(img, img_path, det_engine, det_input_w, det_input_h,
                kpt_engine, kpt_input_size, args, output_dir, is_flip):
    img_h, img_w = img.shape[:2]
    suffix = '_flip' if is_flip else ''

    det_input, scale, pad_w, pad_h = det_preprocess(
        img, det_input_w, det_input_h)
    det_output = det_engine.run(det_input)

    detections = det_postprocess(
        det_output[0], scale, pad_w, pad_h, (img_h, img_w),
        conf_thres=args.det_nms_conf)

    print(f'  Detections: {len(detections)}')
    vis = img.copy()

    pose_targets = []
    for x1, y1, x2, y2, conf, cls_id in detections:
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        draw_detection(vis, ix1, iy1, ix2, iy2, cls_id, conf)
        w, h = x2 - x1, y2 - y1
        ar = w / h if h > 0 else float('inf')
        if (cls_id > args.det_cls_min and conf >= args.det_conf
                and args.ar_min <= ar <= args.ar_max):
            ex1, ey1, ex2, ey2 = expand_bbox(
                x1, y1, x2, y2, args.expand_ratio, img_w, img_h)
            pose_targets.append((ex1, ey1, ex2, ey2, cls_id, conf))

    print(f'  Pose targets: {len(pose_targets)}')

    if pose_targets:
        roi_list = [(p[0], p[1], p[2], p[3]) for p in pose_targets]
        batch_input, meta_list = batch_kpt_preprocess(img, roi_list, kpt_input_size)

        # for i, (vis_img, meta) in enumerate(zip(batch_input, meta_list)):
        #     vis_debug = vis_img.transpose(1, 2, 0).astype(np.uint8)
        #     debug_path = output_dir / f'{img_path.stem}_batch{i}{img_path.suffix}'
        #     cv2.imwrite(str(debug_path), vis_debug)
        #     print(f'  Debug batch {i}: shape={vis_img.shape}, meta={meta}')

        # print(f'  batch_input shape: {batch_input.shape}, dtype: {batch_input.dtype}')
        # print(f'  batch_input range: [{batch_input.min()}, {batch_input.max()}]')
        # print(f'  kpt_engine input_shape: {kpt_engine.input_shape}')
        # print(f'  kpt_engine context tensor shapes:')
        # for name in kpt_engine.input_names + kpt_engine.output_names:
        #     print(f'    {name}: {kpt_engine.context.get_tensor_shape(name)}')

        trt_out = run_batched_inference(kpt_engine, batch_input, kpt_engine.max_batch_size)
        simcc_x, simcc_y = trt_out[0], trt_out[1]
        # print(f'  TRT simcc_x shape: {simcc_x.shape}, simcc_y shape: {simcc_y.shape}')
        kpt_results = batch_kpt_postprocess(
            simcc_x, simcc_y, meta_list, kpt_input_size, args.simcc_split_ratio)

        for (kpts_orig, scores), (_, _, _, _, cls_id, conf) in zip(kpt_results, pose_targets):
            # print(f'  ROI: kpts_orig={kpts_orig[:2]}, scores={scores[:2]}')
            draw_keypoints(vis, kpts_orig, scores, args.kpt_conf, cls_id)

    save_path = output_dir / f'{img_path.stem}{suffix}{img_path.suffix}'
    cv2.imwrite(str(save_path), vis)
    print(f'  Saved: {save_path}')


if __name__ == '__main__':
    main()