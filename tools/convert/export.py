import argparse
import io
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from rfdetr import RFDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Export RF-DETR model to ONNX format")
    parser.add_argument("--weight-path", type=str, required=True, help="path to checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory for exported model")
    parser.add_argument("--output-name", type=str, default=None, help="output filename (without extension)")
    parser.add_argument("--imgsz", type=int, default=384, help="input image size (square, height=width)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for export")
    parser.add_argument("--dynamic-batch", action="store_true", default=False, help="export with dynamic batch dimension")
    parser.add_argument("--fp16", action="store_true", default=False, help="export with FP16 precision")
    parser.add_argument("--trust-checkpoint", action="store_true", default=False, help="allow unsafe checkpoint deserialization")
    parser.add_argument("--test-image", type=str, default=None, help="path to test image for inference visualization")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="confidence threshold for visualization")
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def run_inference(onnx_path, image_path, conf_thres=0.3, target_size=None):
    """
    Run ONNX inference with preprocessing/postprocessing that exactly matches RFDETR.predict().

    Differences from original implementation:
      1. Preprocessing: direct bilinear resize (no letterbox padding) + ImageNet normalization
         (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), matching rfdetr's
         torchvision F.resize(antialias=False) + F.normalize.
      2. Postprocessing: boxes scaled directly to original image size (no padding compensation),
         matching rfdetr's PostProcess._gather_and_scale_boxes.
      3. Logit stripping: RF-DETR ONNX outputs [num_classes+1] columns (last is no-object slot);
         only [:, :-1] is used, matching rfdetr's _run_inference.
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # ONNX NCHW: [batch, channels, height, width]
    _, channels, target_h, target_w = input_shape

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_bgr.shape[:2]

    # ── Preprocessing (matches RFDETR.predict): ──────────────────────────────
    # Direct bilinear resize to target size (no letterbox/padding).
    # antialias=False convention matched by cv2.INTER_LINEAR.
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # HWC → CHW, [0,255] → [0,1]
    chw = resized.transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)

    # ImageNet normalization (matches RFDETR.predict: F.normalize with mean/std)
    _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    _std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    chw = (chw - _mean) / _std

    input_data = chw[np.newaxis]  # (1, 3, H, W)
    # ───────────────────────────────────────────────────────────────────────

    outputs = session.run(None, {input_name: input_data})

    # ── Postprocessing (matches rfdetr export/_onnx/inference.py _run_inference) ──
    # RF-DETR ONNX output: dets=(Q,4) cxcywh norm, labels=(Q, C+1) with no-object slot
    dets, labels = outputs[0][0], outputs[1][0]   # (Q,4), (Q, C+1)

    # Strip background/no-object column (matches rfdetr: logits[:, :-1])
    num_classes = labels.shape[-1] - 1
    labels = labels[:, :-1]   # (Q, C)

    # Per-class sigmoid confidence (matches rfdetr PostProcess._select_topk)
    scores_all = 1.0 / (1.0 + np.exp(-labels))
    scores = scores_all.max(axis=-1)
    cls_ids = scores_all.argmax(axis=-1).astype(int)

    mask = scores > conf_thres
    dets = dets[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    # Convert cxcywh → xyxy (norm) → pixel coords, scaled to original image size
    # (matches rfdetr PostProcess._gather_and_scale_boxes: boxes * [orig_w, orig_h, orig_w, orig_h])
    cx, cy, bw, bh = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    xyxy = np.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5], axis=1)
    xyxy = xyxy * np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)

    x1 = np.clip(xyxy[:, 0], 0, orig_w - 1)
    y1 = np.clip(xyxy[:, 1], 0, orig_h - 1)
    x2 = np.clip(xyxy[:, 2], 0, orig_w - 1)
    y2 = np.clip(xyxy[:, 3], 0, orig_h - 1)
    xyxy_clipped = np.stack([x1, y1, x2, y2], axis=1)
    # ───────────────────────────────────────────────────────────────────────────────

    # ── Visualization ─────────────────────────────────────────────────────────
    vis = image_bgr.copy()
    colors = [
        (220, 20, 60), (0, 139, 139), (255, 140, 0),
        (148, 0, 211), (0, 100, 0), (70, 130, 180),
        (220, 20, 60), (178, 34, 34), (34, 139, 34),
    ]
    for det, conf, cls_id in zip(xyxy_clipped, scores, cls_ids):
        x1_i, y1_i, x2_i, y2_i = det
        color = colors[cls_id % len(colors)]
        cv2.rectangle(vis, (int(x1_i), int(y1_i)), (int(x2_i), int(y2_i)), color, 3)
        label_text = f"cls{cls_id}: {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis, (int(x1_i), int(y1_i) - lh - 8), (int(x1_i) + lw, int(y1_i)), color, -1)
        cv2.putText(vis, label_text, (int(x1_i), int(y1_i) - 3),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def load_model_from_checkpoint(weight_path, trust_checkpoint=False):
    import logging
    old_handlers = logging.root.handlers[:]
    logging.root.handlers = []
    null_handler = logging.NullHandler()
    logging.root.addHandler(null_handler)
    logging.root.setLevel(logging.CRITICAL)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        model = RFDETR.from_checkpoint(weight_path, trust_checkpoint=trust_checkpoint)
    finally:
        sys.stdout = old_stdout
        logging.root.handlers = old_handlers
    return model


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Weight file not found: {args.weight_path}")

    if args.output_dir is None:
        output_dir = Path(args.weight_path).parent
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name
    if output_name is None:
        output_name = Path(args.weight_path).stem

    onnx_path = output_dir / f"{output_name}.onnx"

    if onnx_path.exists():
        print(f"Found existing ONNX model: {onnx_path}")
    else:
        print(f"Loading model from {args.weight_path}...")
        model = load_model_from_checkpoint(args.weight_path, args.trust_checkpoint)

        print(f"Exporting to ONNX with imgsz={args.imgsz}, batch_size={args.batch_size}...")

        model.export(
            output_dir=str(output_dir),
            format="onnx",
            shape=[args.imgsz, args.imgsz],
            batch_size=args.batch_size,
            dynamic_batch=args.dynamic_batch,
            fp16=args.fp16,
        )
        existing_onnx = None
        for f in output_dir.glob("*.onnx"):
            existing_onnx = f
            break
        if existing_onnx:
            print(f"Found existing ONNX model: {existing_onnx}")
            existing_onnx.rename(onnx_path)
            print(f"Renamed {existing_onnx.name} to {onnx_path}")

    if onnx_path.exists():
        print(f"Using ONNX model: {onnx_path}")

    if args.test_image and os.path.exists(args.test_image):
        print(f"\nRunning inference on {args.test_image}...")
        vis_image = run_inference(str(onnx_path), args.test_image, args.conf_thres)

        test_output_dir = Path("runs/test")
        test_output_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        image_name = Path(args.test_image).stem
        output_path = test_output_dir / f"{ts}_{image_name}_onnx.jpg"
        cv2.imwrite(str(output_path), vis_image)
        print(f"Visualization saved to: {output_path}")