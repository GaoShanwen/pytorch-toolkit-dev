import argparse
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
    parser.add_argument("--imgsz", type=int, default=[384, 640], help="input image size")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for export")
    parser.add_argument("--dynamic-batch", action="store_true", default=False, help="export with dynamic batch dimension")
    parser.add_argument("--fp16", action="store_true", default=False, help="export with FP16 precision")
    parser.add_argument("--trust-checkpoint", action="store_true", default=False, help="allow unsafe checkpoint deserialization")
    parser.add_argument("--test-image", type=str, default=None, help="path to test image for inference visualization")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="confidence threshold for visualization")
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def draw_detections(image, dets, labels, conf_thres=0.3, scale=1.0, pad_w=0, pad_h=0):
    orig_h, orig_w = image.shape[:2]
    target_w = orig_w
    target_h = orig_h

    dets = dets.reshape(-1, 4)
    labels = labels.reshape(-1, labels.shape[-1])

    confs = sigmoid(labels.max(axis=-1))
    cls_ids = labels.argmax(axis=-1).astype(int)

    mask = confs > conf_thres
    dets = dets[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    if len(confs) == 0:
        return image

    colors = [
        (220, 20, 60), (0, 139, 139), (255, 140, 0),
        (148, 0, 211), (0, 100, 0), (70, 130, 180),
        (220, 20, 60), (178, 34, 34), (34, 139, 34),
    ]

    for i, (det, conf, cls_id) in enumerate(zip(dets, confs, cls_ids)):
        cx, cy, bw, bh = det

        x1_norm = cx - bw * 0.5
        y1_norm = cy - bh * 0.5
        x2_norm = cx + bw * 0.5
        y2_norm = cy + bh * 0.5

        x1 = (x1_norm * target_w - pad_w) / scale
        y1 = (y1_norm * target_h - pad_h) / scale
        x2 = (x2_norm * target_w - pad_w) / scale
        y2 = (y2_norm * target_h - pad_h) / scale

        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        color = colors[cls_id % len(colors)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

        label = f"cls{cls_id}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (int(x1), int(y1) - label_h - 8), (int(x1) + label_w, int(y1)), color, -1)
        cv2.putText(image, label, (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return image


def draw_detections_yolo(image, output, conf_thres=0.3, scale=1.0, pad_w=0, pad_h=0):
    orig_h, orig_w = image.shape[:2]

    output = output.reshape(-1, 6)
    if len(output) == 0:
        return image

    x1 = output[:, 0]
    y1 = output[:, 1]
    x2 = output[:, 2]
    y2 = output[:, 3]
    confs = output[:, 4]
    cls_ids = output[:, 5].astype(int)

    mask = confs > conf_thres
    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    confs, cls_ids = confs[mask], cls_ids[mask]

    if len(x1) == 0:
        return image

    x1 = (x1 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    x2 = (x2 - pad_w) / scale
    y2 = (y2 - pad_h) / scale

    x1 = x1.clip(0, orig_w - 1)
    y1 = y1.clip(0, orig_h - 1)
    x2 = x2.clip(0, orig_w - 1)
    y2 = y2.clip(0, orig_h - 1)

    colors = [
        (220, 20, 60), (0, 139, 139), (255, 140, 0),
        (148, 0, 211), (0, 100, 0), (70, 130, 180),
        (220, 20, 60), (178, 34, 34), (34, 139, 34),
    ]

    for i, (x1_i, y1_i, x2_i, y2_i, conf, cls_id) in enumerate(zip(x1, y1, x2, y2, confs, cls_ids)):
        color = colors[cls_id % len(colors)]
        cv2.rectangle(image, (int(x1_i), int(y1_i)), (int(x2_i), int(y2_i)), color, 3)

        label = f"cls{cls_id}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (int(x1_i), int(y1_i) - label_h - 8), (int(x1_i) + label_w, int(y1_i)), color, -1)
        cv2.putText(image, label, (int(x1_i), int(y1_i) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return image


def run_inference(onnx_path, image_path, conf_thres=0.3):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    orig_h, orig_w = image.shape[:2]
    target_h, target_w = input_shape[2], input_shape[3]

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(image, (new_w, new_h))
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    input_data = np.zeros((1, 3, target_h, target_w), dtype=np.float32)
    input_data[:, :, pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized.transpose(2, 0, 1) / 255.0

    outputs = session.run(None, {input_name: input_data})

    num_outputs = len(outputs)
    if num_outputs == 2:
        dets, labels = outputs
        vis_image = draw_detections(image.copy(), dets, labels, conf_thres, scale, pad_w, pad_h)
    elif num_outputs == 1:
        output = outputs[0]
        vis_image = draw_detections_yolo(image.copy(), output, conf_thres, scale, pad_w, pad_h)
    else:
        raise ValueError(f"Unexpected number of ONNX outputs: {num_outputs}")

    return vis_image


def load_model_from_checkpoint(weight_path, trust_checkpoint=False):
    import io
    import logging

    ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)

    if 'model' in ckpt and 'args' in ckpt:
        wrapper = {
            'args': ckpt.get('args'),
            'model': ckpt.get('model'),
            'model_name': ckpt.get('model_name'),
            'rfdetr_version': ckpt.get('rfdetr_version'),
            'state_dict': ckpt.get('state_dict'),
        }
        old_handlers = logging.root.handlers[:]
        logging.root.handlers = []
        null_handler = logging.NullHandler()
        logging.root.addHandler(null_handler)
        logging.root.setLevel(logging.CRITICAL)

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            model = RFDETR.from_checkpoint(wrapper, trust_checkpoint=True)
        finally:
            sys.stdout = old_stdout
            logging.root.handlers = old_handlers
        return model
    else:
        return RFDETR.from_checkpoint(weight_path, trust_checkpoint=trust_checkpoint)


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
        existing_onnx = None
        for f in output_dir.glob("*.onnx"):
            existing_onnx = f
            break

        if existing_onnx:
            print(f"Found existing ONNX model: {existing_onnx}")
            existing_onnx.rename(onnx_path)
            print(f"Renamed {existing_onnx.name} to {onnx_path}")
        else:
            print(f"Loading model from {args.weight_path}...")
            model = load_model_from_checkpoint(args.weight_path, args.trust_checkpoint)

            if isinstance(args.imgsz, int):
                args.imgsz = [args.imgsz, args.imgsz]
            print(f"Exporting to ONNX with imgsz={args.imgsz}, batch_size={args.batch_size}...")

            model.export(
                output_dir=str(output_dir),
                format="onnx",
                shape=args.imgsz,
                batch_size=args.batch_size,
                dynamic_batch=args.dynamic_batch,
                fp16=args.fp16,
            )

            rfdetr_onnx = output_dir / "rfdetr-nano.onnx"
            if rfdetr_onnx.exists() and not onnx_path.exists():
                rfdetr_onnx.rename(onnx_path)
                print(f"Renamed rfdetr-nano.onnx to {onnx_path}")

    if onnx_path.exists():
        print(f"Using ONNX model: {onnx_path}")

    if args.test_image and os.path.exists(args.test_image):
        print(f"\nRunning inference on {args.test_image}...")
        vis_image = run_inference(str(onnx_path), args.test_image, args.conf_thres)

        test_output_dir = Path("runs/test")
        test_output_dir.mkdir(parents=True, exist_ok=True)

        image_name = Path(args.test_image).stem
        output_path = test_output_dir / f"{image_name}_result.jpg"
        cv2.imwrite(str(output_path), vis_image)
        print(f"Visualization saved to: {output_path}")