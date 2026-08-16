import argparse
import os
import sys
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

sys.path.append(".")
from rfdetr import RFDETR


def parse_args():
    parser = argparse.ArgumentParser(description="RF-DETR Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="ckpts/detect/BakingRecognizeCOCO/202608141803/checkpoint_best_total.pth",
        help="path to checkpoint",
    )
    parser.add_argument(
        "--img-path",
        type=str,
        required=True,
        help="path to image or directory containing images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="device, e.g. 0, cpu, cuda:0",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="confidence threshold for predictions",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/predict",
        help="directory to save results",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="allow unsafe checkpoint deserialization",
    )
    return parser.parse_args()


def normalize_device(device: str) -> str:
    if device.isdigit():
        return "cuda"
    if device.startswith("cuda:"):
        return "cuda"
    return device


def predict(args: argparse.Namespace) -> None:
    print(args)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"Image path not found: {args.img_path}")

    model = RFDETR.from_checkpoint(args.model, trust_checkpoint=args.trust_checkpoint)

    img_path = Path(args.img_path)
    if img_path.is_file():
        image_paths = [str(img_path)]
    else:
        image_paths = sorted(
            str(p) for p in img_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

    if not image_paths:
        print(f"No images found in {args.img_path}")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)
    class_names = getattr(model.model, "class_names", None) or []

    for image_path in image_paths:
        detections = model.predict(image_path, threshold=args.threshold)
        image = np.array(Image.open(image_path).convert("RGB"))

        labels = []
        if detections.class_id is not None:
            for i, class_id in enumerate(detections.class_id):
                conf = detections.confidence[i] if detections.confidence is not None else 1.0
                if 0 <= class_id < len(class_names):
                    labels.append(f"{class_names[class_id]} {conf:.2f}")
                else:
                    labels.append(f"cls{class_id} {conf:.2f}")

        annotated = box_annotator.annotate(scene=image, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        save_path = save_dir / Path(image_path).name
        Image.fromarray(annotated).save(save_path)
        print(f"Saved: {save_path}")

    print(f"Total: {len(image_paths)} image(s) processed")


if __name__ == "__main__":
    predict(parse_args())