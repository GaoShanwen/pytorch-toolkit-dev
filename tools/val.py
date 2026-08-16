######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.07.29
# filename: val.py
# function: validate dataset use RF-DETR.
######################################################
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from rfdetr import RFDETR
from rfdetr.datasets import detect_roboflow_format
from rfdetr.datasets.yolo import YOLO_IMAGE_EXTENSIONS, _list_yolo_image_paths


def parse_args():
    parser = argparse.ArgumentParser(description="RF-DETR validation script")
    parser.add_argument(
        "--model",
        type=str,
        default="ckpts/detect/BakingRecognizeCOCO/202608141803/checkpoint_best_total.pth",
        help="path to checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/det-dataset/BakingRecognizeCOCO",
        help="path to dataset directory or dataset.yaml",
    )
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="input image size")
    parser.add_argument("--device", type=str, default="0", help="device, e.g. 0, cpu, cuda:0")
    parser.add_argument("--workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--project", type=str, default="runs", help="project directory for validation outputs")
    parser.add_argument("--name", type=str, default="val", help="run name under project directory")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="dataset split to evaluate")
    parser.add_argument("--threshold", type=float, default=0.3, help="confidence threshold for saved visualizations")
    parser.add_argument("--no-save-vis", action="store_true", help="disable prediction visualizations")
    parser.add_argument("--trust-checkpoint", action="store_true", help="allow unsafe checkpoint deserialization")
    return parser.parse_args()


def normalize_device(device: str) -> str:
    if device.isdigit():
        return "cuda"
    if device.startswith("cuda:"):
        return "cuda"
    return device


def resolve_dataset_dir(data_path: str) -> Path:
    path = Path(data_path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return path.parent
    return path


def resolve_output_dir(project: str, name: str) -> Path:
    output_dir = Path(project) / name
    if project == "runs" and name == "val":
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def collect_split_images(dataset_dir: Path, split: str) -> list[str]:
    split_dirs = ("test",) if split == "test" else ("valid", "val")
    try:
        dataset_format = detect_roboflow_format(dataset_dir)
    except ValueError:
        dataset_format = None

    image_paths: list[str] = []
    for split_dir_name in split_dirs:
        split_dir = dataset_dir / split_dir_name
        if not split_dir.exists():
            continue

        if dataset_format == "yolo":
            images_dir = split_dir / "images"
            if images_dir.exists():
                image_paths.extend(_list_yolo_image_paths(str(images_dir)))
        else:
            image_paths.extend(
                sorted(
                    str(path)
                    for path in split_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in YOLO_IMAGE_EXTENSIONS
                )
            )

    return sorted(set(image_paths))


def save_metrics(metrics: dict[str, float], output_dir: Path, args: argparse.Namespace) -> None:
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    summary_lines = ["Validation Results", "=" * 32]
    for key in sorted(metrics):
        summary_lines.append(f"{key}: {metrics[key]:.6f}")
    summary_text = "\n".join(summary_lines) + "\n"

    summary_path = output_dir / "metrics.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    args_path = output_dir / "args.json"
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(summary_text)
    print(f"Metrics saved to {metrics_path}")


def save_visualizations(model: RFDETR, image_paths: list[str], output_dir: Path, threshold: float) -> None:
    import numpy as np
    import supervision as sv
    from PIL import Image

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


def validate(args: argparse.Namespace) -> dict[str, float]:
    print(args)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    dataset_dir = resolve_dataset_dir(args.data)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_dir = resolve_output_dir(args.project, args.name)

    model = RFDETR.from_checkpoint(args.model, trust_checkpoint=args.trust_checkpoint)

    metrics = model.evaluate(
        dataset_dir=str(dataset_dir),
        split=args.split,
        batch_size=args.batch,
        resolution=args.imgsz,
        device=normalize_device(args.device),
        num_workers=args.workers,
        output_dir=str(output_dir),
    )

    save_metrics(metrics, output_dir, args)

    if not args.no_save_vis:
        image_paths = collect_split_images(dataset_dir, args.split)
        if image_paths:
            save_visualizations(model, image_paths, output_dir, args.threshold)
        else:
            print(f"No images found for split '{args.split}' under {dataset_dir}")

    print(f"Validation artifacts saved to {output_dir}")
    return metrics


if __name__ == "__main__":
    validate(parse_args())