"""
RF-DETR validation with class-ID remapping support.
Mirrors the class merging logic from train.py so that validation uses the same
remapped label space as training.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from rfdetr import RFDETR
from rfdetr.datasets.coco import build_roboflow_from_coco
from rfdetr._namespace import _namespace_from_configs
from rfdetr.datasets.yolo import _list_yolo_image_paths
from rfdetr.datasets import detect_roboflow_format
from rfdetr.datasets.yolo import YOLO_IMAGE_EXTENSIONS

from ..data.mixed_data.dataset import MixedRFDETRDataset
from ..utils.vis import save_visualizations, save_error_analysis_visualizations

logger = logging.getLogger(__name__)


def resolve_output_dir(project: str, name: str) -> Path:
    output_dir = Path(project) / name
    if project == "runs" and name == "val":
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_class_names(ds, mc):
    label2cat = None
    coco_src = None
    if getattr(ds, "label2cat", None):
        label2cat = ds.label2cat
        coco_src = getattr(ds, "coco", None)
    else:
        raise AssertionError("Could not find label2cat mapping on dataset; cannot derive class names.")

    max_label = max(label2cat.keys())
    class_names: list[str] = [None] * (max_label + 1)
    for label_idx, cat_id in label2cat.items():
        name = None
        if coco_src is not None and hasattr(coco_src, "cats") and cat_id in coco_src.cats:
            name = coco_src.cats[cat_id]["name"]
        else:
            name = f"class_{label_idx}"
        class_names[int(label_idx)] = str(name)

    assert all(n is not None for n in class_names), "Built class_names contains empty entries"
    return class_names


def validate(args: argparse.Namespace) -> dict[str, float]:
    logger.info("Starting validation with args: %s", args)

    opts = getattr(args, "options", {})
    class_mapping = opts.get("class_mapping")

    output_dir = resolve_output_dir(args.project, args.name)

    model = RFDETR.from_checkpoint(args.model, trust_checkpoint=args.trust_checkpoint)
    mc = model.model_config

    train_config = model.get_train_config(
        dataset_dir=args.data,
        output_dir=str(output_dir),
    )
    ns = _namespace_from_configs(mc, train_config)

    ds_val = build_roboflow_from_coco(args.split, ns, mc.resolution)
    print(f"[PRINT] After build_roboflow_from_coco, ds_val type: {type(ds_val)}")

    class_names = _resolve_class_names(ds_val, mc)
    print(f"[PRINT] _resolve_class_names returned: {class_names}")
    train_config.class_names = class_names
    if hasattr(mc, "num_classes"):
        mc.num_classes = len(class_names)
    logger.info("Resolved class_names from dataset label2cat: %s", class_names)

    if class_mapping:
        ds_val_for_vis = build_roboflow_from_coco(args.split, ns, mc.resolution)
        ds_val = MixedRFDETRDataset(ds_val, class_mapping, background_im_files=None, alpha=None)
        logger.info("Applied class_mapping to validation dataset: %s", class_mapping)
    else:
        ds_val_for_vis = ds_val

    resolved_names = list(train_config.class_names)
    logger.info("DEBUG resolved_names: %s", resolved_names)

    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    eval_mc = mc.model_copy(update={"pretrain_weights": None})
    module = RFDETRModelModule(eval_mc, train_config)
    source_state = model.model.model.state_dict()
    module.model.load_state_dict(source_state)
    module.model.class_names = resolved_names
    if getattr(module.model, "args", None) is not None:
        setattr(module.model.args, "class_names", resolved_names)
    logger.info("DEBUG module.model.class_names set to: %s", module.model.class_names)

    model.model.class_names = resolved_names

    datamodule = RFDETRDataModule(eval_mc, train_config)
    datamodule._dataset_val = ds_val
    datamodule.setup("validate")

    accelerator, devices = model._resolve_trainer_device_kwargs(args.device)
    trainer_kwargs: dict = {"accelerator": accelerator, "devices": devices, "include_training_callbacks": False}
    trainer = build_trainer(train_config, mc, **trainer_kwargs)

    for callback in trainer.callbacks:
        if hasattr(callback, "_cat_id_to_name") and hasattr(callback, "_resolve_class_names"):
            callback._cat_id_to_name = {i: name for i, name in enumerate(resolved_names)}
            callback._class_names = resolved_names
            break

    metrics = trainer.validate(model=module, datamodule=datamodule)
    if isinstance(metrics, list) and len(metrics) > 0:
        metrics = metrics[0]
    save_metrics(metrics, output_dir, args)

    if not args.no_save_vis:
        dataset_dir = Path(args.data)
        split_dirs = ("test",) if args.split == "test" else ("valid", "val")
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
            if not image_paths:
                image_paths.extend(
                    sorted(
                        str(path)
                        for path in split_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in YOLO_IMAGE_EXTENSIONS
                    )
                )

        if image_paths:
            save_error_analysis_visualizations(model, image_paths, ds_val_for_vis, output_dir, args.threshold, class_mapping=class_mapping)
        else:
            logger.warning("No images found for split '%s' under %s", args.split, dataset_dir)

    logger.info("Validation artifacts saved to %s", output_dir)
    return metrics


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

    # print(summary_text)
    print(f"Metrics saved to {metrics_path}")