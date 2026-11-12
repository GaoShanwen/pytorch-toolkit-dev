#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: evaluate.py
# function: evaluate a segmentation checkpoint on a dataset split.
######################################################
import os
import sys
import json
import logging

# make the repo root importable when running as `python tools/evaluate.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# fix the ssl certificates / the huggingface endpoint *before* torch and
# huggingface_hub are imported
from local_lib.utils.net_setup import setup as setup_network

setup_network()

import torch
from tqdm import tqdm

from local_lib.data import create_seg_dataset, create_seg_loader
from local_lib.models import (
    build_model_from_checkpoint,
    create_seg_loss,
    get_checkpoint_classes,
    get_device,
    get_preprocessing_params,
    setup_logging,
)
from local_lib.utils import SegMetricTracker, parse_args

_logger = logging.getLogger("evaluate")


@torch.no_grad()
def evaluate(model, loader, criterion, device, mean, std, num_classes, classes=None,
             ignore_index=255, desc="eval"):
    """Run the evaluation loop and return the aggregated metrics."""
    tracker = SegMetricTracker(num_classes, ignore_index=ignore_index)

    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        images = (images - mean) / std

        logits = model(images)
        loss = criterion(logits, masks)
        pred = logits.argmax(dim=1)  # (N, H, W)
        tracker.update(pred, masks, loss=loss.item())

    return tracker.compute(classes=classes)


def main():
    setup_logging("evaluate")
    cfg, args, args_text = parse_args("Evaluate a segmentation model with smp")

    checkpoint_path = args.checkpoint or cfg.get("checkpoint", "")
    if not checkpoint_path:
        raise ValueError(
            "a checkpoint is required, pass it with --checkpoint <path> or 'checkpoint:' in the config."
        )

    device = get_device(cfg)
    classes = get_checkpoint_classes(cfg, checkpoint_path)
    num_classes = len(classes)
    if num_classes == 0:
        raise ValueError("'classes' must be defined in the config or stored in the checkpoint.")

    model = build_model_from_checkpoint(cfg, checkpoint_path, device, num_classes)
    criterion = create_seg_loss(cfg).to(device)
    mean, std = get_preprocessing_params(cfg)
    mean = mean.to(device)
    std = std.to(device)

    val_split = (cfg.get("data", {}) or {}).get("val_split", "val")
    dataset = create_seg_dataset(cfg, split=val_split, is_training=False)
    loader = create_seg_loader(
        dataset,
        batch_size=int(cfg.get("batch_size", 8)),
        num_workers=int(cfg.get("workers", 4)),
        shuffle=False,
        seed=int(cfg.get("seed", 42)),
    )
    _logger.info(f"evaluating {len(dataset)} images from split '{val_split}'")

    metrics = evaluate(
        model, loader, criterion, device, mean, std,
        num_classes=num_classes, classes=classes,
        ignore_index=int(cfg.get("ignore_index", 255)),
    )

    print("\n================ Evaluation Result ================")
    print(f"loss   : {metrics['loss']:.4f}")
    print(f"IoU    : {metrics['iou']:.4f}")
    print(f"mIoU   : {metrics['miou']:.4f}")
    print("-" * 46)
    for cls, value in metrics["iou_per_class"].items():
        print(f"{cls:>15s}  IoU={value:.4f}")
    print("==================================================")

    results_file = cfg.get("results_file", "")
    if results_file:
        os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        _logger.info(f"results written to {results_file}")


if __name__ == "__main__":
    main()
