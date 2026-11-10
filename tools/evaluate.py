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
from local_lib.models import create_seg_model, create_seg_loss, get_preprocessing_params, load_checkpoint
from local_lib.utils import SegMetricTracker, parse_args

_logger = logging.getLogger("evaluate")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_device(cfg):
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        _logger.warning("cuda is requested but not available, fallback to cpu.")
        device = "cpu"
    return torch.device(device)


def build_model_from_checkpoint(cfg, checkpoint_path, device, num_classes):
    """Rebuild the model, preferring the architecture stored in the checkpoint."""
    model_cfg = dict(cfg)
    if checkpoint_path and os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_cfg = payload.get("config") if isinstance(payload, dict) else None
        if saved_cfg:
            # the checkpoint knows its own architecture; keep the cli overrides
            for key in ("model", "encoder_name", "model_kwargs", "in_channels"):
                if key in saved_cfg and key not in cfg:
                    model_cfg[key] = saved_cfg[key]
            if not cfg.get("classes") and saved_cfg.get("classes"):
                model_cfg["classes"] = saved_cfg["classes"]

    # the checkpoint already holds the trained weights, so there is no need to
    # download the pretrained encoder again
    model_cfg["encoder_weights"] = None

    model = create_seg_model(model_cfg, num_classes=num_classes)
    model.to(device)
    if checkpoint_path:
        load_checkpoint(checkpoint_path, model, map_location=device, strict=False)
        _logger.info(f"loaded checkpoint: {checkpoint_path}")
    else:
        _logger.warning("no checkpoint given, evaluating a randomly initialised model.")
    model.eval()
    return model


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
    setup_logging()
    cfg, args, args_text = parse_args("Evaluate a segmentation model with smp")

    checkpoint_path = args.checkpoint or cfg.get("checkpoint", "")
    if not checkpoint_path:
        raise ValueError(
            "a checkpoint is required, pass it with --checkpoint <path> or 'checkpoint:' in the config."
        )

    device = get_device(cfg)
    classes = cfg.get("classes", []) or []
    num_classes = len(classes)
    if num_classes == 0 and checkpoint_path:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved = payload.get("config", {}) if isinstance(payload, dict) else {}
        classes = saved.get("classes", []) or []
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
