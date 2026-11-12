#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: train.py
# function: train a segmentation model with segmentation_models_pytorch.
######################################################
import os
import sys
import time
import random
import logging

# make the repo root importable when running as `python tools/train.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# fix the ssl certificates / the huggingface endpoint *before* torch and
# huggingface_hub are imported
from local_lib.utils.net_setup import setup as setup_network
from local_lib.utils.net_setup import setup_wandb_observability

setup_network()

import numpy as np
import torch
from tqdm import tqdm

from local_lib.data import create_seg_loaders, get_dataset_name
from local_lib.models import (
    create_seg_model,
    create_seg_loss,
    create_seg_optimizer,
    create_scheduler,
    get_device,
    get_preprocessing_params,
    save_checkpoint,
    setup_logging,
)
from local_lib.utils import SegLogger, SegMetricTracker, parse_args, save_config

_logger = logging.getLogger("train")


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def run_epoch(model, loader, criterion, device, mean, std, optimizer=None, scheduler=None,
              num_classes=2, classes=None, ignore_index=255, scaler=None, desc="train"):
    """Run one epoch and return the aggregated metrics."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    tracker = SegMetricTracker(num_classes, ignore_index=ignore_index)
    ctx = torch.enable_grad() if is_train else torch.inference_mode()

    with ctx:
        for images, masks in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            images = (images - mean) / std

            if scaler is not None and is_train:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                    logits = model(images)
                    loss = criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, masks)
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

            pred = logits.argmax(dim=1)  # (N, H, W)
            tracker.update(pred, masks, loss=loss.item())

    if scheduler is not None and is_train:
        scheduler.step()

    return tracker.compute(classes=classes)


def main():
    setup_logging("train")
    cfg, args, args_text = parse_args("Train a segmentation model with smp")

    # the root of the repo; the git branch (used as the name of the wandb
    # project) is looked up there
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg["repo_root"] = repo_root

    # the wandb system metrics are sent by an opentelemetry exporter of the
    # wandb service; it must be disabled before the service is started
    setup_wandb_observability(
        enable=bool((cfg.get("wandb", {}) or {}).get("observability", False))
    )

    seed = int(cfg.get("seed", 42))
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = get_device(cfg)

    num_classes = len(cfg.get("classes", []) or [])
    if num_classes == 0:
        raise ValueError("'classes' must be defined in the config file.")
    class_names = list(cfg.get("classes", []))

    output_dir = cfg.get("output", "output/ckpts")
    # the name of the dataset is added to the output folder, so the runs of
    # the different datasets are not mixed up
    dataset_name = get_dataset_name(cfg)
    if dataset_name:
        output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    exp_name = cfg.get("experiment") or time.strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ---- model / loss / optimizer ----
    model = create_seg_model(cfg, num_classes=num_classes)
    model.to(device)
    criterion = create_seg_loss(cfg).to(device)
    optimizer = create_seg_optimizer(cfg, model)

    train_loader, val_loader = create_seg_loaders(cfg)
    scheduler = create_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))
    mean, std = get_preprocessing_params(cfg)
    mean = mean.to(device)
    std = std.to(device)

    _logger.info(
        f"model={cfg.get('model', 'UPerNet')} encoder={cfg.get('encoder_name', 'resnet34')} "
        f"classes={num_classes} device={device}"
    )
    _logger.info(f"train imgs={len(train_loader.dataset)}  val imgs={len(val_loader.dataset)}")

    # ---- logger ----
    logger = SegLogger(
        cfg=cfg,
        exp_name=exp_name,
        log_dir=cfg.get("log_dir", "logs"),
        log_wandb=bool(cfg.get("log_wandb", True)),
        tensorboard=cfg.get("tensorboard", ""),
    )
    if logger.wandb_run is not None:
        _logger.info(f"wandb run: {logger.wandb_run.name} ({logger.wandb_run.url})")
        # track the gradients of the model
        logger.watch(
            model,
            criterion=criterion,
            log=cfg.get("wandb", {}).get("watch", "gradients"),
            log_freq=int(cfg.get("wandb", {}).get("watch_freq", 1000)),
        )

    # ---- resume / finetune ----
    start_epoch = 0
    best_metric = 0.0
    resume = args.checkpoint or cfg.get("resume", "")
    if resume:
        from local_lib.models import load_checkpoint

        payload = load_checkpoint(resume, model, map_location=device, strict=False)
        saved_cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
        same_arch = saved_cfg.get("model") == cfg.get("model") and saved_cfg.get(
            "encoder_name"
        ) == cfg.get("encoder_name")
        if same_arch:
            # a full resume: restore the optimizer and continue the schedule
            if isinstance(payload, dict) and "optimizer" in payload:
                try:
                    optimizer.load_state_dict(payload["optimizer"])
                except ValueError as exc:
                    _logger.warning(f"could not restore the optimizer state ({exc}).")
            start_epoch = int(payload.get("epoch") or 0) + 1
            best_metric = float(payload.get("best_metric") or 0.0)
            _logger.info(f"resumed from {resume} at epoch {start_epoch}")
        else:
            # a finetune: only the weights are restored, the schedule restarts
            _logger.info(
                f"loaded the weights of {resume} for finetuning "
                f"(checkpoint arch={saved_cfg.get('model')}, config arch={cfg.get('model')})."
            )

    save_config(cfg, os.path.join(exp_dir, "config.yaml"))

    # ---- train loop ----
    epochs = int(cfg.get("epochs", 100))
    ignore_index = int(cfg.get("ignore_index", 255))
    amp = bool(cfg.get("amp", False))
    scaler = torch.amp.GradScaler(device.type) if amp and device.type == "cuda" else None
    if amp:
        _logger.info("mixed precision (AMP) training enabled")

    best_path = os.path.join(exp_dir, "best.pth")
    last_path = os.path.join(exp_dir, "last.pth")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_metrics = run_epoch(
            model, train_loader, criterion, device, mean, std,
            optimizer=optimizer, scheduler=scheduler, num_classes=num_classes,
            classes=class_names, ignore_index=ignore_index, scaler=scaler,
            desc=f"train {epoch+1}/{epochs}",
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device, mean, std,
            optimizer=None, num_classes=num_classes, classes=class_names,
            ignore_index=ignore_index, desc=f"val   {epoch+1}/{epochs}",
        )

        lr = optimizer.param_groups[0]["lr"]
        logger.update(epoch, {
            "train/loss": train_metrics["loss"],
            "train/miou": train_metrics["miou"],
            "val/loss": val_metrics["loss"],
            "val/miou": val_metrics["miou"],
            "val/iou": val_metrics["iou"],
            "lr": lr,
        })
        for name, value in val_metrics["iou_per_class"].items():
            logger.update(epoch, {f"val/iou/{name}": value})

        improved = val_metrics["miou"] > best_metric
        if improved:
            best_metric = val_metrics["miou"]
            save_checkpoint(best_path, model, cfg, val_metrics, epoch, optimizer, best_metric)

        save_checkpoint(last_path, model, cfg, val_metrics, epoch, optimizer, best_metric)

        _logger.info(
            f"epoch {epoch+1}/{epochs} lr={lr:.6f} "
            f"train_loss={train_metrics['loss']:.4f} train_mIoU={train_metrics['miou']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mIoU={val_metrics['miou']:.4f} "
            f"val_IoU={val_metrics['iou']:.4f} time={time.time()-t0:.1f}s"
            f"{'  *best*' if improved else ''}"
        )
        _logger.info(f"  per-class IoU: {val_metrics['iou_per_class']}")

    logger.close()
    _logger.info(f"training finished. best mIoU={best_metric:.4f}  checkpoint: {best_path}")


if __name__ == "__main__":
    main()
