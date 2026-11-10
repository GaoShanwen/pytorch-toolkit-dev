######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: seg_checkpoint.py
# function: save / load segmentation checkpoints.
######################################################
import os
import logging

import torch


_logger = logging.getLogger("seg_checkpoint")


def save_checkpoint(path, model, cfg, metrics=None, epoch=None, optimizer=None, best_metric=None):
    """Save a checkpoint holding the weights plus the config needed to rebuild."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": cfg,
        "classes": cfg.get("classes", []) if cfg else [],
        "num_classes": len(cfg.get("classes", []) or []) if cfg else 0,
        "encoder_name": cfg.get("encoder_name", "resnet34") if cfg else "resnet34",
        "arch": cfg.get("model", "UPerNet") if cfg else "UPerNet",
        "epoch": epoch,
        "metrics": metrics,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path, model, optimizer=None, map_location="cpu", strict=True):
    """Load a checkpoint into ``model``.

    Returns the checkpoint payload so that the caller can inspect the
    stored config / metrics.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)

    state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # tolerate checkpoints saved from a DataParallel / DDP wrapper
    state_dict = {
        (k[len("module.") :] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }

    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        # only load the parameters whose shape matches, so that a checkpoint
        # of another architecture can still be used as a partial init
        model_state = model.state_dict()
        matched = {
            k: v for k, v in state_dict.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        skipped = [k for k in state_dict if k not in matched]
        if skipped:
            _logger.warning(
                f"{len(skipped)} checkpoint tensors do not match the model and were skipped "
                f"(e.g. {skipped[:3]}...)."
            )
        model.load_state_dict(matched, strict=False)

    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except ValueError as exc:
            _logger.warning(f"could not restore the optimizer state ({exc}); using a fresh one.")
    return payload
