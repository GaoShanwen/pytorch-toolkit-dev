######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.12
# filenaem: seg_common.py
# function: helpers shared by the tools of the segmentation (train, evaluate,
#          inference).
######################################################
import logging
import os

import torch

from .seg_checkpoint import load_checkpoint
from .seg_model import create_seg_model


_logger = logging.getLogger("seg_common")

# the keys of the config that describe the architecture of the model; they are
# taken from the checkpoint when they are not given in the config
ARCH_KEYS = ("model", "encoder_name", "model_kwargs", "in_channels")


def setup_logging(name=None):
    """Configure the logging of a tool.

    Args:
        name: the name of the logger; the root logger is configured when it
            is None.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if name:
        logging.getLogger(name).setLevel(logging.INFO)


def get_device(cfg):
    """Return the device the run uses.

    The cuda device falls back to the cpu when it is not available.
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        _logger.warning("cuda is requested but not available, fallback to cpu.")
        device = "cpu"
    return torch.device(device)


def read_checkpoint_config(checkpoint_path):
    """Return the config stored in a checkpoint.

    Returns an empty dict when the checkpoint holds no config.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {}
    return payload.get("config") or {}


def get_checkpoint_classes(cfg, checkpoint_path):
    """Return the names of the classes of the run.

    The classes of the config win; the ones stored in the checkpoint are used
    when the config holds none.
    """
    classes = list(cfg.get("classes", []) or [])
    if not classes:
        classes = list(read_checkpoint_config(checkpoint_path).get("classes", []) or [])
    return classes


def build_model_from_checkpoint(cfg, checkpoint_path, device, num_classes=None):
    """Rebuild the model of a checkpoint and load its weights.

    The architecture stored in the checkpoint wins over the one of the config,
    so a checkpoint can be run with a config that does not describe it.

    Args:
        cfg: the config of the run.
        checkpoint_path: the checkpoint to load.
        device: the device the model is moved to.
        num_classes: the number of classes of the model; the ones of the
            config are used when it is None.

    Returns:
        The model, in eval mode.
    """
    model_cfg = dict(cfg)
    saved_cfg = read_checkpoint_config(checkpoint_path)
    if saved_cfg:
        # the checkpoint knows its own architecture; the cli overrides win
        for key in ARCH_KEYS:
            if key in saved_cfg and key not in cfg:
                model_cfg[key] = saved_cfg[key]
        if not cfg.get("classes") and saved_cfg.get("classes"):
            model_cfg["classes"] = saved_cfg["classes"]

    # the checkpoint already holds the trained weights, so the pretrained
    # weights of the encoder are not needed and nothing is downloaded
    model_cfg["encoder_weights"] = None

    if num_classes is None:
        num_classes = len(model_cfg.get("classes", []) or [])

    model = create_seg_model(model_cfg, num_classes=num_classes)
    model.to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, map_location=device, strict=False)
        _logger.info(f"loaded checkpoint: {checkpoint_path}")
    else:
        _logger.warning("no checkpoint given, the model has random weights.")

    model.eval()
    return model
