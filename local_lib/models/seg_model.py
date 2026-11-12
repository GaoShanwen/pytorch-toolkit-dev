######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: seg_model.py
# function: build smp segmentation models and losses from config.
######################################################
import logging
import os

import torch
import segmentation_models_pytorch as smp


_logger = logging.getLogger("seg_model")


SUPPORTED_ARCHS = [
    "Unet",
    "UnetPlusPlus",
    "MAnet",
    "Linknet",
    "FPN",
    "PSPNet",
    "PAN",
    "DeepLabV3",
    "DeepLabV3Plus",
    "UPerNet",
    "Segformer",
    "DPT",
]


def _read_state_dict(weights_path):
    """Read the weights of a file into a state dict.

    Both formats of the checkpoints are supported:

    - the '.pth' / '.pt' files of torch (a pickle);
    - the '.safetensors' files.

    Args:
        weights_path: the path of the file.

    Returns:
        The state dict.
    """
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(weights_path, device="cpu")

    return torch.load(weights_path, map_location="cpu", weights_only=False)


def _load_local_encoder_weights(model, weights_path, logger):
    """Load the weights of an encoder from a local file.

    The file holds the weights of the encoder only (e.g. the ``mit_b0.pth``
    of the imagenet pretrained encoder); they are loaded into the encoder of
    the model, the decoder and the segmentation head keep their random
    initialisation.

    Args:
        model: the model to load the weights into.
        weights_path: the path of the file.
        logger: the logger of the caller.
    """
    state_dict = _read_state_dict(weights_path)

    # the checkpoints of a model saved with torch hold the keys of the whole
    # model; only the ones of the encoder are kept here
    state_dict = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    encoder_state = model.encoder.state_dict()
    matched = {
        k: v for k, v in state_dict.items()
        if k in encoder_state and encoder_state[k].shape == v.shape
    }
    if not matched:
        raise RuntimeError(
            f"none of the {len(state_dict)} tensors of '{weights_path}' matches the "
            f"encoder of the model."
        )

    skipped = [k for k in state_dict if k not in matched]
    if skipped:
        logger.info(
            f"{len(skipped)} tensors of '{weights_path}' do not belong to the encoder "
            f"and were skipped (e.g. {skipped[:3]}...)."
        )

    # the MiT encoder of smp overrides 'load_state_dict' and does not accept
    # the 'strict' argument, so the base implementation of torch is used
    torch.nn.Module.load_state_dict(model.encoder, matched, strict=False)
    logger.info(f"loaded the encoder weights from '{weights_path}' ({len(matched)} tensors).")


def create_seg_model(cfg, num_classes=None, encoder_weights="imagenet"):
    """Instantiate a segmentation model described by the config.

    Config keys:
        model: architecture name (see :data:`SUPPORTED_ARCHS`).
        encoder_name: the smp encoder / backbone name.
        encoder_weights: ``imagenet`` / ``None`` / ``<path>``; a local path
            is loaded from the disk instead of being downloaded from the hub.
        model_kwargs: extra keyword arguments forwarded to the constructor.
    """
    arch = cfg.get("model", "UPerNet")
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(
            f"unsupported model '{arch}', expect one of {SUPPORTED_ARCHS}"
        )

    encoder_name = cfg.get("encoder_name", "resnet34")
    in_channels = int(cfg.get("in_channels", 3))
    classes = int(num_classes if num_classes is not None else len(cfg.get("classes", []) or []))

    model_kwargs = dict(cfg.get("model_kwargs", {}) or {})
    model_kwargs.pop("encoder_name", None)
    model_kwargs.pop("encoder_weights", None)
    model_kwargs.pop("in_channels", None)
    model_kwargs.pop("classes", None)

    weights = cfg.get("encoder_weights", encoder_weights)
    if weights in ("none", "None", ""):
        weights = None

    # the weights are on the disk: the model is built without the pretrained
    # weights and they are loaded into the encoder afterwards
    if isinstance(weights, str) and os.path.isfile(weights):
        model = getattr(smp, arch)(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            **model_kwargs,
        )
        _load_local_encoder_weights(model, weights, _logger)
        return model

    try:
        model = getattr(smp, arch)(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
            **model_kwargs,
        )
    except Exception as exc:
        # the pretrained weights are downloaded from the hub; when the host
        # has no access to it the model is built without them instead of
        # crashing the whole run
        if weights is None:
            raise
        _logger.warning(
            f"could not load the encoder weights '{weights}' of '{encoder_name}' ({exc}); "
            "building the model with random weights."
        )
        model = getattr(smp, arch)(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            **model_kwargs,
        )
    return model


def get_preprocessing_params(cfg):
    """Return the ``(mean, std)`` matching the configured encoder.

    The values are read from the smp encoder settings (the 'config.json' of
    the hub); when the hub is not reachable (offline training) the
    well-known ImageNet statistics are used as a fallback.

    The config can point at a local 'config.json' of the encoder, which is
    then read directly and the hub is not asked at all:

    .. code-block:: yaml

        encoder_name: mit_b0
        encoder_weights: weights/mit_b0.pth
        encoder_config: weights/mit_b0.json
    """
    encoder_name = cfg.get("encoder_name", "resnet34")

    # a local 'config.json' of the encoder; it avoids the download of the
    # config of the hub
    encoder_config = cfg.get("encoder_config", "")
    if encoder_config and os.path.isfile(encoder_config):
        try:
            import json

            with open(encoder_config, "r", encoding="utf-8") as f:
                settings = json.load(f)
            mean = torch.tensor(settings["mean"], dtype=torch.float32).view(1, -1, 1, 1)
            std = torch.tensor(settings["std"], dtype=torch.float32).view(1, -1, 1, 1)
            return mean, std
        except (OSError, ValueError, KeyError) as exc:
            _logger.warning(
                f"could not read the encoder config '{encoder_config}' ({exc}); "
                "falling back to the values of the hub."
            )

    try:
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained="imagenet")
        mean = torch.tensor(params["mean"], dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(params["std"], dtype=torch.float32).view(1, -1, 1, 1)
        return mean, std
    except Exception as exc:  # offline / no cached encoder config
        _logger.warning(
            f"could not fetch the preprocessing params of '{encoder_name}' ({exc}); "
            "falling back to the ImageNet mean/std."
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        return mean, std


def create_seg_loss(cfg):
    """Build the loss function described by the config.

    Config keys:
        loss: ``dice`` / ``jaccard`` / ``focal`` / ``ce`` / ``bce``.
        loss_mode: ``multiclass`` (default) or ``binary``.
    """
    loss_name = (cfg.get("loss", "dice") or "dice").lower()
    mode = cfg.get("loss_mode", smp.losses.MULTICLASS_MODE)
    from_logits = bool(cfg.get("from_logits", True))
    ignore_index = cfg.get("ignore_index", 255)
    smooth = float(cfg.get("loss_smooth", 0.0))

    if loss_name == "dice":
        return smp.losses.DiceLoss(mode, from_logits=from_logits, smooth=smooth, ignore_index=ignore_index)
    if loss_name == "jaccard":
        return smp.losses.JaccardLoss(mode, from_logits=from_logits, smooth=smooth, ignore_index=ignore_index)
    if loss_name == "focal":
        return smp.losses.FocalLoss(mode, from_logits=from_logits, ignore_index=ignore_index)
    if loss_name == "ce":
        return torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    if loss_name == "bce":
        return smp.losses.BCEWithLogitsLoss(ignore_index=ignore_index)
    raise ValueError(f"unsupported loss '{loss_name}'")


def create_seg_optimizer(cfg, model):
    """Build the optimizer described by the config."""
    opt_name = (cfg.get("optimizer", "adamw") or "adamw").lower()
    lr = float(cfg.get("lr", 2e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    momentum = float(cfg.get("momentum", 0.9))

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
        )
    raise ValueError(f"unsupported optimizer '{opt_name}'")


def create_scheduler(cfg, optimizer, steps_per_epoch):
    """Build the LR scheduler described by the config.

    Config keys:
        scheduler: ``cosine`` / ``step`` / ``onecycle`` / ``none``.
        min_lr: lower bound used by the cosine schedule.
    """
    name = (cfg.get("scheduler", "cosine") or "cosine").lower()
    epochs = int(cfg.get("epochs", 100))

    if name in ("none", "constant"):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=float(cfg.get("min_lr", 0.0))
        )
    if name == "step":
        milestones = cfg.get("lr_milestones", [int(epochs * 0.6), int(epochs * 0.8)])
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr_max(cfg),
            total_steps=max(1, epochs * steps_per_epoch),
        )
    raise ValueError(f"unsupported scheduler '{name}'")


def lr_max(cfg):
    return float(cfg.get("lr", 2e-4))
