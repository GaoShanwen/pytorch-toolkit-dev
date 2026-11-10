######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: set_parse.py
# function: load yaml config and merge cli overrides.
######################################################
import os
import copy
import logging
import argparse
import yaml

_logger = logging.getLogger("set_parse")


def load_config(config_path):
    """Load a yaml config file and return it as a plain dict."""
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    return cfg


def deep_update(base, override):
    """Recursively merge ``override`` into ``base`` and return ``base``."""
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def apply_overrides(cfg, overrides):
    """Apply ``key.subkey=value`` style overrides onto the config.

    A value is parsed as int / float / bool when possible, otherwise kept as a
    string.  Dotted keys address nested config sections, e.g.
    ``data.root=dataset/seg``.
    """
    cfg = copy.deepcopy(cfg or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"invalid override (expect key=value): {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        parsed = _parse_scalar(value)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = parsed
    return cfg


def _parse_scalar(value):
    """Convert a string into int / float / bool / None when it looks like one."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return [_parse_scalar(v.strip()) for v in value.split(",") if v.strip()]
    return value


def parse_args(description="Train/Evaluate a segmentation model", argv=None):
    """Parse ``--config`` + ``--options`` style arguments.

    Args:
        description: the text of the help of the tool.
        argv: the arguments to parse; ``sys.argv`` is used when it is None.

    Returns ``(cfg, args_namespace, args_text)`` where ``cfg`` is the merged
    yaml config dict and ``args_namespace`` holds the cli-only options.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="", help="path of the yaml config file")
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help="override config entries, e.g. --options epochs=10 data.root=dataset/seg",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="checkpoint to load (evaluate / finetune)",
    )
    parser.add_argument(
        "--wandb",
        dest="log_wandb",
        action="store_true",
        default=None,
        help="enable the wandb logger (overrides 'log_wandb' of the config)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.options)

    # the cli flag has the priority over the config
    if args.log_wandb is not None:
        cfg["log_wandb"] = args.log_wandb

    args_text = yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)
    return cfg, args, args_text


def save_config(cfg, path):
    """Dump the resolved config next to the produced checkpoints."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path
